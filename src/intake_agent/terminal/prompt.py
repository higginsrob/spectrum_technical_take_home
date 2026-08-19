"""Interactive ❯ prompt: Return submits, Shift+Return inserts a newline."""

from __future__ import annotations

import os
import select
import shutil
import signal
import sys
from dataclasses import dataclass
from typing import IO, Any, TextIO

from intake_agent.terminal.style import BRIGHT_MAGENTA, RESET
from intake_agent.terminal.wrap import display_width

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

CLEAR_DOWN = "\033[J"
# Ask the terminal to distinguish Shift+Return from Return (kitty CSI u + xterm).
_KITTY_KB_PUSH = "\x1b[>1;1u"
_KITTY_KB_POP = "\x1b[<u"
_MODIFY_OTHER_ON = "\x1b[>4;1m"
_MODIFY_OTHER_OFF = "\x1b[>4;0m"
_BRACKETED_PASTE_ON = "\x1b[?2004h"
_BRACKETED_PASTE_OFF = "\x1b[?2004l"
_ENTER_KEYS = {"13", "57414"}


@dataclass(frozen=True)
class Key:
    kind: str
    text: str = ""


class PromptBuffer:
    """In-memory prompt editor. Cursor is a code-point index into text."""

    def __init__(self) -> None:
        self.text = ""
        self.cursor = 0

    def insert(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text:
            return
        self.text = self.text[: self.cursor] + text + self.text[self.cursor :]
        self.cursor += len(text)

    def newline(self) -> None:
        self.insert("\n")

    def backspace(self) -> None:
        if self.cursor <= 0:
            return
        self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
        self.cursor -= 1

    def delete(self) -> None:
        if self.cursor >= len(self.text):
            return
        self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]

    def left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def right(self) -> None:
        if self.cursor < len(self.text):
            self.cursor += 1

    def home(self) -> None:
        self.cursor = self.text.rfind("\n", 0, self.cursor) + 1

    def end(self) -> None:
        found = self.text.find("\n", self.cursor)
        self.cursor = len(self.text) if found < 0 else found

    def up(self) -> None:
        start = self.text.rfind("\n", 0, self.cursor) + 1
        col = self.cursor - start
        if start == 0:
            self.cursor = 0
            return
        prev_end = start - 1
        prev_start = self.text.rfind("\n", 0, prev_end) + 1
        self.cursor = min(prev_start + col, prev_end)

    def down(self) -> None:
        start = self.text.rfind("\n", 0, self.cursor) + 1
        col = self.cursor - start
        line_end = self.text.find("\n", self.cursor)
        if line_end < 0:
            self.cursor = len(self.text)
            return
        next_start = line_end + 1
        next_end = self.text.find("\n", next_start)
        if next_end < 0:
            next_end = len(self.text)
        self.cursor = min(next_start + col, next_end)


def _mod_bits(mods: int) -> int:
    return max(0, mods - 1)


def _has_shift(mods: int) -> bool:
    return bool(_mod_bits(mods) & 1)


def _has_ctrl(mods: int) -> bool:
    return bool(_mod_bits(mods) & 4)


def _key_from_code(key: str, mods: int) -> Key:
    """Interpret a CSI-u / modifyOtherKeys code so Return still submits."""
    ctrl = _has_ctrl(mods)
    shift = _has_shift(mods)
    if key in {"3", "99", "67"} and (ctrl or key == "3"):
        return Key("interrupt")
    if key in {"4", "100", "68"} and (ctrl or key == "4"):
        return Key("eof")
    if (key in {"10", "106", "74"} and ctrl) or (key in _ENTER_KEYS and shift):
        return Key("newline")
    if key in _ENTER_KEYS or key == "10":
        return Key("submit")
    if key in {"127", "8"}:
        return Key("backspace")
    return Key("ignore")


def _csi_params(data: bytes) -> tuple[list[str], bytes] | None:
    if not data.startswith(b"\x1b[") or len(data) < 3:
        return None
    final = data[-1:]
    if not (0x40 <= final[0] <= 0x7E):
        return None
    body = data[2:-1].decode("ascii", errors="replace")
    params = body.split(";") if body else []
    return params, final


def decode_key(data: bytes) -> Key:
    """Map a raw key read (or paste payload) onto an editor action."""
    if not data:
        return Key("ignore")
    if data.startswith(b"\x1b[200~") and data.endswith(b"\x1b[201~"):
        inner = data[6:-6].decode("utf-8", errors="replace")
        return Key("paste", inner)
    if data in {b"\r", b"\n", b"\x1bOM"}:
        return Key("submit")
    if data in {b"\x7f", b"\x08"}:
        return Key("backspace")
    if data == b"\x04":
        return Key("eof")
    if data == b"\x03":
        return Key("interrupt")
    if data == b"\x01":
        return Key("home")
    if data == b"\x05":
        return Key("end")
    parsed = _csi_params(data)
    if parsed is not None:
        return _decode_csi(*parsed)
    if data in {b"\x1b[A", b"\x1bOA"}:
        return Key("up")
    if data in {b"\x1b[B", b"\x1bOB"}:
        return Key("down")
    if data in {b"\x1b[C", b"\x1bOC"}:
        return Key("right")
    if data in {b"\x1b[D", b"\x1bOD"}:
        return Key("left")
    if data in {b"\x1b[H", b"\x1b[1~", b"\x1bOH"}:
        return Key("home")
    if data in {b"\x1b[F", b"\x1b[4~", b"\x1b[8~", b"\x1bOF"}:
        return Key("end")
    if data == b"\x1b[3~":
        return Key("delete")
    if data[0] >= 0x20 and data[0] != 0x7F and not data.startswith(b"\x1b"):
        return Key("text", data.decode("utf-8", errors="replace"))
    return Key("ignore")


def _decode_csi(params: list[str], final: bytes) -> Key:
    if final == b"A":
        return Key("up")
    if final == b"B":
        return Key("down")
    if final == b"C":
        return Key("right")
    if final == b"D":
        return Key("left")
    if final == b"H":
        return Key("home")
    if final == b"F":
        return Key("end")
    if final == b"~":
        if params and params[0] == "3":
            return Key("delete")
        if params and params[0] in {"1", "7"}:
            return Key("home")
        if params and params[0] in {"4", "8"}:
            return Key("end")
        key_code = params[0].split(":")[0] if params else ""
        if len(params) >= 3 and key_code == "27":
            try:
                mods = int(params[1].split(":")[0] or "1")
                key = params[2].split(":")[0]
            except ValueError:
                return Key("ignore")
            return _key_from_code(key, mods)
        if key_code in _ENTER_KEYS or key_code == "10":
            try:
                mods = int(params[1].split(":")[0] or "1") if len(params) > 1 else 1
            except ValueError:
                mods = 1
            return _key_from_code(key_code, mods)
    if final == b"u" and params:
        key = params[0].split(":")[0]
        try:
            mods = int(params[1].split(":")[0] or "1") if len(params) > 1 else 1
        except ValueError:
            mods = 1
        return _key_from_code(key, mods)
    return Key("ignore")


def apply_key(buf: PromptBuffer, key: Key) -> str | None:
    """Apply an editor key. Returns 'submit', 'eof', 'interrupt', or None to keep editing."""
    if key.kind == "submit":
        return "submit"
    if key.kind == "eof":
        return "eof" if not buf.text.strip() else "submit"
    if key.kind == "interrupt":
        return "interrupt"
    if key.kind == "newline" or key.kind == "paste":
        buf.insert("\n" if key.kind == "newline" else key.text)
        return None
    action = {
        "text": lambda: buf.insert(key.text),
        "backspace": buf.backspace,
        "delete": buf.delete,
        "left": buf.left,
        "right": buf.right,
        "home": buf.home,
        "end": buf.end,
        "up": buf.up,
        "down": buf.down,
    }.get(key.kind)
    if action is not None:
        action()
    return None


def _chunk(text: str, avail: int) -> list[str]:
    avail = max(1, avail)
    chunks: list[str] = []
    current: list[str] = []
    width = 0
    for char in text:
        char_w = max(1, display_width(char))
        if current and width + char_w > avail:
            chunks.append("".join(current))
            current = [char]
            width = char_w
        else:
            current.append(char)
            width += char_w
    if current or not chunks:
        chunks.append("".join(current))
    return chunks


def render_rows(text: str, styled_prompt: str, glyph: str, width: int) -> list[str]:
    prefix_w = display_width(glyph)
    avail = max(1, width - prefix_w)
    indent = " " * prefix_w
    rows: list[str] = []
    for index, line in enumerate(text.split("\n")):
        prefix = styled_prompt if index == 0 else indent
        pieces = _chunk(line, avail)
        rows.append(prefix + pieces[0])
        for piece in pieces[1:]:
            rows.append(indent + piece)
    return rows


def cursor_row_col(text: str, cursor: int, glyph: str, width: int) -> tuple[int, int]:
    prefix_w = display_width(glyph)
    avail = max(1, width - prefix_w)
    row = 0
    col = prefix_w
    used = 0
    for index, char in enumerate(text):
        if index == cursor:
            return row, col
        if char == "\n":
            row += 1
            col = prefix_w
            used = 0
            continue
        char_w = max(1, display_width(char))
        if used and used + char_w > avail:
            row += 1
            col = prefix_w
            used = 0
        col += char_w
        used += char_w
    return row, col


def read_user_prompt(glyph: str, *, fallback_prompt: str) -> str:
    """Read one user turn. TTY: Shift+Return newline, Return submits."""
    if not sys.stdin.isatty() or not sys.stdout.isatty() or termios is None or tty is None:
        return input(fallback_prompt)
    return _read_tty_prompt(glyph)


def _term_width() -> int:
    try:
        return max(20, shutil.get_terminal_size().columns)
    except OSError:
        return 80


def _wait(fd: int, timeout: float) -> bool:
    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (InterruptedError, ValueError, OSError):
        return False
    return bool(ready)


def _read_byte(fd: int) -> bytes:
    try:
        return os.read(fd, 1)
    except (InterruptedError, OSError):
        return b""


def _read_utf8(fd: int, first: bytes) -> bytes:
    lead = first[0]
    if lead < 0x80:
        need = 1
    elif lead < 0xE0:
        need = 2
    elif lead < 0xF0:
        need = 3
    else:
        need = 4
    buf = bytearray(first)
    while len(buf) < need:
        if not _wait(fd, 0.05):
            break
        nxt = _read_byte(fd)
        if not nxt:
            break
        buf.extend(nxt)
    return bytes(buf)


def _read_csi(fd: int) -> bytes:
    buf = bytearray(b"\x1b[")
    while len(buf) < 64:
        if not _wait(fd, 0.05):
            break
        nxt = _read_byte(fd)
        if not nxt:
            break
        buf.extend(nxt)
        if 0x40 <= nxt[0] <= 0x7E:
            break
    return bytes(buf)


def _read_paste_body(fd: int) -> bytes:
    buf = bytearray()
    end = b"\x1b[201~"
    while len(buf) < 1_000_000:
        try:
            ready, _, _ = select.select([fd], [], [], 0.5)
        except (ValueError, OSError):
            break
        if not ready:
            break
        chunk = os.read(fd, 1024)
        if not chunk:
            break
        buf.extend(chunk)
        idx = buf.find(end)
        if idx >= 0:
            return b"\x1b[200~" + bytes(buf[: idx + len(end)])
    return b"\x1b[200~" + bytes(buf) + end


def _read_key(fd: int) -> bytes:
    first = _read_byte(fd)
    if not first:
        return b""
    if first[0] != 0x1B:
        if first[0] < 0x80:
            return first
        return _read_utf8(fd, first)
    if not _wait(fd, 0.05):
        return first
    nxt = _read_byte(fd)
    if not nxt:
        return first
    if nxt == b"[":
        seq = _read_csi(fd)
        if seq == b"\x1b[200~":
            return _read_paste_body(fd)
        return seq
    if nxt == b"O":
        rest = _read_byte(fd) if _wait(fd, 0.05) else b""
        return first + nxt + rest
    return first + nxt


def _set_prompt_tty(fd: int) -> None:
    """cbreak without CR→NL. Otherwise Return arrives as \\n and inserts a line."""
    mode = termios.tcgetattr(fd)
    mode[tty.IFLAG] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR | termios.IXON)
    mode[tty.LFLAG] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN)
    mode[tty.CC][termios.VMIN] = 1
    mode[tty.CC][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSADRAIN, mode)


def _set_keyboard_modes(stream: TextIO | IO[str], enabled: bool) -> None:
    if enabled:
        stream.write(_KITTY_KB_PUSH + _MODIFY_OTHER_ON + _BRACKETED_PASTE_ON)
    else:
        stream.write(_KITTY_KB_POP + _MODIFY_OTHER_OFF + _BRACKETED_PASTE_OFF)
    stream.flush()


def _paint(
    stream: TextIO | IO[str],
    rows: list[str],
    cursor_row: int,
    cursor_col: int,
    prev_rows: int,
) -> int:
    if prev_rows > 1:
        stream.write(f"\033[{prev_rows - 1}A")
    stream.write("\r" + CLEAR_DOWN)
    stream.write("\n".join(rows))
    last = len(rows) - 1
    up = last - cursor_row
    if up > 0:
        stream.write(f"\033[{up}A")
    stream.write(f"\r\033[{cursor_col + 1}G")
    stream.flush()
    return max(1, len(rows))


def _read_tty_prompt(glyph: str) -> str:
    fd = sys.stdin.fileno()
    stream = sys.stdout
    styled = f"{BRIGHT_MAGENTA}{glyph}{RESET}"
    buf = PromptBuffer()
    saved = termios.tcgetattr(fd)
    prev_winch: Any = None
    painted = 1
    resized = {"flag": False}

    def _on_winch(_signum: int, _frame: Any) -> None:
        resized["flag"] = True

    def _repaint() -> None:
        nonlocal painted
        width = _term_width()
        rows = render_rows(buf.text, styled, glyph, width)
        row, col = cursor_row_col(buf.text, buf.cursor, glyph, width)
        painted = _paint(stream, rows, row, col, painted)

    try:
        _set_prompt_tty(fd)
        _set_keyboard_modes(stream, True)
        if hasattr(signal, "SIGWINCH"):
            prev_winch = signal.getsignal(signal.SIGWINCH)
            signal.signal(signal.SIGWINCH, _on_winch)
        _repaint()
        while True:
            if resized["flag"]:
                resized["flag"] = False
                _repaint()
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except InterruptedError:
                continue
            except (ValueError, OSError):
                break
            if resized["flag"]:
                resized["flag"] = False
                _repaint()
            if not ready:
                continue
            key = decode_key(_read_key(fd))
            result = apply_key(buf, key)
            if result == "submit":
                buf.cursor = len(buf.text)
                _repaint()
                stream.write("\n")
                stream.flush()
                return buf.text
            if result == "eof":
                stream.write("\n")
                stream.flush()
                raise EOFError
            if result == "interrupt":
                raise KeyboardInterrupt
            if key.kind != "ignore":
                _repaint()
    except KeyboardInterrupt:
        try:
            stream.write("\n")
            stream.flush()
        except OSError:
            pass
        raise
    finally:
        if hasattr(signal, "SIGWINCH") and prev_winch is not None:
            signal.signal(signal.SIGWINCH, prev_winch)
        try:
            _set_keyboard_modes(stream, False)
        except OSError:
            pass
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except termios.error:
            pass
    raise EOFError
