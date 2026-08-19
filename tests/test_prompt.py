"""Multiline ❯ prompt: Shift+Return inserts a newline; Return submits."""

from intake_agent.terminal.prompt import (
    Key,
    PromptBuffer,
    apply_key,
    cursor_row_col,
    decode_key,
    read_user_prompt,
    render_rows,
)
from intake_agent.terminal.style import BRIGHT_MAGENTA, RESET


def test_return_submits_shift_return_inserts_newline():
    assert decode_key(b"\r").kind == "submit"
    assert decode_key(b"\n").kind == "submit"
    assert decode_key(b"\x1b[13;2u").kind == "newline"
    assert decode_key(b"\x1b[27;2;13~").kind == "newline"
    assert decode_key(b"\x1b[13;2~").kind == "newline"
    assert decode_key(b"\x1b[13u").kind == "submit"
    assert decode_key(b"\x1b[27;1;13~").kind == "submit"
    assert decode_key(b"\x1b[27;5;10~").kind == "newline"


def test_ctrl_c_and_ctrl_d_from_raw_and_csi():
    assert decode_key(b"\x03").kind == "interrupt"
    assert decode_key(b"\x04").kind == "eof"
    assert decode_key(b"\x1b[27;5;3~").kind == "interrupt"
    assert decode_key(b"\x1b[27;5;99~").kind == "interrupt"
    assert decode_key(b"\x1b[99;5u").kind == "interrupt"
    assert decode_key(b"\x1b[27;5;4~").kind == "eof"
    assert decode_key(b"\x1b[27;5;100~").kind == "eof"
    assert decode_key(b"\x1b[100;5u").kind == "eof"


def test_decode_key_editing_and_paste():
    assert decode_key(b"\x7f").kind == "backspace"
    assert decode_key(b"\x1b[3~").kind == "delete"
    assert decode_key(b"\x1b[D").kind == "left"
    assert decode_key(b"h") == Key("text", "h")
    paste = decode_key(b"\x1b[200~one\r\ntwo\x1b[201~")
    assert paste.kind == "paste"
    assert paste.text == "one\r\ntwo"


def test_buffer_shift_return_then_backspace_joins_lines():
    buf = PromptBuffer()
    apply_key(buf, Key("text", "hello"))
    apply_key(buf, Key("newline"))
    apply_key(buf, Key("text", "world"))
    assert buf.text == "hello\nworld"
    apply_key(buf, Key("home"))
    apply_key(buf, Key("backspace"))
    assert buf.text == "helloworld"
    assert buf.cursor == 5


def test_apply_key_submit_and_eof():
    buf = PromptBuffer()
    assert apply_key(buf, Key("eof")) == "eof"
    apply_key(buf, Key("newline"))
    assert apply_key(buf, Key("eof")) == "eof"
    apply_key(buf, Key("text", "x"))
    assert apply_key(buf, Key("eof")) == "submit"
    assert apply_key(buf, Key("submit")) == "submit"


def test_paste_normalizes_line_endings_without_submitting():
    buf = PromptBuffer()
    assert apply_key(buf, Key("paste", "a\r\nb\rc")) is None
    assert buf.text == "a\nb\nc"


def test_render_rows_indent_continuation_lines():
    styled = f"{BRIGHT_MAGENTA}❯ {RESET}"
    rows = render_rows("hello\nworld", styled, "❯ ", 80)
    assert rows[0].endswith("hello")
    assert rows[1] == "  world"


def test_cursor_moves_to_continuation_line_after_newline():
    assert cursor_row_col("hello\n", 6, "❯ ", 80) == (1, 2)
    assert cursor_row_col("hello", 5, "❯ ", 80) == (0, 7)


def test_up_down_keep_column_between_lines():
    buf = PromptBuffer()
    buf.insert("aaa\nbb")
    buf.up()
    assert buf.cursor == 2
    buf.down()
    assert buf.cursor == len(buf.text)


def test_read_user_prompt_falls_back_to_input_when_not_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: "typed")
    assert read_user_prompt("❯ ", fallback_prompt="x") == "typed"
