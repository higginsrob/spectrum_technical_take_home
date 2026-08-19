"""Abort the in-flight assistant turn from Escape (or Ctrl-C as SIGINT)."""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


def is_abort_key(data: bytes) -> bool:
    """Lone Escape or Ctrl-C byte. Arrow keys (`\\x1b[A`) are not abort."""
    return data in {b"\x1b", b"\x03"}


def _drain_fd(fd: int) -> None:
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 0)
        except (ValueError, OSError):
            return
        if not ready:
            return
        try:
            chunk = os.read(fd, 1024)
        except OSError:
            return
        if not chunk:
            return


def _read_key(fd: int) -> bytes:
    try:
        first = os.read(fd, 1)
    except OSError:
        return b""
    if first != b"\x1b":
        return first
    try:
        ready, _, _ = select.select([fd], [], [], 0.05)
    except (ValueError, OSError):
        return first
    if not ready:
        return first
    try:
        rest = os.read(fd, 31)
    except OSError:
        return first
    return first + rest


class _EscapeWatcher:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved: list[Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if termios is None or tty is None:
            return
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._saved is not None and termios is not None:
            _drain_fd(self._fd)
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except termios.error:
                pass
            self._saved = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=0.5)

    def _run(self) -> None:
        fd = self._fd
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.05)
            except (ValueError, OSError):
                return
            if self._stop.is_set() or not ready:
                continue
            if is_abort_key(_read_key(fd)) and not self._stop.is_set():
                os.kill(os.getpid(), signal.SIGINT)
                return


@contextmanager
def watch_escape_abort(enabled: bool) -> Iterator[None]:
    """While a turn runs, Escape raises SIGINT in the main thread (same as Ctrl-C)."""
    if not enabled or not sys.stdin.isatty() or termios is None:
        yield
        return
    watcher = _EscapeWatcher()
    watcher.start()
    try:
        yield
    finally:
        watcher.stop()
