"""Braille spinner throbber with a status label."""

from __future__ import annotations

import sys
import threading
import time
from typing import IO, TextIO

from intake_agent.terminal.style import BRIGHT_YELLOW, RESET


class Throbber:
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, stream: TextIO | IO[str] | None = None, interval: float = 0.2):
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self.status = "loading..."
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0
        self._lock = threading.Lock()

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._draw()
        self._thread.start()

    def _line(self) -> str:
        with self._lock:
            frame = self.frames[self._frame % len(self.frames)]
            status = self.status
        return f"{BRIGHT_YELLOW}{frame} {status}{RESET}"

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                self._frame += 1
            self._draw()

    def _draw(self) -> None:
        line = self._line()
        try:
            self.stream.write(f"\r{line}\033[K")
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=0.5)
        try:
            self.stream.write("\r\033[K")
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass
