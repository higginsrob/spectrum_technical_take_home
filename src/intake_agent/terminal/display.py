"""Assistant block: throbber, then streaming markdown with SIGWINCH reflow."""

from __future__ import annotations

import shutil
import signal
import sys
from typing import IO, Any, TextIO

from intake_agent.terminal.markdown import render_markdown
from intake_agent.terminal.think import (
    count_tokens,
    merge_think,
    merge_visible,
    message_visible_and_think,
    strip_think_tags,
)
from intake_agent.terminal.throbber import Throbber

CLEAR_DOWN = "\033[J"


class DisplayState:
    """Pure display state — no TTY. Used by AssistantDisplay and tests."""

    def __init__(self) -> None:
        self.phase = "loading"  # loading | thinking | body
        self.source = ""
        self.think_text = ""
        self.status = "loading..."

    def on_think(self, incoming: str) -> None:
        if self.phase == "body" or not incoming:
            return
        self.think_text = merge_think(self.think_text, incoming)
        self.phase = "thinking"
        n = count_tokens(self.think_text)
        self.status = f"thinking {n} tokens"

    def on_text(self, incoming: str) -> None:
        visible, tagged = strip_think_tags(incoming)
        if tagged and self.phase != "body":
            self.on_think(tagged)
        if not visible:
            return
        self.phase = "body"
        self.source = merge_visible(self.source, visible)

    def on_message(self, message: Any, *, visible: bool) -> None:
        text, think = message_visible_and_think(message)
        if think:
            self.on_think(think)
        if visible and text:
            self.on_text(text)


class AssistantDisplay:
    def __init__(
        self,
        stream: TextIO | IO[str] | None = None,
        *,
        use_throbber: bool = True,
        use_live: bool = True,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.use_throbber = use_throbber
        self.use_live = use_live
        self.state = DisplayState()
        self._throbber: Throbber | None = None
        self._printed_lines = 0
        self._prev_winch: Any = None
        self._active = False

    def begin(self) -> None:
        self.state = DisplayState()
        self._printed_lines = 0
        self._active = True
        if self.use_throbber:
            self._throbber = Throbber(self.stream)
            self._throbber.start()
        if self.use_live:
            self._install_winch()

    def on_think(self, incoming: str) -> None:
        self.state.on_think(incoming)
        if self.state.phase == "thinking" and self._throbber is not None:
            self._throbber.set_status(self.state.status)

    def on_text(self, incoming: str) -> None:
        was_body = self.state.phase == "body"
        self.state.on_text(incoming)
        if self.state.phase != "body":
            if self._throbber is not None:
                self._throbber.set_status(self.state.status)
            return
        if not was_body:
            self._stop_throbber()
        if self.use_live:
            self.repaint()

    def on_message(self, message: Any, *, visible: bool) -> None:
        text, think = message_visible_and_think(message)
        if think:
            self.on_think(think)
        if visible and text:
            self.on_text(text)

    def width(self) -> int:
        try:
            return max(20, shutil.get_terminal_size().columns)
        except OSError:
            return 80

    def rendered_lines(self, width: int | None = None) -> list[str]:
        return render_markdown(self.state.source, width or self.width())

    def repaint(self) -> None:
        lines = self.rendered_lines()
        if self._printed_lines > 1:
            self.stream.write(f"\033[{self._printed_lines - 1}A")
        self.stream.write("\r" + CLEAR_DOWN)
        body = "\n".join(lines)
        self.stream.write(body)
        self.stream.flush()
        self._printed_lines = max(1, len(lines))

    def finish(self) -> None:
        self._remove_winch()
        self._stop_throbber()
        if self.state.phase == "body":
            if self.use_live:
                if self._printed_lines:
                    self.stream.write("\n")
            else:
                body = "\n".join(self.rendered_lines())
                if body:
                    self.stream.write(body)
                    self.stream.write("\n")
        self.stream.flush()
        self._active = False

    def _stop_throbber(self) -> None:
        if self._throbber is not None:
            self._throbber.stop()
            self._throbber = None

    def _install_winch(self) -> None:
        if not hasattr(signal, "SIGWINCH"):
            return
        self._prev_winch = signal.getsignal(signal.SIGWINCH)

        def _on_winch(_signum: int, _frame: Any) -> None:
            if self._active and self.state.phase == "body":
                self.repaint()

        signal.signal(signal.SIGWINCH, _on_winch)

    def _remove_winch(self) -> None:
        if not hasattr(signal, "SIGWINCH"):
            return
        if self._prev_winch is not None:
            signal.signal(signal.SIGWINCH, self._prev_winch)
            self._prev_winch = None
