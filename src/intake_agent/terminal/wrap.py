"""Word-boundary wrap of styled spans."""

from __future__ import annotations

import unicodedata

from intake_agent.terminal.style import Span, Style


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _split_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    mode: str | None = None  # "space" | "word"
    for char in text:
        is_space = char.isspace() and char != "\n"
        next_mode = "space" if is_space else "word"
        if char == "\n":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append("\n")
            mode = None
            continue
        if mode is None:
            mode = next_mode
            buf = [char]
            continue
        if next_mode == mode:
            buf.append(char)
        else:
            tokens.append("".join(buf))
            buf = [char]
            mode = next_mode
    if buf:
        tokens.append("".join(buf))
    return tokens


def _hard_break(text: str, width: int) -> list[str]:
    if width < 1:
        width = 1
    pieces: list[str] = []
    current: list[str] = []
    current_w = 0
    for char in text:
        cw = display_width(char)
        if current_w + cw > width and current:
            pieces.append("".join(current))
            current = [char]
            current_w = cw
        else:
            current.append(char)
            current_w += cw
    if current:
        pieces.append("".join(current))
    return pieces or [""]


def wrap_spans(spans: list[Span], width: int) -> list[list[Span]]:
    """Wrap one logical line into visual rows. Breaks on whitespace.

    A single word longer than `width` is hard-split; other words stay intact.
    """
    if width < 1:
        width = 1
    rows: list[list[Span]] = []
    row: list[Span] = []
    row_width = 0

    def flush() -> None:
        nonlocal row, row_width
        if row:
            while row and row[-1].text.isspace():
                row.pop()
            if row:
                rows.append(row)
        row = []
        row_width = 0

    def add_piece(text: str, style: Style) -> None:
        nonlocal row_width
        if not text:
            return
        row.append(Span(text, style))
        row_width += display_width(text)

    for span in spans:
        for token in _split_tokens(span.text):
            if token == "\n":
                flush()
                continue
            is_space = token.isspace()
            tw = display_width(token)
            if is_space and row_width == 0:
                continue
            if row_width + tw <= width:
                add_piece(token, span.style)
                continue
            if row_width == 0:
                pieces = _hard_break(token, width)
                for i, piece in enumerate(pieces):
                    add_piece(piece, span.style)
                    if i < len(pieces) - 1:
                        flush()
                continue
            flush()
            if is_space:
                continue
            if tw <= width:
                add_piece(token, span.style)
            else:
                pieces = _hard_break(token, width)
                for i, piece in enumerate(pieces):
                    add_piece(piece, span.style)
                    if i < len(pieces) - 1:
                        flush()
    flush()
    return rows or [[]]
