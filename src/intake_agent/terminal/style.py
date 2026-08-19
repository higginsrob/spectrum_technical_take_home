"""Styled text spans and ANSI SGR helpers."""

from __future__ import annotations

from dataclasses import dataclass

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_WHITE = "\033[97m"
REVERSE = "\033[7m"


@dataclass(frozen=True)
class Style:
    bold: bool = False
    italic: bool = False
    code: bool = False
    quote: bool = False
    header: int = 0
    table_header: bool = False
    table_sep: bool = False


@dataclass(frozen=True)
class Span:
    text: str
    style: Style = Style()


def merge_style(base: Style, **overrides: object) -> Style:
    data = {
        "bold": base.bold,
        "italic": base.italic,
        "code": base.code,
        "quote": base.quote,
        "header": base.header,
        "table_header": base.table_header,
        "table_sep": base.table_sep,
    }
    data.update(overrides)
    return Style(**data)  # type: ignore[arg-type]


def ansi_prefix(style: Style) -> str:
    codes: list[str] = []
    if style.header == 1:
        codes.extend([BOLD, BRIGHT_CYAN])
    elif style.header >= 2:
        codes.extend([BOLD, CYAN])
    if style.bold:
        codes.append(BOLD)
    if style.italic:
        codes.append(ITALIC)
    if style.code:
        codes.extend([CYAN, REVERSE])
    if style.quote:
        codes.append(DIM)
    if style.table_header:
        codes.append(BOLD)
    if style.table_sep:
        codes.append(DIM)
    if not codes:
        return ""
    return "".join(codes)


def paint_span(span: Span) -> str:
    if not span.text:
        return ""
    prefix = ansi_prefix(span.style)
    if not prefix:
        return span.text
    return f"{prefix}{span.text}{RESET}"
