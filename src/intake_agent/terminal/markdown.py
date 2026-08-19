"""Basic markdown → styled spans. Re-parse the full buffer on every chunk."""

from __future__ import annotations

import re

from intake_agent.terminal.style import Span, Style, merge_style, paint_span
from intake_agent.terminal.wrap import wrap_spans

_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def parse_inline(text: str, base: Style | None = None) -> list[Span]:
    """Parse inline emphasis and code. Unclosed openers apply through EOF (streaming)."""
    return _parse_inline(text, base or Style())


def _parse_inline(text: str, base: Style) -> list[Span]:
    if base.code:
        return [Span(text, base)] if text else []
    spans: list[Span] = []
    i = 0
    n = len(text)
    pending = 0

    markers = (
        ("***", {"bold": True, "italic": True}),
        ("___", {"bold": True, "italic": True}),
        ("**", {"bold": True}),
        ("__", {"bold": True}),
        ("*", {"italic": True}),
        ("_", {"italic": True}),
    )

    def flush(end: int) -> None:
        nonlocal pending
        if end > pending:
            spans.append(Span(text[pending:end], base))
        pending = end

    while i < n:
        if text[i] == "`":
            close = text.find("`", i + 1)
            flush(i)
            if close == -1:
                spans.append(Span(text[i + 1 :], merge_style(base, code=True)))
                return spans
            spans.append(Span(text[i + 1 : close], merge_style(base, code=True)))
            i = close + 1
            pending = i
            continue

        matched = False
        for marker, flags in markers:
            if not text.startswith(marker, i):
                continue
            if marker in {"*", "_"} and i > 0 and text[i - 1].isalnum():
                continue
            close = text.find(marker, i + len(marker))
            flush(i)
            inner = text[i + len(marker) :] if close == -1 else text[i + len(marker) : close]
            spans.extend(_parse_inline(inner, merge_style(base, **flags)))
            if close == -1:
                return spans
            i = close + len(marker)
            pending = i
            matched = True
            break
        if matched:
            continue
        i += 1

    flush(n)
    return spans


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_row(line: str) -> bool:
    return "|" in line and not line.strip().startswith("```")


def _format_table(rows: list[list[str]]) -> list[list[Span]]:
    if not rows:
        return []
    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]
    widths = [0] * col_count
    for r in padded:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    out: list[list[Span]] = []
    sep_style = Style(table_sep=True)
    header_style = Style(table_header=True, bold=True)

    def paint_row(cells: list[str], is_header: bool) -> list[Span]:
        spans: list[Span] = []
        style = header_style if is_header else Style()
        for i, cell in enumerate(cells):
            if i:
                spans.append(Span(" │ ", sep_style))
            spans.append(Span(cell.ljust(widths[i]), style))
        return spans

    body_start = 0
    is_sep = False
    if len(padded) >= 2:
        is_sep = all(
            set((c or "-").replace(":", "").replace(" ", "")) <= {"-"} and "-" in (c or "-")
            for c in padded[1]
        )
    if is_sep:
        out.append(paint_row(padded[0], True))
        sep_cells = ["─" * max(widths[i], 3) for i in range(col_count)]
        sep_spans: list[Span] = []
        for i, cell in enumerate(sep_cells):
            if i:
                sep_spans.append(Span("─┼─", sep_style))
            sep_spans.append(Span(cell, sep_style))
        out.append(sep_spans)
        body_start = 2
    for r in padded[body_start:]:
        out.append(paint_row(r, False))
    return out


def parse_markdown(source: str) -> list[list[Span]]:
    """Parse markdown into logical lines of spans."""
    lines = source.split("\n")
    logical: list[list[Span]] = []
    i = 0
    in_fence = False
    fence_lang = ""
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if table_buf:
            cells = [_split_row(row) for row in table_buf]
            logical.extend(_format_table(cells))
            table_buf = []

    while i < len(lines):
        line = lines[i]
        fence_match = _FENCE.match(line)
        if fence_match and not in_fence:
            flush_table()
            in_fence = True
            fence_lang = fence_match.group(1).strip()
            label = f"```{fence_lang}".rstrip()
            logical.append([Span(label, Style(code=True))])
            i += 1
            continue
        if in_fence:
            if _FENCE.match(line):
                logical.append([Span("```", Style(code=True))])
                in_fence = False
            else:
                logical.append([Span(line, Style(code=True))])
            i += 1
            continue

        if _is_table_row(line):
            table_buf.append(line)
            i += 1
            continue
        flush_table()

        header = _HEADER.match(line)
        if header:
            level = len(header.group(1))
            logical.append(parse_inline(header.group(2), Style(header=level, bold=True)))
            i += 1
            continue

        if line.startswith(">"):
            body = re.sub(r"^>\s?", "", line)
            quote_spans = [Span("│ ", Style(quote=True))]
            quote_spans.extend(parse_inline(body, Style(quote=True, italic=True)))
            logical.append(quote_spans)
            i += 1
            continue

        if line == "":
            logical.append([Span("")])
            i += 1
            continue

        logical.append(parse_inline(line))
        i += 1

    flush_table()
    if in_fence:
        # unclosed fence while streaming — already emitted code lines
        pass
    return logical


def render_markdown(source: str, width: int) -> list[str]:
    """ANSI-colored, word-wrapped lines ready to print."""
    painted: list[str] = []
    for logical in parse_markdown(source):
        rows = wrap_spans(logical, width)
        for row in rows:
            painted.append("".join(paint_span(span) for span in row))
    return painted
