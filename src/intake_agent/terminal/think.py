"""Think/reasoning extraction. Visible reply text never includes think blocks."""

from __future__ import annotations

import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENCODING = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, (len(text) + 3) // 4)


def merge_think(previous: str, incoming: str) -> str:
    """Join stream pieces that may be deltas or full snapshots.

    Do not treat 'incoming in previous' as a duplicate — short tokens like
    quotes or `false` would be dropped and corrupt JSON.
    """
    if not incoming:
        return previous
    if not previous:
        return incoming
    if incoming.startswith(previous):
        return incoming
    if previous.startswith(incoming):
        return previous
    return previous + incoming


merge_visible = merge_think


def _block_text(block: Any) -> tuple[str, str]:
    """Return (visible, think) for one content block."""
    if isinstance(block, str):
        return block, ""
    if isinstance(block, dict):
        kind = str(block.get("type") or "")
        text = str(block.get("text") or block.get("content") or block.get("thinking") or "")
        if kind in {"thinking", "reasoning", "think"}:
            return "", text
        return text, ""
    kind = str(getattr(block, "type", "") or "")
    text = str(getattr(block, "text", "") or getattr(block, "content", "") or "")
    if kind in {"thinking", "reasoning", "think"}:
        return "", text
    return text, ""


def message_visible_and_think(message: Any) -> tuple[str, str]:
    """Split a LangChain message/chunk into visible text and think text."""
    think_parts: list[str] = []
    extra = getattr(message, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "thinking"):
            value = extra.get(key)
            if value:
                think_parts.append(value if isinstance(value, str) else str(value))

    meta = getattr(message, "response_metadata", None) or {}
    if isinstance(meta, dict) and meta.get("thinking"):
        think_parts.append(str(meta["thinking"]))

    info = getattr(message, "generation_info", None) or {}
    if isinstance(info, dict) and info.get("thinking"):
        think_parts.append(str(info["thinking"]))

    content = getattr(message, "content", message)
    visible_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            vis, th = _block_text(block)
            if vis:
                visible_parts.append(vis)
            if th:
                think_parts.append(th)
    elif content is None:
        pass
    else:
        visible_parts.append(content if isinstance(content, str) else str(content))

    visible = "".join(visible_parts)
    stripped, tagged = strip_think_tags(visible)
    if tagged:
        think_parts.append(tagged)
    think = "".join(think_parts)
    return stripped, think


def strip_think_tags(text: str) -> tuple[str, str]:
    """Remove complete and in-progress <think> regions. Returns (visible, think)."""
    if not text:
        return "", ""
    think_bits = [match.group(0) for match in _THINK_BLOCK.finditer(text)]
    visible = _THINK_BLOCK.sub("", text)
    open_match = _THINK_OPEN.search(visible)
    if open_match:
        think_bits.append(visible[open_match.start() :])
        visible = visible[: open_match.start()]
    return visible, "".join(think_bits)
