"""Slash-command parser and formatters for the interactive CLI.

A line is a command when it starts with `/`, or when it is a single word that
matches a known alias (`quit`, `clear`, `status`, `save`, `q`, …). Unknown
slash lines stay in the command system so they are not sent to the model.
Unknown bare words (`hello`) are not commands.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from intake_agent.rules import SLOT_FIELDS, collected_slots

EXIT_ALIASES = frozenset({"exit", "quit", "e", "x", "q"})
CLEAR_ALIASES = frozenset({"clear", "c"})
STATUS_ALIASES = frozenset({"status"})
SAVE_ALIASES = frozenset({"save"})

_CANONICAL: dict[str, str] = {
    **{name: "exit" for name in EXIT_ALIASES},
    **{name: "clear" for name in CLEAR_ALIASES},
    **{name: "status" for name in STATUS_ALIASES},
    **{name: "save" for name in SAVE_ALIASES},
}

COMMAND_HELP = (
    "Commands: /exit /quit /clear /status /save\n"
    "Single-word aliases: exit, quit, e, x, q, clear, c, status, save"
)


@dataclass(frozen=True)
class Command:
    name: str  # exit | clear | status | save | unknown
    args: str
    raw: str


def parse_command(text: str) -> Command | None:
    """Return a Command if this line should not go to the intake graph."""
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("/"):
        body = stripped[1:].lstrip()
        if not body:
            return Command("unknown", "", stripped)
        token, _, rest = body.partition(" ")
        name = _CANONICAL.get(token.strip().lower(), "unknown")
        return Command(name, rest.strip(), stripped)
    if any(ch.isspace() for ch in stripped):
        return None
    resolved = _CANONICAL.get(stripped.lower())
    if resolved is None:
        return None
    return Command(resolved, "", stripped)


def unknown_command_message(raw: str) -> str:
    token = raw.strip().split(None, 1)[0] if raw.strip() else raw
    return f"Unknown command: {token}\n{COMMAND_HELP}"


def clear_terminal(stream: TextIO | None = None) -> None:
    """Clear the screen (and scrollback when the terminal supports it)."""
    out = stream if stream is not None else sys.stdout
    if not hasattr(out, "isatty") or not out.isatty():
        return
    out.write("\033[2J\033[3J\033[H")
    out.flush()


def default_save_path(now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(f"intake_chat_{stamp}.json")


def resolve_save_path(args: str, *, now: datetime | None = None) -> Path:
    if args.strip():
        return Path(args.strip()).expanduser()
    path = default_save_path(now)
    if path.exists():
        return path.with_stem(f"{path.stem}_{uuid.uuid4().hex[:6]}")
    return path


def save_chat_log(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def serialize_message(message: Any) -> dict[str, str]:
    role = str(getattr(message, "type", None) or message.__class__.__name__)
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = str(content)
    return {"role": role, "content": content}


def snapshot_chat(values: dict[str, Any] | None, thread_id: str) -> dict[str, Any]:
    """Flatten graph state into JSON-friendly chat metadata."""
    state = dict(values or {})
    messages = [serialize_message(message) for message in (state.get("messages") or [])]
    return {
        "thread_id": thread_id,
        "turns": sum(1 for message in messages if message["role"] == "human"),
        "tier": state.get("tier"),
        "is_complete": bool(state.get("is_complete")),
        "missing_fields": list(state.get("missing_fields") or []),
        "slots": collected_slots(state),
        "classification_reasoning": state.get("classification_reasoning") or "",
        "off_topic": bool(state.get("off_topic")),
        "safety_crisis": bool(state.get("safety_crisis")),
        "safety_secret": bool(state.get("safety_secret")),
        "ticket": state.get("ticket"),
        "messages": messages,
    }


def chat_log_payload(
    snapshot: dict[str, Any],
    *,
    greeting: str,
    last_prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_keys = (
        "tier",
        "is_complete",
        "missing_fields",
        "slots",
        "classification_reasoning",
        "off_topic",
        "safety_crisis",
        "safety_secret",
        "ticket",
        *SLOT_FIELDS,
    )
    slots = snapshot.get("slots") or {}
    state = {key: snapshot.get(key) for key in state_keys if key != "slots"}
    state.update(slots)
    state["slots"] = slots
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": snapshot.get("thread_id"),
        "greeting": greeting,
        "messages": snapshot.get("messages") or [],
        "state": state,
        "last_prompt": _last_prompt_public(last_prompt),
    }


def format_status(
    snapshot: dict[str, Any],
    last_prompt: dict[str, Any] | None,
    windows: dict[str, int | None],
) -> str:
    lines = ["Last prompt"]
    if not last_prompt:
        lines.append("  (none yet)")
    else:
        lines.append(f"  outcome     {last_prompt.get('outcome') or 'unknown'}")
        if last_prompt.get("error"):
            lines.append(f"  error       {_clip(str(last_prompt['error']))}")
        lines.append(f"  user        {_clip(str(last_prompt.get('user') or '')) or '(empty)'}")
        reply = last_prompt.get("reply") or ""
        lines.append(f"  reply       {_clip(str(reply)) or '(none)'}")
        lines.append(f"  complete    {_yes_no(bool(last_prompt.get('is_complete')))}")

    lines.append("")
    lines.append("Chat")
    lines.append(f"  thread      {snapshot.get('thread_id') or '(none)'}")
    lines.append(f"  turns       {int(snapshot.get('turns') or 0)}")
    tier = snapshot.get("tier")
    lines.append(f"  tier        {tier if tier is not None else '(none)'}")
    lines.append(f"  complete    {_yes_no(bool(snapshot.get('is_complete')))}")
    missing = snapshot.get("missing_fields") or []
    lines.append(f"  missing     {', '.join(missing) if missing else '(none)'}")
    slots = snapshot.get("slots") or {}
    if slots:
        for key, value in slots.items():
            lines.append(f"  {key:<22}{_clip(str(value), 88)}")
    else:
        lines.append("  collected   (none)")
    flags = [
        name
        for name, on in (
            ("off_topic", snapshot.get("off_topic")),
            ("crisis", snapshot.get("safety_crisis")),
            ("secret", snapshot.get("safety_secret")),
        )
        if on
    ]
    lines.append(f"  safety      {', '.join(flags) if flags else '(none)'}")

    lines.append("")
    lines.append("Context window")
    agent_used = _prompt_size(last_prompt, role="agent")
    classify_used = _prompt_size(last_prompt, role="classify")
    lines.append(_fmt_ctx("agent", agent_used, windows.get("agent")))
    lines.append(_fmt_ctx("classify", classify_used, windows.get("classify")))
    return "\n".join(lines)


def _last_prompt_public(last_prompt: dict[str, Any] | None) -> dict[str, Any] | None:
    if not last_prompt:
        return None
    return {
        "outcome": last_prompt.get("outcome"),
        "user": last_prompt.get("user"),
        "reply": last_prompt.get("reply"),
        "is_complete": bool(last_prompt.get("is_complete")),
        "error": last_prompt.get("error"),
        "tokens": last_prompt.get("tokens"),
        "usage": last_prompt.get("usage"),
    }


def _prompt_size(last_prompt: dict[str, Any] | None, *, role: str) -> int | None:
    if not last_prompt:
        return None
    usage = last_prompt.get("usage") or {}
    node = "respond" if role == "agent" else "extract"
    observed = usage.get(node) or {}
    if observed.get("input_tokens") is not None:
        return int(observed["input_tokens"])
    if role == "classify":
        classify_usage = usage.get("classify") or {}
        if classify_usage.get("input_tokens") is not None:
            extract = observed.get("input_tokens")
            other = int(classify_usage["input_tokens"])
            if extract is None:
                return other
            return max(int(extract), other)
    tokens = last_prompt.get("tokens") or {}
    if role == "agent":
        value = tokens.get("respond")
        return int(value) if value is not None else None
    extract = tokens.get("extract")
    classify = tokens.get("classify")
    sizes = [int(n) for n in (extract, classify) if n]
    return max(sizes) if sizes else None


def _fmt_ctx(label: str, used: int | None, limit: int | None) -> str:
    if used is None and limit is None:
        return f"  {label:<10} (unknown)"
    if used is None:
        shown = f"{limit:,}" if limit is not None else "unset"
        return f"  {label:<10} num_ctx {shown} (no prompt yet)"
    if limit is None:
        return f"  {label:<10} {used:,} tokens (num_ctx unset)"
    pct = (100.0 * used / limit) if limit else 0.0
    return f"  {label:<10} {used:,} / {limit:,} tokens ({pct:.0f}%)"


def _clip(text: str, limit: int = 160) -> str:
    one = " ".join(text.split())
    if len(one) <= limit:
        return one
    return one[: limit - 1] + "…"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
