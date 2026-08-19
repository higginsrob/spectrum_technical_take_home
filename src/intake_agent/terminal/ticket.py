"""Human-readable ticket report for the end of an intake session."""

from __future__ import annotations

from typing import Any

from intake_agent.terminal.style import (
    BOLD,
    BRIGHT_CYAN,
    BRIGHT_WHITE,
    BRIGHT_YELLOW,
    DIM,
    RESET,
)

TICKET_LABELS: tuple[tuple[str, str], ...] = (
    ("tier", "Tier"),
    ("routing_team", "Routing team"),
    ("status", "Status"),
    ("customer_name", "Customer name"),
    ("account_number", "Account number"),
    ("issue_summary", "Issue summary"),
    ("category", "Category"),
    ("steps_already_tried", "Steps already tried"),
    ("impact_scope", "Impact scope"),
    ("urgency", "Urgency"),
    ("affected_systems", "Affected systems"),
    ("classification_reasoning", "Classification"),
)

SUBMIT_NOTE = (
    "Your request has been submitted. Please keep an eye out for an email "
    "to keep track of your request."
)


def format_ticket_value(value: Any) -> str | None:
    """Plain display value: no quotes, braces, or brackets. None if the row should hide."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.replace("_", " ").strip()
        return text or None
    if isinstance(value, list):
        parts = [format_ticket_value(item) for item in value]
        joined = ", ".join(part for part in parts if part)
        return joined or None
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def ticket_rows(ticket: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in TICKET_LABELS:
        rendered = format_ticket_value(ticket.get(key))
        if rendered is None:
            continue
        rows.append((label, rendered))
    return rows


def format_ticket_report(ticket: dict[str, Any], *, color: bool = True) -> str:
    """Aligned label / value report, then the submitted-request note."""
    rows = ticket_rows(ticket)
    width = max((len(label) for label, _ in rows), default=8)
    key_sgr = f"{BOLD}{BRIGHT_CYAN}" if color else ""
    val_sgr = f"{BRIGHT_YELLOW}" if color else ""
    title_sgr = f"{BOLD}{BRIGHT_WHITE}" if color else ""
    note_sgr = f"{DIM}" if color else ""
    reset = RESET if color else ""

    lines = [f"{title_sgr}Support ticket{reset}"]
    rule = "─" * max(width + 4, 16)
    lines.append(f"{DIM}{rule}{reset}" if color else rule)
    for label, value in rows:
        padded = f"{label}:".ljust(width + 2)
        lines.append(f"{key_sgr}{padded}{reset}{val_sgr}{value}{reset}")
    lines.append("")
    lines.append(f"{note_sgr}{SUBMIT_NOTE}{reset}")
    lines.append("")
    return "\n".join(lines)
