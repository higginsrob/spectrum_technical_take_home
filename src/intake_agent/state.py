"""LangGraph state container. Pydantic is not used here — see schemas.py."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class IntakeState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    customer_name: str | None
    account_number: str | None
    issue_summary: str | None
    category: str | None
    steps_already_tried: str | None
    impact_scope: str | None
    urgency: str | None
    affected_systems: list[str] | None
    tier: int | None
    classification_reasoning: str
    missing_fields: list[str]
    ticket: dict[str, Any] | None
    is_complete: bool
    off_topic: bool
    safety_crisis: bool
    safety_secret: bool
    account_lookup: dict[str, Any] | None
    kb_hit: dict[str, Any] | None
