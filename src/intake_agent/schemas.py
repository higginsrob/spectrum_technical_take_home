"""Pydantic models for LLM structured output and the final ticket payload."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ImpactScope = Literal["single_user", "multiple", "region"]
Urgency = Literal["business_hours", "after_hours", "critical"]
RoutingTeam = Literal["self_service", "standard_support", "escalation"]


class Extraction(BaseModel):
    """Fields mentioned in the latest user turn. Omit anything not stated."""

    customer_name: str | None = None
    account_number: str | None = None
    issue_summary: str | None = None
    category: str | None = None
    steps_already_tried: str | None = None
    impact_scope: str | None = Field(
        default=None,
        description="single_user, multiple, or region. Natural language is ok; Python normalizes.",
    )
    urgency: str | None = Field(
        default=None,
        description="business_hours, after_hours, or critical. Natural language is ok; Python normalizes.",
    )
    affected_systems: list[str] | None = None
    is_correction: bool = Field(
        default=False,
        description="True if the user is correcting a previously collected value.",
    )
    off_topic: bool = Field(
        default=False,
        description=(
            "True only for jailbreaks or requests unrelated to Spectrum support. "
            "False for greetings (hi, hello) and for answers that fill intake slots."
        ),
    )

    @field_validator("is_correction", "off_topic", mode="before")
    @classmethod
    def _null_bool_is_false(cls, value: object) -> object:
        # Local models emit null for omitted flags; treat that as false.
        return False if value is None else value

    @field_validator("affected_systems", mode="before")
    @classmethod
    def _systems_string_to_list(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return [value]
        return value


class Classification(BaseModel):
    tier: Literal[1, 2, 3] | None = Field(
        default=None,
        description="1, 2, or 3 once a real support issue is known. Null for greetings or insufficient information.",
    )
    reasoning: str = Field(
        default="",
        description="Short justification citing the issue type and the tier definition. Empty if unclassified.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _null_reasoning_is_empty(cls, value: object) -> object:
        return "" if value is None else value


class Ticket(BaseModel):
    tier: Literal[1, 2, 3]
    routing_team: RoutingTeam
    classification_reasoning: str
    customer_name: str
    account_number: str
    issue_summary: str
    category: str | None = None
    steps_already_tried: str | None = None
    impact_scope: str | None = None
    urgency: str | None = None
    affected_systems: list[str] | None = None
    status: str = "ready_for_routing"


ROUTING_BY_TIER: dict[int, RoutingTeam] = {
    1: "self_service",
    2: "standard_support",
    3: "escalation",
}
