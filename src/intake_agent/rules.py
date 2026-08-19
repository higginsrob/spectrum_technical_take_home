"""Deterministic intake rules: required fields, merge/corrections, ticket build.

The LLM never decides completeness or routing. This module does.
"""

from __future__ import annotations

import re
from typing import Any

from intake_agent.schemas import ROUTING_BY_TIER, Ticket

SLOT_FIELDS: tuple[str, ...] = (
    "customer_name",
    "account_number",
    "issue_summary",
    "category",
    "steps_already_tried",
    "impact_scope",
    "urgency",
    "affected_systems",
)

REQUIRED_BY_TIER: dict[int, tuple[str, ...]] = {
    1: ("customer_name", "account_number", "issue_summary"),
    2: (
        "customer_name",
        "account_number",
        "issue_summary",
        "category",
        "steps_already_tried",
    ),
    3: (
        "customer_name",
        "account_number",
        "issue_summary",
        "category",
        "steps_already_tried",
        "impact_scope",
        "urgency",
        "affected_systems",
    ),
}

FIELD_LABELS: dict[str, str] = {
    "customer_name": "customer name",
    "account_number": "account number",
    "issue_summary": "issue summary",
    "category": "category",
    "steps_already_tried": "steps already tried",
    "impact_scope": "impact scope (single_user, multiple, or region)",
    "urgency": "urgency (business_hours, after_hours, or critical)",
    "affected_systems": "affected systems",
}

FIELD_ASK: dict[str, str] = {
    "customer_name": "the name on the account",
    "account_number": "the account number",
    "issue_summary": "what's going on with their Spectrum service",
    "category": "a short category such as outage, billing, troubleshooting, or password_reset",
    "steps_already_tried": "what they've already tried, or confirm they haven't tried anything yet",
    "impact_scope": (
        "whether this is just their connection (single_user), "
        "several customers (multiple), or a wider area (region)"
    ),
    "urgency": "urgency as business_hours, after_hours, or critical",
    "affected_systems": "which systems are affected (modem, TV, phone, internet, plant, etc.)",
}

FIELD_QUESTIONS: dict[str, tuple[str, ...]] = {
    "customer_name": (
        "What's the name on the account?",
        "Whose name should I put on this?",
        "I still need the name on the account.",
    ),
    "account_number": (
        "What's the account number?",
        "Do you have the account number handy?",
        "I still need the account number.",
    ),
    "issue_summary": (
        "What's going on with the service?",
        "What can I get in front of a team for you?",
    ),
    "category": (
        "Is this more of an outage, billing, or troubleshooting issue?",
        "What kind of issue should I label this — outage, billing, troubleshooting?",
    ),
    "steps_already_tried": (
        "Have you already tried restarting the equipment?",
        "What have you tried so far?",
        "Anything else you've already tried?",
    ),
    "impact_scope": (
        "Is this just your connection, or are neighbors down too?",
        "Is it only your house, several customers, or a wider area?",
    ),
    "urgency": (
        "Is this critical right now, after hours, or can it wait for business hours?",
        "How urgent is this — critical, after hours, or business hours?",
    ),
    "affected_systems": (
        "Is this internet only, or are TV and phone out too?",
        "Which systems are affected — internet, TV, phone, modem?",
    ),
}

_SCOPE_REGION = (
    "region",
    "neighborhood",
    "neighbourhood",
    "downtown",
    "city",
    "town",
    "block",
    "zip",
    "whole area",
    "entire area",
    "wider area",
)
_SCOPE_MULTIPLE = (
    "multiple",
    "several",
    "neighbors",
    "neighbours",
    "building",
    "other customer",
    "other people",
    "apartments",
)
_SCOPE_SINGLE = (
    "house",
    "household",
    "home",
    "just me",
    "just my",
    "only me",
    "only my",
    "single",
    "my connection",
    "this account",
    "this house",
)
_URGENCY_CRITICAL = (
    "critical",
    "emergency",
    "red",
    "alarm",
    "asap",
    "right now",
    "so urgent",
    "very urgent",
    "extremely",
)
_URGENCY_AFTER = (
    "after hours",
    "after-hours",
    "tonight",
    "evening",
    "night",
    "weekend",
)
_URGENCY_BUSINESS = (
    "business hours",
    "business_hours",
    "daytime",
    "whenever",
    "not urgent",
    "normal",
)
_STEPS_NEGATIVE = re.compile(
    r"^(no|nope|nah|none|nothing|not yet)(\b|$)|"
    r"\b(did not|didn't|have not|haven't|not tried|haven't tried)\b",
    re.IGNORECASE,
)
_STEPS_EXCEPT = re.compile(
    r"^(?:no|nope)[,.]?\s+(?:just |only )(.+)$",
    re.IGNORECASE,
)
_STEPS_HINT = re.compile(
    r"\b(no|nope|nah|yes|yeah|yep|tried|reboot|restart|modem|nothing|none|"
    r"reset|cycled|unplugged|power.?cycled)\b|"
    r"that'?s it|nothing else",
    re.IGNORECASE,
)
_SYSTEMS_HINT = re.compile(
    r"\b(modem|router|tv|phone|internet|wifi|wi-fi|plant|cmts|equipment|"
    r"devices?|systems?|all|everything|not sure)\b",
    re.IGNORECASE,
)
_YES_RE = re.compile(r"^\s*(y|yes|yeah|yep|yup|yea)(?:\s*[.!]*)?\s*$", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(n|no|nope|nah)(?:\s*[.!]*)?\s*$", re.IGNORECASE)
_DONE_RE = re.compile(
    r"that'?s it|that is all|that'?s all|nothing else|nothing more",
    re.IGNORECASE,
)


def _mentions(blob: str, tokens: tuple[str, ...]) -> bool:
    for token in tokens:
        if " " in token:
            if token in blob:
                return True
        elif re.search(rf"\b{re.escape(token)}\b", blob):
            return True
    return False


def is_yes(text: str) -> bool:
    return bool(_YES_RE.match((text or "").strip()))


def is_no(text: str) -> bool:
    return bool(_NO_RE.match((text or "").strip()))


def infer_category(issue: str | None) -> str | None:
    blob = (issue or "").lower()
    if not blob.strip():
        return None
    if any(token in blob for token in ("password", "login", "sign in", "can't log", "cannot log")):
        return "password_reset"
    if any(token in blob for token in ("bill", "charge", "credit", "payment")):
        return "billing"
    if any(token in blob for token in ("upgrade", "downgrade", "speed")):
        return "service_modification"
    if any(token in blob for token in ("hack", "compromised", "fraud", "stolen")):
        return "security"
    if any(
        token in blob
        for token in ("outage", "went out", "is down", "no internet", "offline", "neighborhood")
    ):
        return "outage"
    if any(token in blob for token in ("wifi", "wi-fi", "modem", "router", "slow", "connect")):
        return "troubleshooting"
    return None


def looks_like_support_issue(text: str) -> bool:
    """True when the utterance itself is a Spectrum problem, not a hello or slot answer."""
    blob = (text or "").strip()
    if len(blob) < 8:
        return False
    if infer_category(blob):
        return True
    lower = blob.lower()
    tokens = (
        "internet",
        "wifi",
        "wi-fi",
        "outage",
        "password",
        "billing",
        "modem",
        "router",
        "cable",
        "tv",
        "phone",
        "email",
        "charge",
        "slow",
        "down",
        "out",
        "disconnect",
        "login",
    )
    return any(token in lower for token in tokens)


def fill_implied_slots(state: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    """Fill category from the issue when the model never asks for it."""
    out = dict(extracted)
    issue = out.get("issue_summary") if is_filled(out.get("issue_summary")) else state.get("issue_summary")
    if not is_filled(out.get("category")) and not is_filled(state.get("category")):
        category = infer_category(issue if isinstance(issue, str) else None)
        if category:
            out["category"] = category
    return out


def with_extracted_slots(state: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for field in SLOT_FIELDS:
        if is_filled(extracted.get(field)):
            merged[field] = extracted[field]
    return merged


def is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def _short_area_scope_answer(blob: str, original: str) -> bool:
    """'in the area' as a slot answer is region; an outage-check question is not."""
    if "?" in original:
        return False
    if not re.search(r"\barea\b", blob):
        return False
    return len(blob.split()) <= 6


def canonicalize_impact_scope(value: Any) -> str | None:
    if not is_filled(value):
        return None
    text = " ".join(str(value).split())
    key = text.lower().replace("-", "_")
    if key in {"single_user", "multiple", "region"}:
        return key
    blob = key.replace("_", " ")
    if _mentions(blob, _SCOPE_REGION) or _short_area_scope_answer(blob, text):
        return "region"
    if _mentions(blob, _SCOPE_MULTIPLE):
        return "multiple"
    if _mentions(blob, _SCOPE_SINGLE):
        return "single_user"
    return None


def canonicalize_urgency(value: Any) -> str | None:
    if not is_filled(value):
        return None
    text = " ".join(str(value).split())
    key = text.lower().replace("-", "_")
    if key in {"business_hours", "after_hours", "critical"}:
        return key
    blob = key.replace("_", " ")
    if _mentions(blob, _URGENCY_CRITICAL):
        return "critical"
    if _mentions(blob, _URGENCY_AFTER):
        return "after_hours"
    if _mentions(blob, _URGENCY_BUSINESS):
        return "business_hours"
    return None


def canonicalize_steps(value: Any) -> str | None:
    if not is_filled(value):
        return None
    text = " ".join(str(value).split())
    excepted = _STEPS_EXCEPT.match(text)
    if excepted:
        rest = excepted.group(1).strip()
        if rest:
            return rest
    if _STEPS_NEGATIVE.search(text):
        return "none yet"
    if _DONE_RE.search(text):
        return "no further steps"
    return text


def scope_from_yes_no(text: str, last_question: str = "") -> str | None:
    question = (last_question or "").lower()
    if is_yes(text):
        if any(token in question for token in ("neighbor", "neighbour", "area", "region", "downtown", "block")):
            return "region" if any(token in question for token in ("region", "downtown", "neighborhood", "neighbourhood")) else "multiple"
        if any(token in question for token in ("house", "home", "household", "connection")):
            return "single_user"
        return "multiple"
    if is_no(text):
        return "single_user"
    return None


def systems_from_yes_no(text: str, last_question: str = "") -> list[str] | None:
    question = (last_question or "").lower()
    if is_yes(text):
        found = [
            name
            for name in ("modem", "router", "tv", "phone", "internet", "wifi")
            if name in question
        ]
        return found or ["additional services"]
    if is_no(text):
        return ["internet"]
    return None


def canonicalize_affected_systems(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,;/]", value) if part.strip()]
        return parts or None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if is_filled(item)]
        return items or None
    text = str(value).strip()
    return [text] if text else None


def normalize_extraction(extracted: dict[str, Any]) -> dict[str, Any]:
    """Map messy model output onto the allowed slot values."""
    out = dict(extracted)
    if "impact_scope" in out:
        out["impact_scope"] = canonicalize_impact_scope(out.get("impact_scope"))
    if "urgency" in out:
        out["urgency"] = canonicalize_urgency(out.get("urgency"))
    if "steps_already_tried" in out:
        out["steps_already_tried"] = canonicalize_steps(out.get("steps_already_tried"))
    if "affected_systems" in out:
        out["affected_systems"] = canonicalize_affected_systems(out.get("affected_systems"))
    return out


_STEPS_STATED = re.compile(
    r"\b(tried|reboot(?:ed|ing)?|restart(?:ed|ing)?|reset|cycled|unplugged|"
    r"power.?cycled|nothing(?: else)?|none yet|haven't tried|have not tried|"
    r"didn't try|did not try|not yet)\b|"
    r"that'?s it|nothing else",
    re.IGNORECASE,
)


def _has_steps_evidence(text: str) -> bool:
    """True when this utterance itself describes troubleshooting, not yes/no to a prompt."""
    blob = text or ""
    return bool(_STEPS_STATED.search(blob) or _DONE_RE.search(blob))


def _system_item_grounded(item: str, blob: str) -> bool:
    token = " ".join(str(item).lower().split())
    if not token:
        return False
    if token in blob:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", token) if len(part) > 2]
    return bool(parts) and all(
        re.search(rf"\b{re.escape(part)}\b", blob) for part in parts
    )


def ground_extraction(extracted: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Drop T2/T3 slots the model guessed without the user stating them.

    Asking "is there an outage in the area?" is not impact_scope=region.
    Silence is not steps="none yet" or urgency=business_hours.
    """
    out = dict(extracted)
    text = user_text or ""
    blob = text.lower()
    if is_filled(out.get("impact_scope")):
        out["impact_scope"] = canonicalize_impact_scope(text)
    if is_filled(out.get("urgency")):
        out["urgency"] = canonicalize_urgency(text)
    if is_filled(out.get("steps_already_tried")):
        out["steps_already_tried"] = (
            canonicalize_steps(text) if _has_steps_evidence(text) else None
        )
    if is_filled(out.get("affected_systems")):
        items = canonicalize_affected_systems(out.get("affected_systems")) or []
        kept = [item for item in items if _system_item_grounded(item, blob)]
        out["affected_systems"] = kept or None
    return out


def apply_answer_to_next_field(
    state: dict[str, Any],
    user_text: str,
    extracted: dict[str, Any],
    *,
    last_question: str = "",
) -> dict[str, Any]:
    """If extract skipped the field we just asked for, fill it from the utterance."""
    nxt = next_missing_field(state)
    if not nxt or is_filled(extracted.get(nxt)):
        return extracted
    text = (user_text or "").strip()
    if not text:
        return extracted
    value: Any = None
    if nxt == "impact_scope":
        value = canonicalize_impact_scope(text) or scope_from_yes_no(text, last_question)
    elif nxt == "urgency":
        value = canonicalize_urgency(text)
    elif nxt == "steps_already_tried":
        if is_yes(text):
            value = (
                "restarted equipment"
                if "restart" in (last_question or "").lower()
                else "tried suggested steps"
            )
        elif _STEPS_HINT.search(text) or is_no(text) or _DONE_RE.search(text):
            value = canonicalize_steps(text)
    elif nxt == "affected_systems":
        value = canonicalize_affected_systems(text) if _SYSTEMS_HINT.search(text) else None
        if not is_filled(value):
            value = systems_from_yes_no(text, last_question)
        if not is_filled(value) and "not sure" in text.lower():
            value = ["unknown"]
    elif nxt == "account_number":
        digits = re.search(r"\b(\d{5,})\b", text)
        if digits:
            value = digits.group(1)
    elif nxt == "issue_summary":
        if looks_like_support_issue(text):
            value = text
    if not is_filled(value):
        return extracted
    inferred = dict(extracted)
    inferred[nxt] = value
    return inferred


def required_fields_for(tier: int | None) -> tuple[str, ...]:
    if tier is None:
        return REQUIRED_BY_TIER[1]
    return REQUIRED_BY_TIER[int(tier)]


def missing_fields(state: dict[str, Any]) -> list[str]:
    required = required_fields_for(state.get("tier"))
    return [field for field in required if not is_filled(state.get(field))]


def is_intake_complete(state: dict[str, Any]) -> bool:
    if state.get("safety_crisis"):
        return False
    if state.get("tier") not in (1, 2, 3):
        return False
    return len(missing_fields(state)) == 0


def merge_extraction(
    state: dict[str, Any],
    extracted: dict[str, Any],
    *,
    is_correction: bool = False,
) -> dict[str, Any]:
    """Apply newly mentioned slots onto state.

    Filled incoming values overwrite (covers restated fields and corrections).
    Empty incoming values never clear an existing slot unless this turn is an
    explicit correction that includes that field as empty — which we still
    refuse, so slots are only replaced by new filled values.
    """
    del is_correction  # documented hook; filled values already overwrite
    updates: dict[str, Any] = {}
    for field in SLOT_FIELDS:
        incoming = extracted.get(field)
        if is_filled(incoming):
            updates[field] = incoming
    return updates


def can_classify(state: dict[str, Any]) -> bool:
    """Need an issue before asking the model to pick a tier."""
    return is_filled(state.get("issue_summary"))


def next_missing_field(state: dict[str, Any]) -> str | None:
    missing = missing_fields(state)
    if not missing:
        return None
    if "issue_summary" in missing:
        return "issue_summary"
    return missing[0]


def _pick_question(field: str, last_reply: str) -> str:
    options = FIELD_QUESTIONS.get(field) or (FIELD_ASK.get(field, field),)
    last = (last_reply or "").casefold()
    for option in options:
        if option.casefold() not in last:
            return option
    return options[-1]


def intake_spoken_line(
    state: dict[str, Any],
    *,
    user_text: str = "",
    last_reply: str = "",
) -> str | None:
    """Deterministic intake ask — no LLM, so it cannot re-introduce itself."""
    nxt = next_missing_field(state)
    if not nxt:
        return None
    question = _pick_question(nxt, last_reply)
    restated_issue = (
        nxt != "issue_summary"
        and is_filled(state.get("issue_summary"))
        and looks_like_support_issue(user_text)
    )
    if restated_issue:
        return f"I've got that. {question}"
    name = state.get("customer_name")
    if is_filled(name) and nxt != "customer_name" and user_text.strip() and not is_yes(user_text) and not is_no(user_text):
        first = str(name).split()[0]
        return f"Thanks, {first}. {question}"
    return question


def collected_slots(state: dict[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in SLOT_FIELDS if is_filled(state.get(field))}


def build_ticket(state: dict[str, Any]) -> Ticket:
    if not is_intake_complete(state):
        raise ValueError("Cannot emit a ticket until all required fields are present.")
    tier = int(state["tier"])
    return Ticket(
        tier=tier,  # type: ignore[arg-type]
        routing_team=ROUTING_BY_TIER[tier],
        classification_reasoning=state.get("classification_reasoning") or "",
        customer_name=state["customer_name"],
        account_number=state["account_number"],
        issue_summary=state["issue_summary"],
        category=state.get("category"),
        steps_already_tried=state.get("steps_already_tried"),
        impact_scope=state.get("impact_scope"),
        urgency=state.get("urgency"),
        affected_systems=state.get("affected_systems"),
        status="ready_for_routing",
    )
