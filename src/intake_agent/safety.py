"""Deterministic user-safety scanners. The LLM does not own these gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from intake_agent.rules import looks_like_support_issue

# High-precision phrases only — this is a take-home rail, not a moderator.
_JAILBREAK_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+"
    r"(instructions|rules|prompts?)|"
    r"dump\s+(your\s+)?(system\s+)?prompt|"
    r"you\s+are\s+now\s+dan|"
    r"\bjailbreak\b|"
    r"pretend\s+you\s+(are\s+not|have\s+no)\s+(rules|restrictions|guidelines)",
    re.IGNORECASE,
)

_CRISIS_RE = re.compile(
    r"\b("
    r"suicid(?:e|al)|"
    r"kill\s+my\s*self|"
    r"killing\s+my\s*self|"
    r"want\s+to\s+die|"
    r"end\s+my\s+life|"
    r"self[-\s]?harm|"
    r"hurt\s+my\s*self"
    r")\b",
    re.IGNORECASE,
)

_PASSWORD_RE = re.compile(
    r"(?:my\s+)?(?:password|passwd|pwd|pin)\s*(?:is|:|=)\s+(\S+)",
    re.IGNORECASE,
)
_SSN_RE = re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")
_PAN_RE = re.compile(r"\b((?:\d[ \-]?){13,19})\b")

_REDACTED = "[redacted]"


@dataclass(frozen=True)
class SafetySignal:
    jailbreak: bool = False
    secret: bool = False
    crisis: bool = False
    secret_values: tuple[str, ...] = ()


RespondPhase = Literal["greeting", "steer", "intake", "safety"]


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _looks_like_pan(raw: str) -> bool:
    digits = _digits_only(raw)
    return 13 <= len(digits) <= 19


def scan_user_text(text: str) -> SafetySignal:
    """Scan the latest user turn for jailbreak, secrets, and crisis language."""
    blob = text or ""
    secrets: list[str] = []
    for match in _PASSWORD_RE.finditer(blob):
        token = match.group(1).strip().rstrip(".,;:!?")
        if token and token.lower() not in {"reset", "wrong", "incorrect"}:
            secrets.append(token)
    for match in _SSN_RE.finditer(blob):
        secrets.append(match.group(1))
    for match in _PAN_RE.finditer(blob):
        raw = match.group(1)
        if _looks_like_pan(raw):
            secrets.append(raw.strip())
            digits = _digits_only(raw)
            if digits != raw.strip():
                secrets.append(digits)

    unique_secrets = tuple(dict.fromkeys(secrets))
    return SafetySignal(
        jailbreak=bool(_JAILBREAK_RE.search(blob)),
        secret=bool(unique_secrets),
        crisis=bool(_CRISIS_RE.search(blob)),
        secret_values=unique_secrets,
    )


def redact_value(value: Any, secret_values: tuple[str, ...]) -> Any:
    """Remove secret tokens from a slot. Whole-value secrets become None."""
    if not secret_values or value is None:
        return value
    if isinstance(value, list):
        cleaned = [redact_value(item, secret_values) for item in value]
        return [item for item in cleaned if item not in (None, "", _REDACTED)]
    if not isinstance(value, str):
        return value
    redacted = value
    for secret in secret_values:
        if not secret:
            continue
        redacted = redacted.replace(secret, _REDACTED)
    if redacted.strip() in {"", _REDACTED}:
        return None
    # Collapse leftover "is [redacted]" noise when the field was only a secret.
    stripped = re.sub(r"\s+", " ", redacted).strip()
    return stripped or None


def apply_safety_to_extraction(
    extracted: dict[str, Any],
    signal: SafetySignal,
) -> dict[str, Any]:
    """Drop crisis-as-issue and strip secrets before slots are merged."""
    cleaned = dict(extracted)
    if signal.crisis:
        cleaned["issue_summary"] = None
        cleaned["category"] = None
    if signal.secret_values:
        for key, value in list(cleaned.items()):
            cleaned[key] = redact_value(value, signal.secret_values)
    return cleaned


_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|yo|good\s+(?:morning|afternoon|evening|day))"
    r"(?:\s+there)?[\s,.!?]*$",
    re.IGNORECASE,
)


def looks_like_greeting(text: str) -> bool:
    """True for a hello-only turn. Jailbreaks and real requests do not match."""
    return bool(_GREETING_RE.match(text or ""))


_COURTESY_RE = re.compile(
    r"^\s*(?:(?:ok|okay|oh)\s+)?"
    r"(?:thanks|thank\s+you|thx|ty|appreciate\s+it)"
    r"(?:\s+(?:so\s+much|a\s+lot))?"
    r"[\s,.!?]*$",
    re.IGNORECASE,
)


def looks_like_courtesy(text: str) -> bool:
    """True for thanks-only small talk. Not an issue and not off-topic."""
    return bool(_COURTESY_RE.match(text or ""))


_SCOPE_NEEDLES = (
    "what can you help",
    "how can you help",
    "what do you help",
    "what kind of problem",
    "what kind of issue",
    "what problems can i",
    "what issues can i",
    "what can i report",
)


def looks_like_scope_question(text: str) -> bool:
    """True for 'what can you help with' — not an issue, not off-topic."""
    blob = " ".join((text or "").lower().split())
    return any(needle in blob for needle in _SCOPE_NEEDLES)


_BRIEF_ANSWER_RE = re.compile(
    r"^\s*(y|n|yes|no|yeah|yep|yup|nah|nope|ok|okay|sure|idk|dunno|"
    r"not sure|that'?s it|thats it|nothing else|nothing|none|n/?a)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def looks_like_brief_answer(text: str) -> bool:
    """Yes/no/nope/that's it — slot answers, never off-topic chat."""
    return bool(_BRIEF_ANSWER_RE.match(text or ""))


def overlay_safety(
    extracted_off_topic: bool,
    signal: SafetySignal,
    *,
    user_text: str = "",
    filled_slots: bool = False,
) -> dict[str, bool]:
    """Combine model off_topic with Python rails.

    Jailbreaks always win. Greetings, courtesy (thanks), brief slot answers,
    capability questions, and turns that filled an intake slot are never
    off-topic on their own.
    """
    if signal.jailbreak:
        off_topic = True
    elif (
        filled_slots
        or looks_like_greeting(user_text)
        or looks_like_courtesy(user_text)
        or looks_like_scope_question(user_text)
        or looks_like_brief_answer(user_text)
        or looks_like_support_issue(user_text)
    ):
        off_topic = False
    else:
        off_topic = bool(extracted_off_topic)
    return {
        "off_topic": off_topic,
        "safety_crisis": signal.crisis,
        "safety_secret": signal.secret,
    }


def _has_assistant_reply(state: dict[str, Any]) -> bool:
    """True once the agent has already spoken in this thread."""
    for message in state.get("messages") or []:
        if getattr(message, "type", None) == "ai":
            return True
    return False


def respond_phase(state: dict[str, Any]) -> RespondPhase:
    """Python-computed reply mode so the model is not guessing the tone.

    ``greeting`` is first-reply only. Later turns with no issue yet are
    ``steer`` so the model cannot keep re-introducing itself.
    """
    if (
        state.get("safety_crisis")
        or state.get("safety_secret")
        or state.get("off_topic")
    ):
        return "safety"
    issue = state.get("issue_summary")
    if issue is None or (isinstance(issue, str) and not issue.strip()):
        if _has_assistant_reply(state):
            return "steer"
        return "greeting"
    return "intake"
