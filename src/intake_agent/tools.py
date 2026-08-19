"""Stub tools with clear interfaces. Called from Python, not from a ReAct loop."""

from __future__ import annotations

from typing import Any

_ACCOUNTS: dict[str, dict[str, Any]] = {
    "44556677": {
        "found": True,
        "account_number": "44556677",
        "name": "Jane Doe",
        "status": "active",
        "plan": "Internet Gig",
    },
    "99887766": {
        "found": True,
        "account_number": "99887766",
        "name": "Carlos Mendoza",
        "status": "active",
        "plan": "Internet + TV",
    },
}

_KB: list[dict[str, Any]] = [
    {
        "id": "password-reset",
        "title": "Reset Spectrum password",
        "tier_hint": 1,
        "keywords": ("password", "reset", "login", "can't log in", "cant log in"),
        "summary": "Self-service password reset via identity portal. Confirm name and account.",
    },
    {
        "id": "billing-dispute",
        "title": "Billing dispute",
        "tier_hint": 2,
        "keywords": ("bill", "charge", "refund", "dispute"),
        "summary": "Standard support reviews charges and credits. Collect category and steps tried.",
    },
    {
        "id": "outage",
        "title": "Service outage",
        "tier_hint": 3,
        "keywords": ("outage", "down", "no internet", "region"),
        "summary": "Escalate plant/CMTS outages. Collect impact, urgency, and affected systems.",
    },
]


def lookup_account(account_number: str) -> dict[str, Any]:
    """Look up a customer account. Demo data only — not a live billing system."""
    key = (account_number or "").strip()
    hit = _ACCOUNTS.get(key)
    if hit:
        return dict(hit)
    return {"found": False, "account_number": key}


def search_kb(query: str) -> dict[str, Any] | None:
    """Keyword search over a tiny FAQ corpus. Returns the best hit or None."""
    text = (query or "").lower()
    if not text:
        return None
    for article in _KB:
        if any(keyword in text for keyword in article["keywords"]):
            return {
                "id": article["id"],
                "title": article["title"],
                "tier_hint": article["tier_hint"],
                "summary": article["summary"],
            }
    return None
