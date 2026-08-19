"""No-LLM tests for completeness, corrections, and re-tier field changes."""

from intake_agent.rules import (
    apply_answer_to_next_field,
    build_ticket,
    can_classify,
    canonicalize_impact_scope,
    canonicalize_steps,
    canonicalize_urgency,
    fill_implied_slots,
    ground_extraction,
    infer_category,
    is_intake_complete,
    looks_like_support_issue,
    intake_spoken_line,
    merge_extraction,
    missing_fields,
    next_missing_field,
    normalize_extraction,
    with_extracted_slots,
)
from intake_agent.tools import lookup_account, search_kb


def _base(**overrides):
    state = {
        "customer_name": None,
        "account_number": None,
        "issue_summary": None,
        "category": None,
        "steps_already_tried": None,
        "impact_scope": None,
        "urgency": None,
        "affected_systems": None,
        "tier": None,
        "classification_reasoning": "",
    }
    state.update(overrides)
    return state


def test_tier1_incomplete_without_core_fields():
    state = _base(tier=1, customer_name="Jane Doe")
    assert missing_fields(state) == ["account_number", "issue_summary"]
    assert is_intake_complete(state) is False


def test_tier1_complete_with_core_fields():
    state = _base(
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_summary="Password reset",
    )
    assert missing_fields(state) == []
    assert is_intake_complete(state) is True
    ticket = build_ticket(state)
    assert ticket.tier == 1
    assert ticket.routing_team == "self_service"
    assert ticket.status == "ready_for_routing"
    assert ticket.category is None


def test_tier2_requires_category_and_steps():
    state = _base(
        tier=2,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_summary="Unexpected charge on last bill",
    )
    assert missing_fields(state) == ["category", "steps_already_tried"]
    state["category"] = "billing"
    state["steps_already_tried"] = "Reviewed the PDF bill"
    assert is_intake_complete(state) is True
    assert build_ticket(state).routing_team == "standard_support"


def test_tier3_requires_impact_urgency_and_systems():
    state = _base(
        tier=3,
        customer_name="Carlos Mendoza",
        account_number="99887766",
        issue_summary="Regional internet outage downtown",
        category="outage",
        steps_already_tried="Power-cycled modem; checked status page",
    )
    assert missing_fields(state) == ["impact_scope", "urgency", "affected_systems"]
    state["impact_scope"] = "region"
    state["urgency"] = "critical"
    state["affected_systems"] = ["DOCSIS plant", "CMTS"]
    assert is_intake_complete(state) is True
    ticket = build_ticket(state)
    assert ticket.routing_team == "escalation"
    assert ticket.affected_systems == ["DOCSIS plant", "CMTS"]


def test_unclassified_uses_tier1_required_set():
    state = _base(customer_name="Jane")
    assert "account_number" in missing_fields(state)
    assert "category" not in missing_fields(state)
    assert is_intake_complete(state) is False


def test_correction_overwrites_filled_slot():
    state = _base(customer_name="Jane Doe", account_number="111")
    updates = merge_extraction(
        state,
        {"customer_name": "Jane Doe-Chen"},
        is_correction=True,
    )
    assert updates["customer_name"] == "Jane Doe-Chen"
    assert "account_number" not in updates


def test_empty_extraction_does_not_clear_slots():
    state = _base(customer_name="Jane Doe", account_number="44556677")
    updates = merge_extraction(
        state,
        {"customer_name": None, "account_number": "  "},
    )
    assert updates == {}


def test_restated_value_overwrites_without_correction_flag():
    state = _base(account_number="111")
    updates = merge_extraction(state, {"account_number": "44556677"})
    assert updates["account_number"] == "44556677"


def test_tier_up_adds_missing_fields():
    state = _base(
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_summary="Internet is down for the whole block",
    )
    assert is_intake_complete(state) is True
    state["tier"] = 3
    missing = missing_fields(state)
    assert "category" in missing
    assert "steps_already_tried" in missing
    assert "impact_scope" in missing
    assert "urgency" in missing
    assert "affected_systems" in missing
    assert is_intake_complete(state) is False


def test_tier_down_does_not_block_emit():
    state = _base(
        tier=3,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_summary="Forgot password",
        category="account",
        steps_already_tried="Tried the forgot-password link",
        impact_scope="single_user",
        urgency="business_hours",
        affected_systems=["identity portal"],
    )
    assert is_intake_complete(state) is True
    state["tier"] = 1
    assert missing_fields(state) == []
    ticket = build_ticket(state)
    assert ticket.tier == 1
    assert ticket.routing_team == "self_service"
    # Extra T3 slots remain on the payload but are not required.
    assert ticket.impact_scope == "single_user"


def test_next_missing_field_asks_for_issue_first():
    state = _base(tier=2)
    assert next_missing_field(state) == "issue_summary"
    state["issue_summary"] = "Password reset"
    assert next_missing_field(state) == "customer_name"
    state["customer_name"] = "Jane"
    assert next_missing_field(state) == "account_number"


def test_greeting_is_not_classifiable():
    assert can_classify(_base()) is False
    assert can_classify(_base(issue_summary="Forgot password")) is True


def test_crisis_flag_blocks_intake_complete():
    state = _base(
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_summary="Forgot password",
        safety_crisis=True,
    )
    assert is_intake_complete(state) is False


def test_build_ticket_rejects_incomplete_state():
    try:
        build_ticket(_base(tier=1, customer_name="Jane"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lookup_account_known_and_unknown():
    assert lookup_account("44556677")["found"] is True
    assert lookup_account("000")["found"] is False


def test_search_kb_password_hint():
    hit = search_kb("I forgot my password and can't log in")
    assert hit is not None
    assert hit["tier_hint"] == 1
    assert search_kb("unrelated poetry") is None


def test_canonicalize_natural_language_slots():
    assert canonicalize_impact_scope("the whole house") == "single_user"
    assert canonicalize_impact_scope("the whole neighborhood") == "region"
    assert canonicalize_impact_scope("in the area") == "region"
    assert canonicalize_impact_scope("the whole area is down") == "region"
    assert canonicalize_impact_scope("is there an outage in the area?") is None
    assert canonicalize_impact_scope("is there an outage in the area? my internet is out") is None
    assert canonicalize_urgency("so urgent, like red alarm") == "critical"
    assert canonicalize_urgency("10 minutes ago") is None
    assert canonicalize_steps("no I did not") == "none yet"
    assert canonicalize_steps("nope") == "none yet"
    assert canonicalize_steps("restarted the modem") == "restarted the modem"
    assert canonicalize_steps("no, just restart the modem") == "restart the modem"


def test_ground_extraction_drops_guessed_tier3_slots():
    guessed = {
        "issue_summary": "Internet is out",
        "category": "outage",
        "steps_already_tried": "none yet",
        "impact_scope": "region",
        "urgency": "business_hours",
        "affected_systems": ["internet"],
    }
    cleaned = ground_extraction(
        guessed,
        "is there an outage in the area?  my internet is out",
    )
    assert cleaned["impact_scope"] is None
    assert cleaned["urgency"] is None
    assert cleaned["steps_already_tried"] is None
    assert cleaned["affected_systems"] == ["internet"]
    asserted = ground_extraction(
        guessed,
        "There's no internet for most of downtown. We power-cycled the modem. It's critical.",
    )
    assert asserted["impact_scope"] == "region"
    assert asserted["urgency"] == "critical"
    assert "power-cycled" in asserted["steps_already_tried"]


def test_normalize_extraction_maps_messy_json():
    cleaned = normalize_extraction(
        {
            "impact_scope": "whole house",
            "urgency": "red alarm",
            "steps_already_tried": "nope",
            "affected_systems": "modem, TV",
        }
    )
    assert cleaned["impact_scope"] == "single_user"
    assert cleaned["urgency"] == "critical"
    assert cleaned["steps_already_tried"] == "none yet"
    assert cleaned["affected_systems"] == ["modem", "TV"]


def test_apply_answer_fills_asked_field_when_extract_omits_it():
    state = _base(
        tier=3,
        customer_name="Rob Higgins",
        account_number="234567",
        issue_summary="internet went out",
        category="outage",
        steps_already_tried="none yet",
    )
    assert next_missing_field(state) == "impact_scope"
    filled = apply_answer_to_next_field(state, "the whole house", {})
    assert filled["impact_scope"] == "single_user"
    state["impact_scope"] = "single_user"
    filled = apply_answer_to_next_field(state, "so urgent, like red alarm", {})
    assert filled["urgency"] == "critical"
    state["urgency"] = "critical"
    filled = apply_answer_to_next_field(
        state, "not sure, I think all of them", {}
    )
    assert filled["affected_systems"] == ["not sure", "I think all of them"]


def test_apply_answer_does_not_stuff_phone_into_impact():
    state = _base(
        tier=3,
        customer_name="Rob Higgins",
        account_number="234567",
        issue_summary="internet went out",
        category="outage",
        steps_already_tried="none yet",
    )
    assert apply_answer_to_next_field(state, "123-123-1234", {}) == {}


def test_infer_category_from_outage_issue():
    assert infer_category("my internet went out at 1234 west 23rd") == "outage"
    assert infer_category("I forgot my password") == "password_reset"
    assert looks_like_support_issue("my internet went out")
    assert not looks_like_support_issue("hi")


def test_first_message_issue_fills_summary_when_extract_omits_it():
    filled = apply_answer_to_next_field(_base(), "my internet went out", {})
    assert filled["issue_summary"] == "my internet went out"


def test_intake_spoken_line_does_not_reintroduce():
    state = _base(tier=2, issue_summary="my internet went out")
    line = intake_spoken_line(state, user_text="my internet went out")
    assert line is not None
    assert "name" in line.lower()
    assert "Hi" not in line
    assert "Spectrum" not in line
    again = intake_spoken_line(
        state,
        user_text="my internet went out",
        last_reply=line,
    )
    assert again != line
    assert "name" in again.lower()


def test_yes_fills_steps_once_category_is_implied():
    state = _base(
        tier=2,
        customer_name="Rob Higgins",
        account_number="1234445",
        issue_summary="my internet went out",
    )
    extracted = fill_implied_slots(state, {})
    assert extracted["category"] == "outage"
    filled = apply_answer_to_next_field(
        with_extracted_slots(state, extracted),
        "yes",
        extracted,
        last_question="Have you already tried restarting your equipment?",
    )
    assert filled["steps_already_tried"] == "restarted equipment"


def test_yes_to_house_question_is_single_user():
    state = _base(
        tier=3,
        customer_name="Rob Higgins",
        account_number="1234445",
        issue_summary="internet went out",
        category="outage",
        steps_already_tried="restarted equipment",
    )
    filled = apply_answer_to_next_field(
        state,
        "yes",
        {},
        last_question="Is there anyone else in the house who's also losing their connection?",
    )
    assert filled["impact_scope"] == "single_user"
