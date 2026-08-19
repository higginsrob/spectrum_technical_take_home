"""Ticket report formatting — no JSON punctuation."""

from intake_agent.terminal.style import BRIGHT_CYAN, BRIGHT_YELLOW, RESET
from intake_agent.terminal.ticket import (
    SUBMIT_NOTE,
    format_ticket_report,
    format_ticket_value,
    ticket_rows,
)


def _sample_ticket(**overrides):
    ticket = {
        "tier": 2,
        "routing_team": "standard_support",
        "classification_reasoning": "Single-user outage.",
        "customer_name": "Rob Higgins",
        "account_number": "1234445",
        "issue_summary": "Internet went out",
        "category": "outage",
        "steps_already_tried": "restarted equipment",
        "impact_scope": "single_user",
        "urgency": "critical",
        "affected_systems": ["internet", "modem"],
        "status": "ready_for_routing",
    }
    ticket.update(overrides)
    return ticket


def test_format_ticket_value_drops_json_punctuation():
    assert format_ticket_value("standard_support") == "standard support"
    assert format_ticket_value(["internet", "modem"]) == "internet, modem"
    assert format_ticket_value(None) is None
    assert format_ticket_value([]) is None
    assert format_ticket_value(2) == "2"


def test_ticket_rows_skip_empty_optionals():
    rows = dict(
        ticket_rows(
            _sample_ticket(
                category=None,
                steps_already_tried=None,
                impact_scope=None,
                urgency=None,
                affected_systems=None,
                classification_reasoning="",
            )
        )
    )
    assert rows["Customer name"] == "Rob Higgins"
    assert "Category" not in rows
    assert "Affected systems" not in rows


def test_format_ticket_report_is_plain_when_uncolored():
    report = format_ticket_report(_sample_ticket(), color=False)
    assert "{" not in report
    assert "}" not in report
    assert "[" not in report
    assert "]" not in report
    assert '"' not in report
    assert "Customer name:" in report
    assert "Rob Higgins" in report
    assert "internet, modem" in report
    assert "standard support" in report
    assert SUBMIT_NOTE in report
    assert report.index("Support ticket") < report.index(SUBMIT_NOTE)
    assert report.endswith("\n")
    assert report.rstrip().endswith("your request.")


def test_format_ticket_report_colors_label_and_value():
    report = format_ticket_report(_sample_ticket(), color=True)
    assert BRIGHT_CYAN in report
    assert BRIGHT_YELLOW in report
    assert RESET in report
    assert SUBMIT_NOTE in report
