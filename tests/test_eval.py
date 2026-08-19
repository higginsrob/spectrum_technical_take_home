"""Eval harness: scorer and case inventory are no-LLM. Live replay is --eval."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from intake_agent.eval import (
    CASES,
    EvalCase,
    ReplayResult,
    examples_dir,
    load_script,
    replay_script,
    score_replay,
)


def test_load_script_skips_comments_and_blanks(tmp_path: Path):
    path = tmp_path / "demo.txt"
    path.write_text("# header\n\nHi\n# note\nJane Doe\n", encoding="utf-8")
    assert load_script(path) == ["Hi", "Jane Doe"]


def test_every_example_script_has_an_eval_case():
    root = examples_dir()
    scripts = {path.name for path in root.glob("*.txt")}
    covered = {case.script for case in CASES}
    assert scripts == covered, f"uncovered={scripts - covered} extra={covered - scripts}"


def test_cases_point_at_real_files():
    root = examples_dir()
    for case in CASES:
        assert (root / case.script).is_file(), case.script


def _t1_ticket(**overrides):
    ticket = {
        "tier": 1,
        "routing_team": "self_service",
        "classification_reasoning": "password reset",
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "Forgot Spectrum password and cannot log in",
        "category": None,
        "status": "ready_for_routing",
    }
    ticket.update(overrides)
    return ticket


def test_score_replay_passes_matching_ticket():
    case = EvalCase(
        script="tier1_password_reset.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("password",),
        min_turns=3,
    )
    replay = ReplayResult(complete=True, turns=3, ticket=_t1_ticket())
    assert score_replay(case, replay) == []


def test_score_replay_catches_wrong_tier_name_and_secret():
    case = EvalCase(
        script="refuse_password_share.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("log",),
        forbid=("hunter2",),
        min_turns=3,
    )
    replay = ReplayResult(
        complete=True,
        turns=3,
        ticket=_t1_ticket(
            issue_summary="cannot log in, password hunter2",
            customer_name="Jane Smith",
        ),
    )
    failures = score_replay(case, replay)
    assert any("customer_name" in item for item in failures)
    assert any("hunter2" in item for item in failures)


def test_score_replay_rejects_uncorrected_name():
    case = EvalCase(
        script="correction_name.txt",
        tier=1,
        customer_name="Jane Doe-Chen",
        account_number="44556677",
        issue_contains=("password",),
        name_not=("Jane Smith",),
        min_turns=3,
    )
    replay = ReplayResult(
        complete=True,
        turns=4,
        ticket=_t1_ticket(customer_name="Jane Smith"),
    )
    failures = score_replay(case, replay)
    assert any("Jane Doe-Chen" in item or "Jane Smith" in item for item in failures)


def test_score_replay_incomplete_script():
    case = EvalCase(
        script="tier1_password_reset.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("password",),
        min_turns=2,
    )
    replay = ReplayResult(complete=False, turns=2, ticket=None)
    failures = score_replay(case, replay)
    assert any("complete" in item for item in failures)


def test_replay_script_stops_when_complete():
    class _Graph:
        def __init__(self):
            self.calls = 0

        def invoke(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "messages": [HumanMessage(content="hi"), AIMessage(content="Name?")],
                    "is_complete": False,
                }
            return {
                "messages": [AIMessage(content="Routing.")],
                "is_complete": True,
                "ticket": _t1_ticket(),
            }

    graph = _Graph()
    result = replay_script(graph, ["Hi", "Jane Doe", "unused fallback"])
    assert result.complete is True
    assert result.turns == 2
    assert graph.calls == 2
    assert result.ticket["account_number"] == "44556677"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=lambda c: Path(c.script).stem)
def test_live_example_script(case: EvalCase, request):
    if not request.config.getoption("--eval"):
        pytest.skip("live LLM eval; pass --eval")
    from intake_agent.eval import run_case
    from intake_agent.graph import build_graph
    from intake_agent.llm import LlmConfigError

    try:
        graph = build_graph()
    except LlmConfigError as exc:
        pytest.skip(str(exc))
    score = run_case(graph, case, examples_dir())
    assert score.ok, score.error or "; ".join(score.failures)
