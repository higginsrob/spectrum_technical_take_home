"""Golden-script eval harness: replay examples/ and score the emitted ticket.

Default pytest stays no-LLM (the scorer and case inventory). Live runs:

    python -m intake_agent.eval
    pytest --eval -q
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from intake_agent.graph import build_graph
from intake_agent.llm import LlmConfigError
from intake_agent.schemas import ROUTING_BY_TIER


def load_script(path: Path) -> list[str]:
    """User utterances from a --script file. Skip blanks and # comments."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def examples_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.getenv("INTAKE_EXAMPLES_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    candidate = repo / "examples"
    if candidate.is_dir():
        return candidate
    return Path.cwd() / "examples"


@dataclass(frozen=True)
class EvalCase:
    """Expected ticket for one examples/*.txt script."""

    script: str
    tier: int
    customer_name: str
    account_number: str
    issue_contains: tuple[str, ...]
    category_contains: tuple[str, ...] = ()
    impact_scope: str | None = None
    urgency: str | None = None
    forbid: tuple[str, ...] = ()
    name_not: tuple[str, ...] = ()
    min_turns: int = 3
    must_complete: bool = True

    @property
    def routing_team(self) -> str:
        return ROUTING_BY_TIER[self.tier]


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        script="greeting_then_wifi.txt",
        tier=2,
        customer_name="Alex Kim",
        account_number="44556677",
        issue_contains=("wifi",),
        category_contains=("trouble",),
        min_turns=3,
    ),
    EvalCase(
        script="tier1_password_reset.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("password",),
    ),
    EvalCase(
        script="tier2_billing_dispute.txt",
        tier=2,
        customer_name="Priya Shah",
        account_number="44556677",
        issue_contains=("bill",),
        category_contains=("bill",),
    ),
    EvalCase(
        script="tier2_speed_upgrade.txt",
        tier=2,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("upgrade",),
    ),
    EvalCase(
        script="correction_name.txt",
        tier=1,
        customer_name="Jane Doe-Chen",
        account_number="44556677",
        issue_contains=("password",),
        name_not=("Jane Smith",),
    ),
    EvalCase(
        script="jailbreak_redirect.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("password",),
        min_turns=3,
    ),
    EvalCase(
        script="refuse_password_share.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("log",),
        forbid=("hunter2-secret", "hunter2"),
    ),
    EvalCase(
        script="tier3_outage.txt",
        tier=3,
        customer_name="Carlos Mendoza",
        account_number="99887766",
        issue_contains=("downtown",),
        category_contains=("outage",),
        impact_scope="region",
        urgency="critical",
        min_turns=5,
    ),
    EvalCase(
        script="tier3_account_compromised.txt",
        tier=3,
        customer_name="Carlos Mendoza",
        account_number="99887766",
        issue_contains=("hack",),
        category_contains=("security",),
        impact_scope="single_user",
        urgency="critical",
        min_turns=5,
    ),
    EvalCase(
        script="reclassify_billing_to_outage.txt",
        tier=3,
        customer_name="Priya Shah",
        account_number="44556677",
        issue_contains=("neighborhood",),
        impact_scope="region",
        urgency="critical",
        min_turns=5,
    ),
    EvalCase(
        script="tier_down_outage_to_password.txt",
        tier=1,
        customer_name="Jane Doe",
        account_number="44556677",
        issue_contains=("password",),
    ),
)


@dataclass
class ReplayResult:
    complete: bool
    turns: int
    ticket: dict[str, Any] | None
    replies: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    case: EvalCase
    ok: bool
    failures: list[str]
    replay: ReplayResult | None = None
    error: str | None = None


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_norm(item) for item in value)
    return " ".join(str(value).strip().lower().split())


def _ticket_blob(ticket: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in ticket.values():
        parts.append(_norm(value))
    return " ".join(parts)


def _contains_all(haystack: str, needles: Sequence[str]) -> list[str]:
    missing: list[str] = []
    blob = _norm(haystack)
    for needle in needles:
        if _norm(needle) not in blob:
            missing.append(needle)
    return missing


def score_replay(case: EvalCase, replay: ReplayResult) -> list[str]:
    """Compare a replayed ticket to the golden case. Empty list = pass."""
    failures: list[str] = []
    if replay.turns < case.min_turns:
        failures.append(f"min_turns {case.min_turns}, got {replay.turns}")
    if case.must_complete and not replay.complete:
        failures.append("script ended before the ticket was complete")
        return failures
    if not case.must_complete:
        return failures
    ticket = replay.ticket or {}
    if ticket.get("tier") != case.tier:
        failures.append(f"tier {case.tier}, got {ticket.get('tier')}")
    if ticket.get("routing_team") != case.routing_team:
        failures.append(
            f"routing_team {case.routing_team}, got {ticket.get('routing_team')}"
        )
    if _norm(ticket.get("customer_name")) != _norm(case.customer_name):
        failures.append(
            f"customer_name {case.customer_name!r}, got {ticket.get('customer_name')!r}"
        )
    if str(ticket.get("account_number") or "") != case.account_number:
        failures.append(
            f"account_number {case.account_number}, got {ticket.get('account_number')}"
        )
    for missing in _contains_all(str(ticket.get("issue_summary") or ""), case.issue_contains):
        failures.append(f"issue_summary missing {missing!r}")
    if case.category_contains:
        for missing in _contains_all(str(ticket.get("category") or ""), case.category_contains):
            failures.append(f"category missing {missing!r}")
    if case.impact_scope and _norm(ticket.get("impact_scope")) != _norm(case.impact_scope):
        failures.append(f"impact_scope {case.impact_scope}, got {ticket.get('impact_scope')}")
    if case.urgency and _norm(ticket.get("urgency")) != _norm(case.urgency):
        failures.append(f"urgency {case.urgency}, got {ticket.get('urgency')}")
    got_name = _norm(ticket.get("customer_name"))
    for banned in case.name_not:
        if _norm(banned) and _norm(banned) == got_name:
            failures.append(f"customer_name should not be {banned!r}")
    blob = _ticket_blob(ticket)
    for token in case.forbid:
        if _norm(token) and _norm(token) in blob:
            failures.append(f"ticket stored forbidden {token!r}")
    if ticket.get("status") != "ready_for_routing":
        failures.append(f"status ready_for_routing, got {ticket.get('status')}")
    return failures


def _last_ai_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages") or []):
        role = getattr(message, "type", None)
        if role == "ai" or message.__class__.__name__ == "AIMessage":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def replay_script(graph: Any, lines: Sequence[str]) -> ReplayResult:
    """Feed user turns into a compiled graph until emit or the script ends."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    replies: list[str] = []
    ticket: dict[str, Any] | None = None
    complete = False
    turns = 0
    for user in lines:
        turns += 1
        state = graph.invoke({"messages": [HumanMessage(content=user)]}, config)
        replies.append(_last_ai_text(state))
        if state.get("is_complete") and state.get("ticket"):
            ticket = dict(state["ticket"])
            complete = True
            break
    return ReplayResult(
        complete=complete,
        turns=turns,
        ticket=ticket,
        replies=replies,
    )


def run_case(graph: Any, case: EvalCase, root: Path) -> CaseScore:
    path = root / case.script
    if not path.is_file():
        return CaseScore(
            case=case,
            ok=False,
            failures=[f"missing script {path}"],
        )
    try:
        replay = replay_script(graph, load_script(path))
    except Exception as exc:  # noqa: BLE001
        return CaseScore(case=case, ok=False, failures=[], error=str(exc))
    failures = score_replay(case, replay)
    return CaseScore(
        case=case,
        ok=not failures,
        failures=failures,
        replay=replay,
    )


def _select_cases(only: str | None) -> tuple[EvalCase, ...]:
    if not only:
        return CASES
    key = only.lower().strip()
    picked = tuple(
        case
        for case in CASES
        if key in case.script.lower() or key in Path(case.script).stem.lower()
    )
    return picked


def _print_report(scores: Sequence[CaseScore]) -> None:
    name_w = max((len(Path(s.case.script).stem) for s in scores), default=8)
    header = f"{'CASE':<{name_w}}  RESULT  TIER  TURNS  NOTES"
    print(header)
    print("-" * len(header))
    for score in scores:
        stem = Path(score.case.script).stem
        result = "PASS" if score.ok else "FAIL"
        replay = score.replay
        tier = ""
        turns = ""
        if replay is not None:
            turns = str(replay.turns)
            if replay.ticket:
                tier = str(replay.ticket.get("tier") or "")
        notes = score.error or "; ".join(score.failures)
        print(f"{stem:<{name_w}}  {result:<6}  {tier:<4}  {turns:<5}  {notes}")
    passed = sum(1 for s in scores if s.ok)
    print()
    print(f"{passed}/{len(scores)} passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay examples/ against the live graph and score tickets."
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        help="Directory of .txt scripts (default: repo examples/).",
    )
    parser.add_argument(
        "--only",
        help="Substring filter on the script name (e.g. wifi, jailbreak).",
    )
    args = parser.parse_args(argv)
    picked = _select_cases(args.only)
    if not picked:
        print(f"No eval cases matched {args.only!r}.", file=sys.stderr)
        return 2
    root = examples_dir(args.examples_dir)
    try:
        graph = build_graph()
    except LlmConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    scores = [run_case(graph, case, root) for case in picked]
    _print_report(scores)
    return 0 if all(score.ok for score in scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
