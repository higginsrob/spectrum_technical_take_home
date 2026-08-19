"""Explicit LangGraph intake turn: extract → classify → validate → respond|emit."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from intake_agent.llm import get_llms
from intake_agent.prompts import (
    CLASSIFY_INSTRUCTIONS,
    EXTRACT_INSTRUCTIONS,
    RESPOND_INSTRUCTIONS,
)
from intake_agent.rules import (
    FIELD_ASK,
    SLOT_FIELDS,
    apply_answer_to_next_field,
    build_ticket,
    can_classify,
    collected_slots,
    fill_implied_slots,
    ground_extraction,
    intake_spoken_line,
    is_filled,
    is_intake_complete,
    merge_extraction,
    missing_fields,
    next_missing_field,
    normalize_extraction,
    with_extracted_slots,
)
from intake_agent.safety import (
    apply_safety_to_extraction,
    overlay_safety,
    respond_phase,
    scan_user_text,
)
from intake_agent.schemas import Classification, Extraction
from intake_agent.state import IntakeState
from intake_agent.terminal.think import (
    count_tokens,
    message_visible_and_think,
    merge_visible,
)
from intake_agent.tools import lookup_account, search_kb

ROUTING_LABELS = {
    1: "Self-Service",
    2: "Standard Support",
    3: "Escalation",
}


def _latest_user_text(state: IntakeState) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _latest_assistant_text(state: IntakeState) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _is_chat_turn(message: Any) -> bool:
    kind = getattr(message, "type", None)
    return kind in {"human", "ai"} or isinstance(message, (HumanMessage, AIMessage))


def conversation_messages(state: IntakeState, *, limit: int = 12) -> list:
    """Recent human/AI turns for the respond model. Extract/classify stay snapshot-only."""
    selected = [
        message for message in (state.get("messages") or []) if _is_chat_turn(message)
    ]
    if limit and len(selected) > limit:
        return selected[-limit:]
    return selected


def respond_control_block(state: IntakeState) -> str:
    nxt = next_missing_field(dict(state))
    missing = state.get("missing_fields") or missing_fields(dict(state))
    last_reply = _latest_assistant_text(state)
    ask = FIELD_ASK.get(nxt, nxt) if nxt else "(none)"
    already = ", ".join(collected_slots(dict(state)).keys()) or "(none)"
    return (
        f"Phase: {respond_phase(dict(state))}\n"
        f"Tier: {state.get('tier')}\n"
        f"Reasoning: {state.get('classification_reasoning')}\n"
        f"Off-topic: {state.get('off_topic', False)}\n"
        f"Safety crisis: {state.get('safety_crisis', False)}\n"
        f"Safety secret: {state.get('safety_secret', False)}\n"
        f"Collected:\n{_slots_block(state)}\n"
        f"Already collected (do not re-ask): {already}\n"
        f"Missing fields: {missing}\n"
        f"Next field: {nxt}\n"
        f"Ask only for: {ask}\n"
        f"Account lookup: {state.get('account_lookup')}\n"
        f"KB hint: {state.get('kb_hit')}\n"
        f"Your last reply (do not repeat or reuse its opener): {last_reply or '(none)'}\n"
        "Never ask for a phone number, service address, callback, or start time.\n"
        "Never say you have routed this, dispatched a tech, or opened a ticket. "
        "Code emits the ticket when fields are complete.\n"
        "If Next field is set, you must ask for that and nothing else.\n"
        "Reply to the customer now. Output only the spoken line."
    )


def _slots_block(state: IntakeState) -> str:
    slots = collected_slots(dict(state))
    if not slots:
        return "(none yet)"
    lines = []
    for key, value in slots.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def prompt_token_counts(state: IntakeState) -> dict[str, int]:
    """Estimate tokens that extract / classify / respond would send this turn."""
    user_text = _latest_user_text(state)
    extract_text = (
        f"{EXTRACT_INSTRUCTIONS}\n"
        f"Collected so far:\n{_slots_block(state)}\n\n"
        f"Latest user message:\n{user_text}"
    )
    counts = {"extract": count_tokens(extract_text), "classify": 0, "respond": 0}
    if can_classify(dict(state)):
        classify_text = (
            f"{CLASSIFY_INSTRUCTIONS}\n"
            f"Issue summary: {state.get('issue_summary')}\n"
            f"Collected slots:\n{_slots_block(state)}\n\n"
            f"Latest user message:\n{user_text}"
        )
        counts["classify"] = count_tokens(classify_text)
    spoken = None
    if respond_phase(dict(state)) == "intake":
        spoken = intake_spoken_line(
            dict(state),
            user_text=user_text,
            last_reply=_latest_assistant_text(state),
        )
    if spoken:
        counts["respond"] = count_tokens(spoken)
    else:
        respond_parts = []
        for message in respond_messages(state):
            respond_parts.append(_message_text(message))
        counts["respond"] = count_tokens("\n".join(respond_parts))
    return counts


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _coerce_json_text(content: Any) -> str:
    if isinstance(content, str):
        text = content.strip()
    else:
        text = json.dumps(content)
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _is_system_message(message: Any) -> bool:
    return isinstance(message, SystemMessage) or getattr(message, "type", None) == "system"


def collapse_leading_system_messages(messages: list) -> list:
    """Qwen 3.x chat templates allow only one system message, and only at index 0.

    Merge consecutive leading SystemMessages so JSON-mode hints can share a turn
    with node instructions. Later system messages are left unchanged.
    """
    if not messages:
        return list(messages)
    leading: list[str] = []
    rest_start = 0
    for i, message in enumerate(messages):
        if not _is_system_message(message):
            break
        text = _message_text(message).strip()
        if text:
            leading.append(text)
        rest_start = i + 1
    if rest_start <= 1:
        return list(messages)
    merged = SystemMessage(content="\n\n".join(leading))
    return [merged, *messages[rest_start:]]


def _concat_stream_text(llm: BaseChatModel, messages: list) -> str:
    messages = collapse_leading_system_messages(messages)
    if not hasattr(llm, "stream"):
        reply = llm.invoke(messages)
        visible, _ = message_visible_and_think(reply)
        return visible
    assembled: Any = None
    for chunk in llm.stream(messages):
        try:
            assembled = chunk if assembled is None else assembled + chunk
        except TypeError:
            visible, _ = message_visible_and_think(chunk)
            assembled_text = (
                message_visible_and_think(assembled)[0] if assembled is not None else ""
            )
            assembled = merge_visible(assembled_text, visible)
    if assembled is None:
        return ""
    if isinstance(assembled, str):
        return assembled
    visible, _ = message_visible_and_think(assembled)
    return visible


def invoke_structured(llm: BaseChatModel, schema: type, messages: list, retries: int = 1):
    """Parse a Pydantic schema from the model.

    OpenAI uses native structured output via tool calling. ``json_schema``
    (ChatOpenAI's default) attaches the Pydantic instance to the OpenAI
    ``parsed`` field, which Pydantic then warns about when LangSmith traces
    the response. Function calling returns the same object without that dump.
    Ollama's tool-calling path often returns empty objects for optional-field
    schemas, so local models are bound with ``format="json"`` and validated
    by Pydantic instead. Ollama calls stream so think tokens can update the
    CLI throbber while JSON stays hidden.
    """
    last_error: Exception | None = None
    provider = llm.__class__.__name__
    attempts = retries + 1

    if provider == "ChatOpenAI":
        structured = llm.with_structured_output(schema, method="function_calling")
        for _ in range(attempts):
            try:
                result = structured.invoke(messages)
                if result is None:
                    raise ValueError("structured output returned None")
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"Failed to parse structured {schema.__name__} from the model: {last_error}"
        ) from last_error

    hint = SystemMessage(
        content=(
            f"Respond with JSON only that matches this schema. "
            f"Use null for unknown optional strings and arrays. "
            f"Booleans must be true or false, never null. Do not wrap in markdown.\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
    )
    payload = collapse_leading_system_messages([hint, *messages])
    json_llm = llm.bind(format="json") if hasattr(llm, "bind") else llm
    for _ in range(attempts):
        try:
            raw_text = _concat_stream_text(json_llm, payload)
            return schema.model_validate_json(_coerce_json_text(raw_text))
        except Exception as exc:  # noqa: BLE001
            if "not found" in str(exc).lower() and "404" in str(exc):
                raise
            last_error = exc
            try:
                raw = json_llm.invoke(payload)
                visible, _ = message_visible_and_think(raw)
                content = visible or (
                    raw.content if isinstance(getattr(raw, "content", None), str) else ""
                )
                return schema.model_validate_json(_coerce_json_text(content))
            except Exception as fallback_exc:  # noqa: BLE001
                last_error = fallback_exc
    raise RuntimeError(
        f"Failed to parse structured {schema.__name__} from the model: {last_error}"
    ) from last_error


def make_extract_node(llm: BaseChatModel):
    def extract_fields(state: IntakeState) -> dict[str, Any]:
        user_text = _latest_user_text(state)
        signal = scan_user_text(user_text)
        extraction: Extraction = invoke_structured(
            llm,
            Extraction,
            [
                SystemMessage(content=EXTRACT_INSTRUCTIONS),
                HumanMessage(
                    content=(
                        f"Collected so far:\n{_slots_block(state)}\n\n"
                        f"Latest user message:\n{user_text}"
                    )
                ),
            ],
        )
        cleaned = apply_safety_to_extraction(extraction.model_dump(), signal)
        cleaned = normalize_extraction(cleaned)
        cleaned = ground_extraction(cleaned, user_text)
        cleaned = fill_implied_slots(dict(state), cleaned)
        cleaned = apply_answer_to_next_field(
            with_extracted_slots(dict(state), cleaned),
            user_text,
            cleaned,
            last_question=_latest_assistant_text(state),
        )
        cleaned = fill_implied_slots(with_extracted_slots(dict(state), cleaned), cleaned)
        filled_slots = any(is_filled(cleaned.get(field)) for field in SLOT_FIELDS)
        updates = merge_extraction(
            dict(state),
            cleaned,
            is_correction=extraction.is_correction,
        )
        updates.update(
            overlay_safety(
                extraction.off_topic,
                signal,
                user_text=user_text,
                filled_slots=filled_slots,
            )
        )
        return updates

    return extract_fields


def make_classify_node(llm: BaseChatModel):
    def classify_tier(state: IntakeState) -> dict[str, Any]:
        if not can_classify(dict(state)):
            return {}
        user_text = _latest_user_text(state)
        classification: Classification = invoke_structured(
            llm,
            Classification,
            [
                SystemMessage(content=CLASSIFY_INSTRUCTIONS),
                HumanMessage(
                    content=(
                        f"Issue summary: {state.get('issue_summary')}\n"
                        f"Collected slots:\n{_slots_block(state)}\n\n"
                        f"Latest user message:\n{user_text}"
                    )
                ),
            ],
        )
        if classification.tier not in (1, 2, 3):
            return {}
        return {
            "tier": int(classification.tier),
            "classification_reasoning": classification.reasoning,
        }

    return classify_tier


def validate_completeness(state: IntakeState) -> dict[str, Any]:
    missing = missing_fields(dict(state))
    updates: dict[str, Any] = {
        "missing_fields": missing,
        "is_complete": is_intake_complete(dict(state)),
    }

    account_number = state.get("account_number")
    if is_filled(account_number) and not state.get("account_lookup"):
        updates["account_lookup"] = lookup_account(str(account_number))

    issue = state.get("issue_summary") or _latest_user_text(state)
    if is_filled(issue) and not state.get("kb_hit"):
        hit = search_kb(str(issue))
        if hit:
            updates["kb_hit"] = hit

    return updates


def route_after_validate(state: IntakeState) -> Literal["respond", "emit"]:
    if state.get("safety_crisis"):
        return "respond"
    if is_intake_complete(dict(state)):
        return "emit"
    return "respond"


def respond_messages(state: IntakeState) -> list:
    """Greeting / steer / safety keep recent chat. Intake asks are Python-owned."""
    control = respond_control_block(state)
    return [
        SystemMessage(content=f"{RESPOND_INSTRUCTIONS}\n\n---\n{control}"),
        *conversation_messages(state),
    ]


def make_respond_node(llm: BaseChatModel):
    def respond(state: IntakeState) -> dict[str, Any]:
        spoken = None
        if respond_phase(dict(state)) == "intake":
            spoken = intake_spoken_line(
                dict(state),
                user_text=_latest_user_text(state),
                last_reply=_latest_assistant_text(state),
            )
        if spoken:
            content = spoken
        else:
            content = _concat_stream_text(llm, respond_messages(state))
        return {"messages": [AIMessage(content=content)]}

    return respond


def emit_ticket(state: IntakeState) -> dict[str, Any]:
    ticket = build_ticket(dict(state))
    team = ROUTING_LABELS[ticket.tier]
    thanks = (
        f"Thanks, {ticket.customer_name} — I've got what I need. "
        if ticket.customer_name
        else ""
    )
    secret_note = (
        "Please don't share passwords, PINs, or card numbers — we never need them "
        "to open a ticket. "
        if state.get("safety_secret")
        else ""
    )
    confirmation = (
        f"{thanks}{secret_note}"
        f"I'm routing this as a Tier {ticket.tier} request to {team}."
    )
    return {
        "ticket": ticket.model_dump(),
        "is_complete": True,
        "missing_fields": [],
        "messages": [AIMessage(content=confirmation)],
    }


def build_graph(
    llm: BaseChatModel | None = None,
    classify_llm: BaseChatModel | None = None,
    agent_llm: BaseChatModel | None = None,
):
    """Compile the intake graph with an in-memory checkpointer.

    Extract and classify share the classify transport (fast structured JSON).
    Respond uses the agent transport. A single `llm=` is used for both roles.
    """
    if classify_llm is None or agent_llm is None:
        if llm is not None:
            default_classify, default_agent = llm, llm
        else:
            default_classify, default_agent = get_llms()
        classify_llm = classify_llm or default_classify
        agent_llm = agent_llm or default_agent
    graph = StateGraph(IntakeState)
    graph.add_node("extract", make_extract_node(classify_llm))
    graph.add_node("classify", make_classify_node(classify_llm))
    graph.add_node("validate", validate_completeness)
    graph.add_node("respond", make_respond_node(agent_llm))
    graph.add_node("emit", emit_ticket)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "classify")
    graph.add_edge("classify", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"respond": "respond", "emit": "emit"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("emit", END)
    return graph.compile(checkpointer=MemorySaver())
