"""Deterministic graph nodes: validate, route, emit — no LLM."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from intake_agent.graph import (
    _coerce_json_text,
    _latest_assistant_text,
    collapse_leading_system_messages,
    conversation_messages,
    emit_ticket,
    invoke_structured,
    make_respond_node,
    prompt_token_counts,
    respond_control_block,
    route_after_validate,
    validate_completeness,
)
from intake_agent.schemas import Extraction
from intake_agent.schemas import Classification
from intake_agent.tools import lookup_account


class _RecordingLlm:
    def __init__(self, payload: str):
        self.payload = payload
        self.seen: list[list] = []

    def bind(self, **_kwargs):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=self.payload)

    def stream(self, messages):
        self.seen.append(messages)
        yield AIMessage(content=self.payload)


def test_latest_assistant_text_skips_user_turns():
    state = {
        "messages": [
            HumanMessage(content="I forgot my password"),
            AIMessage(content="May I have your full name?"),
            HumanMessage(content="Jane Doe"),
        ]
    }
    assert _latest_assistant_text(state) == "May I have your full name?"


def test_conversation_messages_keeps_recent_chat_turns():
    system = SystemMessage(content="ignore")
    turns = [
        HumanMessage(content=f"u{i}") if i % 2 == 0 else AIMessage(content=f"a{i}")
        for i in range(16)
    ]
    selected = conversation_messages({"messages": [system, *turns]}, limit=4)
    assert system not in selected
    assert [m.content for m in selected] == ["u12", "a13", "u14", "a15"]


def test_respond_sends_history_not_a_status_human_message():
    llm = _RecordingLlm("What's going on with the service?")
    node = make_respond_node(llm)
    human1 = HumanMessage(content="hi")
    ai1 = AIMessage(content="Hi there! How can I help you today?")
    human2 = HumanMessage(content="what can you help me with?")
    node({"messages": [human1, ai1, human2]})
    payload = llm.seen[0]
    assert getattr(payload[0], "type", None) == "system"
    assert "Phase: steer" in payload[0].content
    assert payload[1:] == [human1, ai1, human2]


def test_intake_respond_does_not_replay_transcript():
    llm = _RecordingLlm("SHOULD NOT BE USED")
    node = make_respond_node(llm)
    updates = node(
        {
            "messages": [HumanMessage(content="my internet went out")],
            "issue_summary": "my internet went out",
            "tier": 2,
        }
    )
    text = updates["messages"][0].content
    assert "name" in text.lower()
    assert "Hi" not in text
    assert "Spectrum Support Intake" not in text
    assert llm.seen == []


def test_restated_issue_varies_name_ask_without_hello():
    llm = _RecordingLlm("SHOULD NOT BE USED")
    node = make_respond_node(llm)
    first = node(
        {
            "messages": [HumanMessage(content="my internet went out")],
            "issue_summary": "my internet went out",
            "tier": 2,
        }
    )["messages"][0].content
    second = node(
        {
            "messages": [
                HumanMessage(content="my internet went out"),
                AIMessage(content=first),
                HumanMessage(content="my internet went out"),
            ],
            "issue_summary": "my internet went out",
            "tier": 2,
        }
    )["messages"][0].content
    assert "name" in second.lower()
    assert "Hi" not in second
    assert second != first


def test_respond_control_block_first_turn_is_greeting():
    block = respond_control_block({"messages": [HumanMessage(content="hi")]})
    assert "Phase: greeting" in block
    assert "I understand" not in block
    assert "Never say you have routed" in block


def test_respond_control_block_after_cli_hello_is_steer():
    block = respond_control_block(
        {
            "messages": [
                AIMessage(
                    content=(
                        "Hi — I'm Spectrum Support Intake. I'll get you to the right team. "
                        "What's going on today?"
                    )
                ),
                HumanMessage(content="thanks"),
            ]
        }
    )
    assert "Phase: steer" in block
    assert "Phase: greeting" not in block


def test_classification_allows_unclassified_greeting():
    parsed = Classification.model_validate({"tier": None, "reasoning": ""})
    assert parsed.tier is None


def test_extraction_null_booleans_default_false():
    parsed = Extraction.model_validate(
        {
            "steps_already_tried": "restarted modem",
            "is_correction": None,
            "off_topic": None,
        }
    )
    assert parsed.is_correction is False
    assert parsed.off_topic is False
    assert parsed.steps_already_tried == "restarted modem"


def test_invoke_structured_accepts_null_booleans():
    llm = _RecordingLlm(
        '{"steps_already_tried": "restarted modem", "is_correction": null, "off_topic": null}'
    )
    parsed = invoke_structured(
        llm,
        Extraction,
        [SystemMessage(content="Extract slots."), HumanMessage(content="yes I did")],
        retries=0,
    )
    assert parsed.is_correction is False
    assert parsed.off_topic is False
    assert parsed.steps_already_tried == "restarted modem"


def test_coerce_json_text_strips_fences_and_prose():
    assert _coerce_json_text('```json\n{"tier": 1}\n```') == '{"tier": 1}'
    assert _coerce_json_text('Here you go: {"a": 1} thanks') == '{"a": 1}'


def test_collapse_leading_system_messages_merges_prefix():
    human = HumanMessage(content="hi")
    merged = collapse_leading_system_messages(
        [
            SystemMessage(content="JSON only."),
            SystemMessage(content="Extract slots."),
            human,
        ]
    )
    assert len(merged) == 2
    assert isinstance(merged[0], SystemMessage)
    assert merged[0].content == "JSON only.\n\nExtract slots."
    assert merged[1] is human


def test_collapse_leading_system_messages_leaves_single_system():
    messages = [SystemMessage(content="Ask one question."), HumanMessage(content="hi")]
    assert collapse_leading_system_messages(messages) == messages


def test_invoke_structured_sends_one_system_message():
    llm = _RecordingLlm('{"issue_summary": null, "off_topic": false}')
    invoke_structured(
        llm,
        Extraction,
        [
            SystemMessage(content="Extract slots."),
            HumanMessage(content="Latest user message:\nhi"),
        ],
        retries=0,
    )
    assert llm.seen
    roles = [getattr(m, "type", None) for m in llm.seen[0]]
    assert roles.count("system") == 1
    assert roles[0] == "system"


def test_invoke_structured_openai_uses_function_calling():
    class ChatOpenAI:
        def __init__(self):
            self.kwargs = None

        def with_structured_output(self, schema, **kwargs):
            self.kwargs = kwargs

            class _Runner:
                def invoke(self, _messages):
                    return schema(off_topic=False)

            return _Runner()

    llm = ChatOpenAI()
    result = invoke_structured(
        llm,  # type: ignore[arg-type]
        Extraction,
        [HumanMessage(content="hi")],
        retries=0,
    )
    assert llm.kwargs == {"method": "function_calling"}
    assert result.off_topic is False


def test_validate_incomplete_tier1():
    state = {
        "messages": [HumanMessage(content="reset my password")],
        "tier": 1,
        "customer_name": "Jane Doe",
        "account_number": None,
        "issue_summary": "Password reset",
    }
    updates = validate_completeness(state)
    assert updates["is_complete"] is False
    assert "account_number" in updates["missing_fields"]


def test_validate_complete_looks_up_account_and_kb():
    state = {
        "messages": [HumanMessage(content="I forgot my password")],
        "tier": 1,
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "I forgot my password and can't log in",
    }
    updates = validate_completeness(state)
    assert updates["is_complete"] is True
    assert updates["missing_fields"] == []
    assert updates["account_lookup"] == lookup_account("44556677")
    assert updates["kb_hit"]["id"] == "password-reset"


def test_route_after_validate():
    assert route_after_validate({"tier": 1, "customer_name": "A"}) == "respond"
    complete = {
        "tier": 1,
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "Password reset",
    }
    assert route_after_validate(complete) == "emit"


def test_route_after_validate_crisis_never_emits():
    complete = {
        "tier": 1,
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "Password reset",
        "safety_crisis": True,
    }
    assert route_after_validate(complete) == "respond"


def test_emit_ticket_payload():
    state = {
        "tier": 1,
        "classification_reasoning": "Password reset is self-service.",
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "Password reset",
        "messages": [HumanMessage(content="hi")],
    }
    updates = emit_ticket(state)
    assert updates["is_complete"] is True
    assert updates["ticket"]["routing_team"] == "self_service"
    assert updates["ticket"]["status"] == "ready_for_routing"
    assert isinstance(updates["messages"][0], AIMessage)
    assert "Thanks, Jane Doe" in updates["messages"][0].content
    assert "Tier 1" in updates["messages"][0].content


def test_emit_ticket_mentions_secret_when_flagged():
    state = {
        "tier": 1,
        "classification_reasoning": "Password reset is self-service.",
        "customer_name": "Jane Doe",
        "account_number": "44556677",
        "issue_summary": "Password reset",
        "safety_secret": True,
        "messages": [HumanMessage(content="hi")],
    }
    text = emit_ticket(state)["messages"][0].content
    assert "passwords" in text.lower()
    assert "Tier 1" in text


def test_prompt_token_counts_classify_after_issue():
    greeting = prompt_token_counts({"messages": [HumanMessage(content="hi")]})
    assert greeting["extract"] > 0
    assert greeting["classify"] == 0
    assert greeting["respond"] > greeting["extract"]

    intake = prompt_token_counts(
        {
            "messages": [
                HumanMessage(content="I forgot my password"),
                AIMessage(content="What's the name on the account?"),
            ],
            "issue_summary": "Forgot password",
            "tier": 1,
        }
    )
    assert intake["classify"] > 0
    # Intake asks are a compact next-field prompt, not a growing transcript.
    assert intake["respond"] > 0
    assert intake["extract"] >= greeting["extract"]
