"""CLI turn abort: Ctrl-C / Escape cancel the in-flight generation, not the session."""

from io import StringIO
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from intake_agent.cli import GREETING, TurnAborted, rollback_turn, run_turn, seed_printed_greeting
from intake_agent.graph import build_graph
from intake_agent.safety import respond_phase
from intake_agent.terminal.abort import is_abort_key
from intake_agent.terminal.display import AssistantDisplay


def _display() -> AssistantDisplay:
    return AssistantDisplay(stream=StringIO(), use_throbber=False, use_live=False)


def test_is_abort_key_escape_and_ctrl_c():
    assert is_abort_key(b"\x1b") is True
    assert is_abort_key(b"\x03") is True
    assert is_abort_key(b"\x1b[A") is False
    assert is_abort_key(b"q") is False
    assert is_abort_key(b"") is False


class _FakeGraph:
    def __init__(self, *, items=None, interrupt=False, checkpoint_id="ckpt-1"):
        self.items = list(items or [])
        self.interrupt = interrupt
        self.checkpoint_id = checkpoint_id
        self.updated: list[tuple] = []
        self.closed = False

    def get_state(self, config):
        configurable = {"thread_id": config["configurable"]["thread_id"]}
        if self.checkpoint_id is not None:
            configurable["checkpoint_id"] = self.checkpoint_id
        return SimpleNamespace(
            config={"configurable": configurable},
            values={},
        )

    def update_state(self, config, values):
        self.updated.append((config, values))
        return config

    def stream(self, *_args, **_kwargs):
        def gen():
            try:
                if self.interrupt:
                    raise KeyboardInterrupt()
                yield from self.items
            finally:
                self.closed = True

        return gen()


def test_run_turn_keyboard_interrupt_rolls_back_and_closes_stream():
    graph = _FakeGraph(interrupt=True)
    config = {"configurable": {"thread_id": "t1"}}
    try:
        run_turn(graph, "hello", config, _display())
        assert False, "expected TurnAborted"
    except TurnAborted:
        pass
    assert graph.updated == [
        ({"configurable": {"thread_id": "t1", "checkpoint_id": "ckpt-1"}}, {})
    ]
    assert graph.closed is True


def test_run_turn_success_does_not_rollback():
    graph = _FakeGraph(items=[("updates", {"extract": {}})])
    config = {"configurable": {"thread_id": "t1"}}
    result = run_turn(graph, "hello", config, _display())
    assert result["ticket"] is None
    assert result["is_complete"] is False
    assert result["prompt_tokens"]["respond"] > 0
    assert graph.updated == []


def test_run_turn_records_usage_metadata():
    chunk = AIMessage(
        content="What's going on today?",
        usage_metadata={"input_tokens": 42, "output_tokens": 7, "total_tokens": 49},
    )
    graph = _FakeGraph(items=[("messages", (chunk, {"langgraph_node": "respond"}))])
    config = {"configurable": {"thread_id": "t1"}}
    result = run_turn(graph, "hello", config, _display())
    assert result["usage"]["respond"]["input_tokens"] == 42
    assert result["usage"]["respond"]["output_tokens"] == 7


def test_rollback_first_turn_starts_new_thread():
    graph = _FakeGraph(checkpoint_id=None)
    config = {"configurable": {"thread_id": "old"}}
    pre = graph.get_state(config)
    rollback_turn(graph, config, pre)
    assert config["configurable"]["thread_id"] != "old"
    assert graph.updated == []


class _JsonLlm:
    def __init__(self, payload: str):
        self.payload = payload

    def bind(self, **_kwargs):
        return self

    def invoke(self, _messages):
        return AIMessage(content=self.payload)

    def stream(self, _messages):
        yield AIMessage(content=self.payload)


class _AbortAfter:
    def __init__(self, graph, after: int):
        self._graph = graph
        self.after = after

    def get_state(self, config):
        return self._graph.get_state(config)

    def update_state(self, config, values):
        return self._graph.update_state(config, values)

    def stream(self, *args, **kwargs):
        n = 0
        for item in self._graph.stream(*args, **kwargs):
            n += 1
            yield item
            if n >= self.after:
                raise KeyboardInterrupt()


def test_aborted_turn_does_not_keep_partial_checkpoint():
    classify = _JsonLlm(
        '{"issue_summary": "password", "off_topic": false, "customer_name": "Jane"}'
    )
    agent = _JsonLlm("What is your account number?")
    graph = build_graph(classify_llm=classify, agent_llm=agent)
    config = {"configurable": {"thread_id": "abort-session"}}

    first = run_turn(graph, "I forgot my password", config, _display())
    assert first["is_complete"] is False
    after_first = graph.get_state(config)
    first_messages = list(after_first.values.get("messages") or [])
    assert after_first.values.get("customer_name") == "Jane"

    wrapper = _AbortAfter(graph, after=1)
    try:
        run_turn(wrapper, "account is 44556677", config, _display())
        assert False, "expected TurnAborted"
    except TurnAborted:
        pass

    rolled = graph.get_state(config)
    assert rolled.next == ()
    assert [getattr(m, "content", None) for m in rolled.values.get("messages") or []] == [
        getattr(m, "content", None) for m in first_messages
    ]
    assert rolled.values.get("account_number") in {None, ""}
    assert "44556677" not in [
        getattr(m, "content", "") for m in rolled.values.get("messages") or []
    ]

    second = run_turn(graph, "account is 44556677", config, _display())
    assert second["is_complete"] is False
    after_second = graph.get_state(config)
    texts = [getattr(m, "content", "") for m in after_second.values.get("messages") or []]
    assert any("44556677" in text for text in texts)
    assert any("forgot" in text.lower() for text in texts)


def test_seed_printed_greeting_makes_thanks_steer_not_reintro():
    classify = _JsonLlm('{"issue_summary": null, "off_topic": true}')
    agent = _JsonLlm("Sure — internet, TV, phone, or billing, what's going on?")
    graph = build_graph(classify_llm=classify, agent_llm=agent)
    config = {"configurable": {"thread_id": "seed-hello"}}
    seed_printed_greeting(graph, config)
    seeded = graph.get_state(config)
    texts = [getattr(m, "content", "") for m in seeded.values.get("messages") or []]
    assert GREETING in texts

    result = run_turn(graph, "thanks", config, _display())
    assert result["is_complete"] is False
    after = graph.get_state(config)
    values = dict(after.values)
    assert respond_phase(values) == "steer"
    replies = [
        getattr(m, "content", "")
        for m in values.get("messages") or []
        if getattr(m, "type", None) == "ai" or isinstance(m, AIMessage)
    ]
    assert replies[-1] != GREETING
    assert "Spectrum Support Intake" not in replies[-1]
