"""Slash commands: parse aliases, format /status, save JSON, CLI dispatch."""

from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from intake_agent.cli import GREETING, run_session
from intake_agent.commands import (
    Command,
    chat_log_payload,
    clear_terminal,
    format_status,
    parse_command,
    resolve_save_path,
    save_chat_log,
    snapshot_chat,
    unknown_command_message,
)


def test_parse_slash_and_single_word_aliases():
    for raw, name in (
        ("/exit", "exit"),
        ("exit", "exit"),
        ("/quit", "exit"),
        ("quit", "exit"),
        ("e", "exit"),
        ("x", "exit"),
        ("q", "exit"),
        ("/Q", "exit"),
        ("/clear", "clear"),
        ("clear", "clear"),
        ("/c", "clear"),
        ("c", "clear"),
        ("/status", "status"),
        ("status", "status"),
        ("/save", "save"),
        ("save", "save"),
    ):
        command = parse_command(raw)
        assert command is not None, raw
        assert command.name == name, raw
        assert command.args == ""


def test_parse_slash_save_with_path():
    command = parse_command("/save /tmp/chat.json")
    assert command == Command("save", "/tmp/chat.json", "/save /tmp/chat.json")


def test_parse_not_a_command():
    assert parse_command("hello") is None
    assert parse_command("please quit") is None
    assert parse_command("save out.json") is None
    assert parse_command("I forgot my password") is None
    assert parse_command("") is None
    assert parse_command("   ") is None


def test_parse_unknown_slash_stays_in_command_system():
    command = parse_command("/foo")
    assert command is not None
    assert command.name == "unknown"
    assert "Unknown command: /foo" in unknown_command_message(command.raw)
    assert "/status" in unknown_command_message(command.raw)


def test_format_status_before_any_turn():
    snapshot = snapshot_chat({}, "thread-1")
    text = format_status(snapshot, None, {"agent": 8192, "classify": 4096})
    assert "Last prompt" in text
    assert "(none yet)" in text
    assert "thread      thread-1" in text
    assert "turns       0" in text
    assert "num_ctx 8,192 (no prompt yet)" in text
    assert "num_ctx 4,096 (no prompt yet)" in text


def test_format_status_last_prompt_and_context_ratio():
    snapshot = snapshot_chat(
        {
            "messages": [
                HumanMessage(content="I forgot my password"),
                AIMessage(content="What's the name on the account?"),
            ],
            "issue_summary": "Forgot password",
            "customer_name": "Jane Doe",
            "tier": 1,
            "missing_fields": ["account_number"],
            "is_complete": False,
        },
        "abc",
    )
    last = {
        "outcome": "ok",
        "user": "I forgot my password",
        "reply": "What's the name on the account?",
        "is_complete": False,
        "tokens": {"extract": 200, "classify": 150, "respond": 1024},
        "usage": {"respond": {"input_tokens": 1200, "output_tokens": 20}},
    }
    text = format_status(snapshot, last, {"agent": 8192, "classify": 4096})
    assert "outcome     ok" in text
    assert "Forgot password" in text
    assert "Jane Doe" in text
    assert "1,200 / 8,192 tokens (15%)" in text
    assert "200 / 4,096 tokens (5%)" in text
    assert "turns       1" in text
    assert "tier        1" in text


def test_save_chat_log_round_trip(tmp_path: Path):
    snapshot = snapshot_chat(
        {
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
            "tier": 2,
            "issue_summary": "wifi is down",
        },
        "tid",
    )
    payload = chat_log_payload(snapshot, greeting="Hi there", last_prompt=None)
    path = tmp_path / "chat.json"
    saved = save_chat_log(path, payload)
    data = saved.read_text(encoding="utf-8")
    assert '"thread_id": "tid"' in data
    assert '"role": "human"' in data
    assert "wifi is down" in data
    assert "Hi there" in data


def test_resolve_save_path_default_and_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime(2026, 8, 18, 21, 30, 0)
    assert resolve_save_path("", now=now) == Path("intake_chat_20260818_213000.json")
    assert resolve_save_path("~/out.json").expanduser().name == "out.json"


def test_clear_terminal_writes_ansi_when_tty():
    stream = StringIO()
    stream.isatty = lambda: True  # type: ignore[method-assign]
    clear_terminal(stream)
    assert "\033[2J" in stream.getvalue()
    assert "\033[H" in stream.getvalue()


class _SessionGraph:
    def __init__(self):
        self.streamed: list[str] = []
        self.thread_ids: list[str] = []
        self.values: dict = {}
        self.seeded: list[str] = []

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        self.thread_ids.append(thread_id)
        return SimpleNamespace(
            config={"configurable": {"thread_id": thread_id}},
            values=self.values,
        )

    def update_state(self, config, values):
        messages = values.get("messages") or []
        for message in messages:
            self.seeded.append(getattr(message, "content", ""))
        merged = dict(self.values)
        if messages:
            merged["messages"] = [*(merged.get("messages") or []), *messages]
        for key, value in values.items():
            if key != "messages":
                merged[key] = value
        self.values = merged
        return config

    def stream(self, payload, _config, **_kwargs):
        messages = payload.get("messages") or []
        if messages:
            self.streamed.append(getattr(messages[0], "content", ""))
        if False:
            yield None


def test_run_session_quit_aliases_exit_without_graph(monkeypatch):
    graph = _SessionGraph()
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    assert run_session(iter(["quit"])) == 0
    assert run_session(iter(["/exit"])) == 0
    assert run_session(iter(["q"])) == 0
    assert graph.streamed == []


def test_run_session_unknown_slash_is_not_sent_to_model(monkeypatch, capsys):
    graph = _SessionGraph()
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    assert run_session(iter(["/nope", "quit"])) == 0
    assert graph.streamed == []
    out = capsys.readouterr().out
    assert "Unknown command: /nope" in out


def test_run_session_hello_is_a_normal_turn(monkeypatch):
    graph = _SessionGraph()
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    assert run_session(iter(["hello"])) == 1
    assert graph.streamed == ["hello"]
    assert graph.seeded == [GREETING]


def test_run_session_clear_reseeds_greeting(monkeypatch, capsys):
    graph = _SessionGraph()
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    assert run_session(iter(["/clear", "quit"])) == 0
    assert graph.seeded == [GREETING, GREETING]
    out = capsys.readouterr().out
    assert out.count("Spectrum Support Intake") == 2


def test_run_session_clear_resets_thread_and_reprint_greeting(monkeypatch, capsys):
    graph = _SessionGraph()
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    assert run_session(iter(["status", "/clear", "status", "quit"])) == 0
    assert graph.thread_ids[0] != graph.thread_ids[-1]
    out = capsys.readouterr().out
    assert out.count("Spectrum Support Intake") == 2


def test_run_session_status_and_save(monkeypatch, tmp_path, capsys):
    graph = _SessionGraph()
    graph.values = {
        "messages": [HumanMessage(content="wifi is slow")],
        "issue_summary": "wifi is slow",
        "tier": 2,
        "missing_fields": ["customer_name"],
    }
    monkeypatch.setattr("intake_agent.cli.build_graph", lambda: graph)
    dest = tmp_path / "log.json"
    assert run_session(iter(["status", f"/save {dest}", "quit"])) == 0
    out = capsys.readouterr().out
    assert "Last prompt" in out
    assert "wifi is slow" in out
    assert dest.is_file()
    assert "wifi is slow" in dest.read_text(encoding="utf-8")
    assert f"Saved chat log to {dest}" in out
