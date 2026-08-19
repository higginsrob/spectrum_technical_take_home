"""CLI for the support intake agent. Interactive by default; --script for demos."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

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
from intake_agent.eval import load_script
from intake_agent.graph import build_graph, prompt_token_counts
from intake_agent.llm import LlmConfigError, context_windows
from intake_agent.terminal.abort import watch_escape_abort
from intake_agent.terminal.display import AssistantDisplay
from intake_agent.terminal.prompt import read_user_prompt
from intake_agent.terminal.style import BRIGHT_MAGENTA, RESET
from intake_agent.terminal.think import message_visible_and_think
from intake_agent.terminal.ticket import format_ticket_report

GREETING = (
    "Hi — I'm Spectrum Support Intake. I'll get you to the right team. "
    "What's going on today?"
)
PROMPT_GLYPH = "❯ "
PROMPT = f"{BRIGHT_MAGENTA}{PROMPT_GLYPH}{RESET}"
# \001/\002 mark non-printing SGR so readline counts only the glyph.
PROMPT_INPUT = f"\001{BRIGHT_MAGENTA}\002{PROMPT_GLYPH}\001{RESET}\002"


def _script_lines(path: Path) -> Iterator[str]:
    return iter(load_script(path))


def _unpack_stream_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2 and item[0] in {"messages", "updates"}:
        return str(item[0]), item[1]
    if isinstance(item, tuple) and len(item) == 3 and item[1] in {"messages", "updates"}:
        return str(item[1]), item[2]
    return "updates", item


def _message_node(metadata: Any) -> str:
    if isinstance(metadata, dict):
        return str(metadata.get("langgraph_node") or "")
    return ""


class TurnAborted(Exception):
    """User cancelled the in-flight turn (Ctrl-C or Escape)."""

    def __init__(
        self,
        prompt_tokens: dict[str, int] | None = None,
        usage: dict[str, dict[str, int]] | None = None,
    ) -> None:
        super().__init__("turn aborted")
        self.prompt_tokens = prompt_tokens or {}
        self.usage = usage or {}


def _ai_text(message: Any) -> str:
    visible, _ = message_visible_and_think(message)
    return visible


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001
        pass


def rollback_turn(graph: Any, config: dict, pre: Any) -> None:
    """Drop an aborted turn's checkpoint so the next prompt starts from pre-turn state."""
    pre_cfg = getattr(pre, "config", None) or {}
    checkpoint_id = (pre_cfg.get("configurable") or {}).get("checkpoint_id")
    if checkpoint_id:
        graph.update_state(pre_cfg, {})
        return
    config.setdefault("configurable", {})["thread_id"] = str(uuid.uuid4())


def _message_usage(message: Any) -> dict[str, int]:
    """Pull prompt/completion token counts off a LangChain chunk when present."""
    found: dict[str, int] = {}
    usage = getattr(message, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        if usage.get("input_tokens") is not None:
            found["input_tokens"] = int(usage["input_tokens"])
        if usage.get("output_tokens") is not None:
            found["output_tokens"] = int(usage["output_tokens"])
    meta = getattr(message, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return found
    if meta.get("prompt_eval_count") is not None:
        found.setdefault("input_tokens", int(meta["prompt_eval_count"]))
    if meta.get("eval_count") is not None:
        found.setdefault("output_tokens", int(meta["eval_count"]))
    nested = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(nested, dict):
        if nested.get("prompt_tokens") is not None:
            found.setdefault("input_tokens", int(nested["prompt_tokens"]))
        if nested.get("completion_tokens") is not None:
            found.setdefault("output_tokens", int(nested["completion_tokens"]))
        if nested.get("input_tokens") is not None:
            found.setdefault("input_tokens", int(nested["input_tokens"]))
        if nested.get("output_tokens") is not None:
            found.setdefault("output_tokens", int(nested["output_tokens"]))
    return found


def _estimate_prompt_tokens(graph: Any, config: dict, user: str) -> dict[str, int]:
    try:
        pre = graph.get_state(config)
    except Exception:  # noqa: BLE001
        pre = None
    values = dict(getattr(pre, "values", None) or {})
    messages = list(values.get("messages") or [])
    values["messages"] = [*messages, HumanMessage(content=user)]
    try:
        return prompt_token_counts(values)
    except Exception:  # noqa: BLE001
        return {}


def run_turn(
    graph: Any,
    user: str,
    config: dict,
    display: AssistantDisplay,
    *,
    watch_abort: bool = False,
) -> dict[str, Any]:
    ticket: dict[str, Any] | None = None
    complete = False
    stream_iter: Any = None
    aborted = False
    started = False
    pre: Any = None
    usage_by_node: dict[str, dict[str, int]] = {}
    prompt_tokens = _estimate_prompt_tokens(graph, config, user)
    try:
        pre = graph.get_state(config)
        display.begin()
        started = True
        stream_iter = graph.stream(
            {"messages": [HumanMessage(content=user)]},
            config,
            stream_mode=["messages", "updates"],
        )
        with watch_escape_abort(watch_abort):
            for item in stream_iter:
                mode, data = _unpack_stream_item(item)
                if mode == "messages":
                    chunk, metadata = data
                    node = _message_node(metadata)
                    display.on_message(chunk, visible=node == "respond")
                    usage = _message_usage(chunk)
                    if node and usage:
                        usage_by_node[node] = usage
                elif mode == "updates" and isinstance(data, dict) and "emit" in data:
                    update = data["emit"] or {}
                    ticket = update.get("ticket")
                    complete = bool(update.get("is_complete"))
                    for message in update.get("messages") or []:
                        text = _ai_text(message)
                        if text:
                            display.on_text(text)
        return {
            "ticket": ticket,
            "is_complete": complete,
            "prompt_tokens": prompt_tokens,
            "usage": usage_by_node,
        }
    except KeyboardInterrupt:
        aborted = True
    finally:
        if stream_iter is not None:
            _close_stream(stream_iter)
        if started:
            display.finish()
    if aborted:
        if pre is not None:
            try:
                rollback_turn(graph, config, pre)
            except Exception:  # noqa: BLE001
                pass
        raise TurnAborted(prompt_tokens=prompt_tokens, usage=usage_by_node) from None
    return {
        "ticket": ticket,
        "is_complete": complete,
        "prompt_tokens": prompt_tokens,
        "usage": usage_by_node,
    }


def _graph_snapshot(graph: Any, config: dict) -> dict[str, Any]:
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    try:
        state = graph.get_state(config)
        values = getattr(state, "values", None) or {}
    except Exception:  # noqa: BLE001
        values = {}
    return snapshot_chat(dict(values), thread_id)


def _print_greeting() -> None:
    print()
    print(GREETING)
    print()


def seed_printed_greeting(graph: Any, config: dict) -> None:
    """Store the CLI hello so the first user turn is steer, not a second intro."""
    graph.update_state(config, {"messages": [AIMessage(content=GREETING)]})


def _dispatch_command(
    command: Command,
    *,
    graph: Any,
    config: dict,
    last_prompt: dict[str, Any] | None,
) -> str:
    """Run a slash command. Returns 'exit', 'clear', or 'continue'."""
    if command.name == "exit":
        return "exit"
    if command.name == "unknown":
        print()
        print(unknown_command_message(command.raw))
        print()
        return "continue"
    if command.name == "clear":
        clear_terminal()
        config.setdefault("configurable", {})["thread_id"] = str(uuid.uuid4())
        _print_greeting()
        seed_printed_greeting(graph, config)
        return "clear"
    if command.name == "status":
        print()
        print(format_status(_graph_snapshot(graph, config), last_prompt, context_windows()))
        print()
        return "continue"
    if command.name == "save":
        print()
        path = resolve_save_path(command.args)
        payload = chat_log_payload(
            _graph_snapshot(graph, config),
            greeting=GREETING,
            last_prompt=last_prompt,
        )
        try:
            saved = save_chat_log(path, payload)
        except OSError as exc:
            print(f"Could not save chat log: {exc}", file=sys.stderr)
        else:
            print(f"Saved chat log to {saved}")
        print()
        return "continue"
    print()
    print(unknown_command_message(command.raw))
    print()
    return "continue"


def run_session(lines: Iterator[str] | None = None) -> int:
    try:
        graph = build_graph()
    except LlmConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
    scripted = lines is not None
    tty = bool(sys.stdout.isatty())
    display = AssistantDisplay(
        use_throbber=tty and not scripted,
        use_live=tty,
    )
    last_prompt: dict[str, Any] | None = None

    _print_greeting()
    seed_printed_greeting(graph, config)

    while True:
        if scripted:
            try:
                user = next(lines)  # type: ignore[arg-type]
            except StopIteration:
                print("Script ended before the ticket was complete.", file=sys.stderr)
                return 1
            print(f"{PROMPT}{user}")
        else:
            try:
                user = read_user_prompt(PROMPT_GLYPH, fallback_prompt=PROMPT_INPUT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not user:
                continue

        command = parse_command(user)
        if command is not None:
            action = _dispatch_command(
                command, graph=graph, config=config, last_prompt=last_prompt
            )
            if action == "exit":
                if scripted:
                    print()
                return 0
            if action == "clear":
                last_prompt = None
            continue

        print()
        try:
            result = run_turn(
                graph,
                user,
                config,
                display,
                watch_abort=not scripted,
            )
        except TurnAborted as aborted:
            last_prompt = {
                "user": user,
                "outcome": "aborted",
                "reply": display.state.source,
                "is_complete": False,
                "tokens": aborted.prompt_tokens,
                "usage": aborted.usage,
            }
            print()
            if scripted:
                return 0
            continue
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as exc:
            print(f"Intake failed: {exc}", file=sys.stderr)
            return 1

        last_prompt = {
            "user": user,
            "outcome": "ok",
            "reply": display.state.source,
            "is_complete": bool(result.get("is_complete")),
            "tokens": result.get("prompt_tokens") or {},
            "usage": result.get("usage") or {},
        }
        print()
        if result.get("is_complete") and result.get("ticket"):
            print(
                format_ticket_report(
                    result["ticket"],
                    color=bool(sys.stdout.isatty()),
                )
            )
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualify a support request and emit a structured ticket."
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="Replay user turns from a text file (one utterance per line).",
    )
    args = parser.parse_args(argv)
    lines = _script_lines(args.script) if args.script else None
    return run_session(lines)


if __name__ == "__main__":
    raise SystemExit(main())
