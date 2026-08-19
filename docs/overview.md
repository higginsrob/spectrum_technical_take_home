# Overview

This repo is a working prototype of a **support intake agent** for a SpectrumGPT-style take-home: greet a customer, classify the request into Tier 1 / 2 / 3, collect the fields that tier requires, and emit a structured JSON ticket.

Python owns control flow (required fields, completeness, routing). The LLM owns language (extract, classify, reply). Reviewers run it with `pip install` and a conversation in the terminal.

## Run

Python 3.11+ and either [Ollama](https://ollama.com) or an OpenAI key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
ollama pull gemma4:26b
ollama pull gemma4:e4b
python -m intake_agent --script examples/greeting_then_wifi.txt
python -m intake_agent --script examples/tier1_password_reset.txt
intake-agent
```

Set `LLM_PROVIDER=openai` in `.env` to use `ChatOpenAI` instead of local Ollama (`OPENAI_AGENT_MODEL` / `OPENAI_CLASSIFY_MODEL`). Use tagged Ollama names (`gemma4:26b`, `gemma4:e4b`); an untagged `gemma4` is mapped onto an installed `gemma4:*` model.

Extract and classify use the **classify** transport (`OLLAMA_CLASSIFY_*`); the customer reply uses the **agent** transport (`OLLAMA_AGENT_*`). Classify defaults to `think=off` so a large agent model is not loaded for hidden JSON. See `.env.example` for per-role knobs (`THINK`, `NUM_CTX`, `KEEP_ALIVE`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `REPEAT_PENALTY`, `SEED`, `NUM_PREDICT`). Unset keys keep factory defaults (agent `think=on`, classify `think=off`, both `temperature=0`).

## Layout

| Path | Role |
| --- | --- |
| [src/intake_agent/graph.py](../src/intake_agent/graph.py) | LangGraph turn: extract → classify → validate → respond or emit |
| [src/intake_agent/rules.py](../src/intake_agent/rules.py) | Required-field matrix, merge/corrections, ticket build |
| [src/intake_agent/safety.py](../src/intake_agent/safety.py) | Jailbreak, secret, and crisis scanners |
| [src/intake_agent/eval.py](../src/intake_agent/eval.py) | Replay `examples/` and score tickets |
| [src/intake_agent/cli.py](../src/intake_agent/cli.py) | `❯` prompt, slash commands, streaming, `--script` |
| [src/intake_agent/commands.py](../src/intake_agent/commands.py) | `/exit` `/clear` `/status` `/save` |
| [src/intake_agent/terminal/](../src/intake_agent/terminal/) | Throbber, markdown, word wrap, SIGWINCH reflow |
| [src/intake_agent/prompts.py](../src/intake_agent/prompts.py) | System and per-node instructions |
| [src/intake_agent/tools.py](../src/intake_agent/tools.py) | Stub `lookup_account` / `search_kb` |
| [examples/](../examples/) | Multi-turn transcripts (one user answer per ask) |
| [tests/](../tests/) | No-LLM unit tests |

## Docs

- [Architecture](architecture.md) — graph, state, why LangGraph
- [Interactive CLI](cli.md) — streaming, think tokens, markdown, resize

The root [README](../README.md) is the reviewer-facing quick start and design-decision writeup.
