# Interactive CLI

The intake agent talks in the terminal. Orchestration is unchanged; this document is the **display** contract.

## Layout

```

Hi — I'm Spectrum Support Intake. I'll get you to the right team. What's going on today?

❯ I forgot my password

Can you confirm your **account number**?

❯
```

- Prompt prefix is `❯ ` (not `You:` / `Agent:`).
- Return submits the turn. **Shift+Return** inserts a newline in the prompt so a turn can span multiple lines.
- After Return, the CLI prints one extra newline: a blank line between the user line and the assistant block.
- After the assistant block, one blank line before the next `❯ `.
- `--script` prints `❯ {line}`, then the same blank line, then the reply. Demo transcripts stay deterministic.

Quit with `/exit`, `quit`, `q`, Ctrl-C at the `❯ ` prompt, or Ctrl-D.

A line that starts with `/`, or a single word that matches a known alias, is a **slash command** and is not sent to the model:

| Command | Aliases | Effect |
| --- | --- | --- |
| `/exit` | `exit`, `/quit`, `quit`, `e`, `x`, `q` | Close the program |
| `/clear` | `clear`, `/c`, `c` | Clear the terminal, drop chat history, start a new session |
| `/status` | `status` | Last prompt outcome, session metadata, context window (`num_ctx` vs prompt size) |
| `/save` | `save`, `/save path.json` | Export the chat log as JSON |

Unknown `/foo` prints the command list. Ordinary single words (`hello`, `wifi`) still go to intake.

During a turn (throbber or streaming reply), **Ctrl-C or Escape** cancels that generation, drops the aborted turn's graph checkpoint, and returns to `❯ `. Partial visible text already painted is left on screen. A second Ctrl-C at the empty prompt still quits.

## Turn timeline

1. User presses Return (Shift+Return only adds a newline).
2. Blank line.
3. **Throbber** (interactive TTY only): braille spinner `⠋ → ⠙ → ⠹ → ⠸ → ⠼ → ⠴ → ⠦ → ⠧ → ⠇ → ⠏` plus `loading...`.
4. If think/reasoning tokens arrive, the label becomes `thinking N tokens`. Think **text is never printed**.
5. First visible content token: throbber is cleared and replaced by streaming markdown.
6. Further tokens (and terminal resize) reflow the current assistant block on word boundaries.
7. Block finishes; blank line; next prompt or ticket JSON.

Visible text comes only from the `respond` node. Extract/classify JSON is hidden. The `emit` confirmation is not an LLM stream; the CLI treats the emit update as the visible reply, then prints the ticket JSON after a blank line.

## Think tokens

Ollama thinking models (`qwen3`, `gpt-oss`, …) default to `OLLAMA_AGENT_THINK=on` (`reasoning=True`) on the customer-facing reply, so reasoning lands in `additional_kwargs["reasoning_content"]` instead of the reply. Set `OLLAMA_AGENT_THINK` to `low` / `medium` / `high` (gpt-oss intensity) or `off` to disable. Extract/classify use `OLLAMA_CLASSIFY_THINK` (default `off`) so hidden JSON does not wait on a reasoning model.

The CLI also:

1. Reads `reasoning_content` / `thinking` metadata.
2. Treats content blocks typed as thinking/reasoning as think.
3. Strips `<think>...</think>` (and an in-progress `<think>` with no close) from visible text.

`N` is `tiktoken` `cl100k_base` token count of the accumulated think text (character/4 fallback if tiktoken is missing).

Throbber runs only when stdout is a TTY **and** the session is interactive. `--script` still streams markdown on a TTY but skips the spinner so logs stay readable. Pipes/CI: no throbber, no SIGWINCH; the assistant block prints once when the turn ends.

## Markdown subset

The renderer re-parses the **full accumulated source** on every chunk (not incremental DOM). Supported:

| Construct | Rendering |
| --- | --- |
| `#`–`######` headers | Bold; H1 bright cyan, H2+ cyan |
| `**bold**`, `*italic*` / `_italic_` | Bold / italic SGR |
| `***emphasis***` | Bold + italic |
| `> quote` | Dim, leading `│` |
| `` `code` `` and fenced ` ``` ` | Cyan + reverse |
| Pipe tables | Bold header, dim `│` / `─` grid |

Unclosed markers apply through end-of-buffer so partial `**bol` still looks bold while streaming. Mid-word `_` in identifiers is ignored.

## Wrap and resize

[wrap.py](../src/intake_agent/terminal/wrap.py) wraps styled spans at `shutil.get_terminal_size().columns`. Breaks are at whitespace. A word longer than the width is hard-split; nothing else is.

`SIGWINCH` (Unix) moves to the start of the current assistant block, clears to end of screen, and repaints. The handler is removed when the block finishes so the next `❯ ` line is not rewritten.

## Code map

| Module | Responsibility |
| --- | --- |
| `cli.py` | Prompt loop, slash commands, `graph.stream(stream_mode=["messages", "updates"])`, turn abort |
| `commands.py` | Parse `/exit` `/clear` `/status` `/save` and format status / chat JSON |
| `terminal/prompt.py` | Interactive `❯` editor; Shift+Return newline, Return submits |
| `terminal/abort.py` | Escape → SIGINT while a turn is running (cbreak stdin) |
| `terminal/throbber.py` | Background braille spinner |
| `terminal/think.py` | Split visible vs think, token count |
| `terminal/markdown.py` | Parse + ANSI paint |
| `terminal/wrap.py` | Word-boundary wrap |
| `terminal/display.py` | Phase machine + repaint |
