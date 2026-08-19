"""No-LLM tests for wrap, markdown, think stripping, and display state."""

from types import SimpleNamespace

from intake_agent.terminal.display import DisplayState
from intake_agent.terminal.markdown import parse_inline, parse_markdown
from intake_agent.terminal.style import Style
from intake_agent.terminal.think import message_visible_and_think, strip_think_tags
from intake_agent.terminal.wrap import wrap_spans


def _texts(rows):
    return ["".join(span.text for span in row) for row in rows]


def test_wrap_breaks_on_word_boundaries():
    spans = parse_inline("hello world")
    rows = wrap_spans(spans, 8)
    assert _texts(rows) == ["hello", "world"]
    joined = "".join(_texts(rows))
    assert "hel" not in [row.strip() for row in _texts(rows) if row.strip() == "hel"]
    assert "hello" in joined
    assert all("hello world" not in row or " " in row for row in _texts(rows))


def test_wrap_does_not_split_short_words():
    rows = wrap_spans(parse_inline("hello world"), 8)
    for row in _texts(rows):
        assert "hel" != row
        assert "lo" != row


def test_wrap_hard_splits_overlong_word():
    rows = wrap_spans(parse_inline("abcdefghij"), 4)
    assert _texts(rows) == ["abcd", "efgh", "ij"]


def test_markdown_bold_italic_emphasis_code():
    spans = parse_inline("**bold** *italic* ***both*** `code`")
    by_text = {span.text: span.style for span in spans if span.text.strip()}
    assert by_text["bold"].bold is True
    assert by_text["italic"].italic is True
    assert by_text["both"].bold is True and by_text["both"].italic is True
    assert by_text["code"].code is True


def test_markdown_header_quote_fence_table():
    source = "\n".join(
        [
            "# Title",
            "> quoted",
            "```py",
            "x = 1",
            "```",
            "| A | B |",
            "| --- | --- |",
            "| 1 | 2 |",
        ]
    )
    lines = parse_markdown(source)
    styles = [[span.style for span in line] for line in lines]
    assert any(s.header == 1 for s in styles[0])
    assert any(s.quote for s in styles[1])
    assert any(s.code for s in styles[2])  # fence label
    assert any(s.code for s in styles[3])  # code body
    header_row = lines[5]
    assert any(span.style.table_header for span in header_row)
    sep_row = lines[6]
    assert any(span.style.table_sep for span in sep_row)


def test_strip_think_tags_hides_think_keeps_visible():
    visible, think = strip_think_tags("<think>abc</think>visible")
    assert visible == "visible"
    assert "abc" in think
    assert "<think>" not in visible


def test_strip_in_progress_think_tag():
    visible, think = strip_think_tags("hello <think>still going")
    assert visible == "hello "
    assert "still going" in think


def test_message_reasoning_content_is_think_not_visible():
    message = SimpleNamespace(
        content="answer",
        additional_kwargs={"reasoning_content": "secret chain"},
        response_metadata={},
        generation_info=None,
    )
    visible, think = message_visible_and_think(message)
    assert visible == "answer"
    assert "secret chain" in think


def test_display_think_updates_status_without_body():
    state = DisplayState()
    state.on_think("aaaa")
    assert state.phase == "thinking"
    assert state.source == ""
    assert state.status.startswith("thinking ")
    assert state.status.endswith("tokens")


def test_display_first_content_replaces_throbber_phase():
    state = DisplayState()
    state.on_think("aaaa")
    state.on_text("Hello **there**")
    assert state.phase == "body"
    assert state.source == "Hello **there**"
    state.on_text("Hello **there**!")
    assert state.source == "Hello **there**!"


def test_display_think_tags_in_text_do_not_enter_source():
    state = DisplayState()
    state.on_text("<think>nope</think>Ask for the account number.")
    assert "nope" not in state.source
    assert "Ask for the account number." in state.source
    assert state.phase == "body"
