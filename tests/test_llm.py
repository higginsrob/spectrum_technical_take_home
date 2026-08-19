from intake_agent.llm import (
    LlmConfigError,
    context_windows,
    ollama_options,
    parse_keep_alive,
    parse_num_ctx,
    parse_think,
    resolve_ollama_model,
)


def test_resolve_exact_tag():
    assert resolve_ollama_model("gemma4:26b", ["gemma4:26b", "gemma4:e4b"]) == "gemma4:26b"


def test_resolve_untagged_family_to_26b():
    available = ["gemma4:26b", "gemma4:e4b", "qwen3.5:9b"]
    assert resolve_ollama_model("gemma4", available) == "gemma4:26b"


def test_resolve_prefers_latest_when_present():
    available = ["gemma4:26b", "gemma4:latest"]
    assert resolve_ollama_model("gemma4", available) == "gemma4:latest"


def test_resolve_missing_model_lists_installed():
    try:
        resolve_ollama_model("nope", ["gemma4:e4b"], env_name="OLLAMA_CLASSIFY_MODEL")
        assert False, "expected LlmConfigError"
    except LlmConfigError as exc:
        message = str(exc)
        assert "nope" in message
        assert "gemma4:e4b" in message
        assert "OLLAMA_CLASSIFY_MODEL" in message


def test_parse_think_aliases():
    assert parse_think("on") is True
    assert parse_think("OFF") is False
    assert parse_think("low") == "low"
    assert parse_think("medium") == "medium"
    assert parse_think("high") == "high"
    try:
        parse_think("maybe", name="OLLAMA_CLASSIFY_THINK")
        assert False, "expected LlmConfigError"
    except LlmConfigError as exc:
        assert "OLLAMA_CLASSIFY_THINK" in str(exc)


def test_parse_num_ctx_accepts_commas_and_listed_sizes():
    assert parse_num_ctx("8_192") == 8192
    assert parse_num_ctx("65,536") == 65536
    try:
        parse_num_ctx("3000")
        assert False, "expected LlmConfigError"
    except LlmConfigError:
        pass


def test_parse_keep_alive_presets():
    assert parse_keep_alive("-1") == -1
    assert parse_keep_alive("forever") == -1
    assert parse_keep_alive("4 minutes") == "4m"
    assert parse_keep_alive("15m") == "15m"
    assert parse_keep_alive("1 hour") == "1h"


def test_ollama_options_reads_agent_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_AGENT_THINK", "high")
    monkeypatch.setenv("OLLAMA_AGENT_NUM_CTX", "16384")
    monkeypatch.setenv("OLLAMA_AGENT_KEEP_ALIVE", "forever")
    monkeypatch.setenv("OLLAMA_AGENT_TEMPERATURE", "0.2")
    monkeypatch.setenv("OLLAMA_AGENT_TOP_P", "0.8")
    monkeypatch.setenv("OLLAMA_AGENT_TOP_K", "20")
    monkeypatch.setenv("OLLAMA_AGENT_REPEAT_PENALTY", "1.2")
    monkeypatch.setenv("OLLAMA_AGENT_SEED", "7")
    monkeypatch.setenv("OLLAMA_AGENT_NUM_PREDICT", "256")
    options = ollama_options("agent")
    assert options["reasoning"] == "high"
    assert options["num_ctx"] == 16384
    assert options["keep_alive"] == -1
    assert options["temperature"] == 0.2
    assert options["top_p"] == 0.8
    assert options["top_k"] == 20
    assert options["repeat_penalty"] == 1.2
    assert options["seed"] == 7
    assert options["num_predict"] == 256


def test_ollama_options_reads_classify_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLASSIFY_THINK", "off")
    monkeypatch.setenv("OLLAMA_CLASSIFY_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_CLASSIFY_KEEP_ALIVE", "15m")
    monkeypatch.setenv("OLLAMA_CLASSIFY_TEMPERATURE", "0")
    monkeypatch.setenv("OLLAMA_CLASSIFY_NUM_PREDICT", "256")
    options = ollama_options("classify")
    assert options["reasoning"] is False
    assert options["num_ctx"] == 4096
    assert options["keep_alive"] == "15m"
    assert options["temperature"] == 0
    assert options["num_predict"] == 256


def test_ollama_options_roles_are_independent(monkeypatch):
    monkeypatch.setenv("OLLAMA_AGENT_THINK", "high")
    monkeypatch.setenv("OLLAMA_AGENT_NUM_CTX", "16384")
    monkeypatch.setenv("OLLAMA_CLASSIFY_THINK", "off")
    monkeypatch.setenv("OLLAMA_CLASSIFY_NUM_CTX", "2048")
    agent = ollama_options("agent")
    classify = ollama_options("classify")
    assert agent["reasoning"] == "high"
    assert agent["num_ctx"] == 16384
    assert classify["reasoning"] is False
    assert classify["num_ctx"] == 2048


def _clear_role_env(monkeypatch, role: str) -> None:
    prefix = "OLLAMA_AGENT_" if role == "agent" else "OLLAMA_CLASSIFY_"
    for suffix in (
        "THINK",
        "NUM_CTX",
        "KEEP_ALIVE",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "REPEAT_PENALTY",
        "SEED",
        "NUM_PREDICT",
    ):
        monkeypatch.delenv(f"{prefix}{suffix}", raising=False)


def test_ollama_agent_options_defaults(monkeypatch):
    _clear_role_env(monkeypatch, "agent")
    options = ollama_options("agent")
    assert options["reasoning"] is True
    assert options["temperature"] == 0
    assert "num_ctx" not in options


def test_ollama_classify_options_defaults(monkeypatch):
    _clear_role_env(monkeypatch, "classify")
    options = ollama_options("classify")
    assert options["reasoning"] is False
    assert options["temperature"] == 0
    assert "num_ctx" not in options


def test_context_windows_reads_num_ctx(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_AGENT_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_CLASSIFY_NUM_CTX", "4096")
    assert context_windows() == {"agent": 8192, "classify": 4096}


def test_context_windows_openai_has_no_num_ctx(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert context_windows() == {"agent": None, "classify": None}
