"""LLM factory. Ollama is the default; OpenAI is an env-var fallback.

Two transports: a fast classify model for extract/classify JSON, and an agent
model for the customer-facing reply.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

LlmRole = Literal["agent", "classify"]

DEFAULT_OLLAMA_AGENT_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_CLASSIFY_MODEL = "gemma4:e4b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

NUM_CTX_CHOICES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
_KEEP_ALIVE_ALIASES: dict[str, int | str] = {
    "-1": -1,
    "forever": -1,
    "infinite": -1,
    "inf": -1,
    "4m": "4m",
    "4 min": "4m",
    "4 minutes": "4m",
    "240": "4m",
    "15m": "15m",
    "15 min": "15m",
    "15 minutes": "15m",
    "900": "15m",
    "1h": "1h",
    "1 hour": "1h",
    "60m": "1h",
    "3600": "1h",
}


class LlmConfigError(RuntimeError):
    """Misconfigured or missing model — fail before the conversation starts."""


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _role_prefix(role: LlmRole) -> str:
    return "OLLAMA_AGENT_" if role == "agent" else "OLLAMA_CLASSIFY_"


def _model_env_name(role: LlmRole, provider: Literal["ollama", "openai"]) -> str:
    prefix = "OLLAMA" if provider == "ollama" else "OPENAI"
    suffix = "AGENT_MODEL" if role == "agent" else "CLASSIFY_MODEL"
    return f"{prefix}_{suffix}"


def parse_think(raw: str, name: str = "OLLAMA_THINK") -> bool | str:
    key = raw.strip().lower()
    mapping: dict[str, bool | str] = {
        "on": True,
        "true": True,
        "1": True,
        "yes": True,
        "off": False,
        "false": False,
        "0": False,
        "no": False,
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    if key not in mapping:
        raise LlmConfigError(
            f"{name} must be on, low, medium, high, or off (got {raw!r})."
        )
    return mapping[key]


def parse_num_ctx(raw: str, name: str = "OLLAMA_NUM_CTX") -> int:
    cleaned = raw.strip().replace(",", "").replace("_", "")
    try:
        value = int(cleaned)
    except ValueError as exc:
        raise LlmConfigError(f"{name} must be an integer (got {raw!r}).") from exc
    if value not in NUM_CTX_CHOICES:
        allowed = ", ".join(str(n) for n in NUM_CTX_CHOICES)
        raise LlmConfigError(f"{name} must be one of: {allowed} (got {value}).")
    return value


def parse_keep_alive(raw: str, name: str = "OLLAMA_KEEP_ALIVE") -> int | str:
    key = " ".join(raw.strip().lower().replace("_", " ").split())
    if key in _KEEP_ALIVE_ALIASES:
        return _KEEP_ALIVE_ALIASES[key]
    raise LlmConfigError(
        f"{name} must be -1/forever, 4 minutes, 15 minutes, or 1 hour "
        f"(got {raw!r})."
    )


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise LlmConfigError(f"{name} must be a number (got {raw!r}).") from exc


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw.replace(",", "").replace("_", ""))
    except ValueError as exc:
        raise LlmConfigError(f"{name} must be an integer (got {raw!r}).") from exc


def ollama_options(role: LlmRole) -> dict[str, object]:
    """Optional ChatOllama kwargs from the environment. Omit a key to use Ollama's default."""
    prefix = _role_prefix(role)
    options: dict[str, object] = {}

    think = _env(f"{prefix}THINK")
    options["reasoning"] = (
        parse_think(think, f"{prefix}THINK") if think is not None else role == "agent"
    )

    if (raw := _env(f"{prefix}NUM_CTX")) is not None:
        options["num_ctx"] = parse_num_ctx(raw, f"{prefix}NUM_CTX")
    if (raw := _env(f"{prefix}KEEP_ALIVE")) is not None:
        options["keep_alive"] = parse_keep_alive(raw, f"{prefix}KEEP_ALIVE")
    if (raw := _env(f"{prefix}TEMPERATURE")) is not None:
        options["temperature"] = _parse_float(f"{prefix}TEMPERATURE", raw)
    else:
        options["temperature"] = 0
    if (raw := _env(f"{prefix}TOP_P")) is not None:
        options["top_p"] = _parse_float(f"{prefix}TOP_P", raw)
    if (raw := _env(f"{prefix}TOP_K")) is not None:
        options["top_k"] = _parse_int(f"{prefix}TOP_K", raw)
    if (raw := _env(f"{prefix}REPEAT_PENALTY")) is not None:
        options["repeat_penalty"] = _parse_float(f"{prefix}REPEAT_PENALTY", raw)
    if (raw := _env(f"{prefix}SEED")) is not None:
        options["seed"] = _parse_int(f"{prefix}SEED", raw)
    if (raw := _env(f"{prefix}NUM_PREDICT")) is not None:
        options["num_predict"] = _parse_int(f"{prefix}NUM_PREDICT", raw)
    return options


def list_ollama_models(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise LlmConfigError(
            f"Could not reach Ollama at {base_url}. Is it running? ({exc})"
        ) from exc
    names: list[str] = []
    for model in payload.get("models") or []:
        name = model.get("name") or model.get("model")
        if name:
            names.append(name)
    return names


def resolve_ollama_model(
    requested: str,
    available: list[str],
    env_name: str = "OLLAMA_AGENT_MODEL",
) -> str:
    """Map a short tag like gemma4 onto an installed name like gemma4:26b."""
    requested = requested.strip()
    if not requested:
        raise LlmConfigError(f"{env_name} is empty.")
    if requested in available:
        return requested

    family = requested.split(":")[0]
    candidates = [
        name
        for name in available
        if name == requested or name == f"{family}:latest" or name.startswith(f"{family}:")
    ]
    if not candidates:
        installed = ", ".join(available) if available else "(none)"
        raise LlmConfigError(
            f"Ollama model '{requested}' is not installed.\n"
            f"  Installed: {installed}\n"
            f"  Fix: ollama pull {requested}\n"
            f"  Or set {env_name} to one of the installed names."
        )

    for preferred in (
        requested,
        f"{family}:latest",
        f"{family}:26b",
        f"{family}:e4b",
        f"{family}:8b",
        f"{family}:7b",
    ):
        if preferred in candidates:
            return preferred
    return sorted(candidates, key=len)[0]


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def _openai_llm(role: LlmRole) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    env_name = _model_env_name(role, "openai")
    return ChatOpenAI(
        model=os.getenv(env_name, DEFAULT_OPENAI_MODEL),
        temperature=0,
    )


def _ollama_llm(role: LlmRole, base_url: str, available: list[str]) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    env_name = _model_env_name(role, "ollama")
    default = (
        DEFAULT_OLLAMA_AGENT_MODEL if role == "agent" else DEFAULT_OLLAMA_CLASSIFY_MODEL
    )
    requested = os.getenv(env_name, default)
    model = resolve_ollama_model(requested, available, env_name)
    if model != requested:
        print(f"Ollama model '{requested}' not found; using '{model}'.")
    return ChatOllama(model=model, base_url=base_url, **ollama_options(role))


def get_llm(role: LlmRole) -> BaseChatModel:
    if _provider() == "openai":
        return _openai_llm(role)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return _ollama_llm(role, base_url, list_ollama_models(base_url))


def get_llms() -> tuple[BaseChatModel, BaseChatModel]:
    """Return (classify_llm, agent_llm). Lists Ollama models once."""
    if _provider() == "openai":
        return _openai_llm("classify"), _openai_llm("agent")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    available = list_ollama_models(base_url)
    return (
        _ollama_llm("classify", base_url, available),
        _ollama_llm("agent", base_url, available),
    )


def context_windows() -> dict[str, int | None]:
    """Configured NUM_CTX per role. None means unset (provider default)."""
    if _provider() == "openai":
        return {"agent": None, "classify": None}
    return {
        "agent": _ctx_size("agent"),
        "classify": _ctx_size("classify"),
    }


def _ctx_size(role: LlmRole) -> int | None:
    value = ollama_options(role).get("num_ctx")
    return int(value) if isinstance(value, int) else None
