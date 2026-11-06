"""The one place that knows which LLM is in use. Switch providers with LLM_MODEL in .env."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Claude-only settings.
# max_tokens: langchain-anthropic otherwise defaults to the model's 128k ceiling, which is
# too large for the non-streaming calls this repo makes.
# fallbacks: a safety-classifier refusal is retried server-side on a suitable model
# instead of surfacing as an empty reply. Remove these two keys to opt out.
_ANTHROPIC_DEFAULTS: dict[str, Any] = {
    "max_tokens": 16000,
    "betas": ["server-side-fallback-2026-07-01"],
    "model_kwargs": {"fallbacks": "default"},
}


def get_llm(model: str | None = None, **kwargs: Any) -> Any:
    """Build the chat model named by ``model`` or LLM_MODEL, e.g. 'anthropic:claude-opus-5-5' or 'openai:gpt-4o'.

    Supports any OpenAI-compatible inference provider (vLLM, LM Studio, Ollama, LocalAI,
    OpenRouter, Groq, Together, etc.) by configuring OPENAI_BASE_URL (or BASE_URL) and
    OPENAI_API_KEY (or API_KEY) in .env.
    """
    load_dotenv()
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example) to use a real LLM.")

    base_url = (
        kwargs.pop("base_url", None)
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("LLM_BASE_URL")
    )
    api_key = (
        kwargs.pop("api_key", None)
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("LLM_API_KEY")
    )

    model_provider: str | None = None
    if ":" in model:
        prefix, model_name = model.split(":", 1)
        if prefix in ("openai-compatible", "openai_compatible", "openai"):
            model_provider = "openai"
        else:
            model_provider = prefix
    else:
        model_name = model
        env_provider = os.environ.get("LLM_PROVIDER")
        if env_provider in ("openai-compatible", "openai_compatible", "openai"):
            model_provider = "openai"
        elif env_provider:
            model_provider = env_provider
        elif base_url:
            model_provider = "openai"

    if model_provider == "anthropic" or (not model_provider and model_name.startswith("claude")):
        kwargs = {**_ANTHROPIC_DEFAULTS, **kwargs}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
    elif model_provider == "openai" or (not model_provider and any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-"))):
        model_provider = "openai"
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif base_url:
            # OpenAI-compatible local services (e.g. vLLM, LM Studio, Ollama) often don't
            # require auth, but the openai client errors if no api_key is provided.
            kwargs["api_key"] = "dummy"

        if "organization" not in kwargs and os.environ.get("OPENAI_ORGANIZATION"):
            kwargs["organization"] = os.environ["OPENAI_ORGANIZATION"]
    else:
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

    if model_provider:
        return init_chat_model(model_name, model_provider=model_provider, **kwargs)
    return init_chat_model(model, **kwargs)


def reply_text(reply: Any) -> str:
    """The text of a chat reply, as a plain string.

    Don't use reply.content for this. Claude replies that include a thinking block come back as a
    list of content blocks, and thinking can't be turned off on current Opus models.
    """
    if hasattr(reply, "text"):
        return str(reply.text)
    if hasattr(reply, "content"):
        return str(reply.content)
    return str(reply)


def structured(llm: Any, schema: type, *, method: str | None = None) -> Any:
    """Bind a Pydantic schema to an LLM.

    The explicit method matters: the default on ChatAnthropic is forced tool calling, which
    current Claude models reject. json_schema is also accepted by OpenAI and Ollama.
    Can be configured via LLM_STRUCTURED_METHOD in .env (defaults to 'json_schema').
    """
    method = method or os.environ.get("LLM_STRUCTURED_METHOD", "json_schema")
    return llm.with_structured_output(schema, method=method)
