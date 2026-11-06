# Agent Guidelines & System Architecture

## Architecture Overview
The codebase (`plan1`) defines the `planagents` library for plan-based LLM agents:
- `planagents.llm`: LLM model factory (`get_llm`), reply text extractor (`reply_text`), and structured output binder (`structured`).
- `planagents.dag`: Graph execution engine, topological sorting, dependency tracking, task dispatch.
- `planagents.schemas`: Pydantic models for plans, tasks, validation, and execution results.

## Model Provider Configurations
Providers are configured via environment variables in `.env` (see `.env.example`):
- `LLM_MODEL`: Provider/model identifier. Accepts:
  - `openai:<model>`
  - `openai-compatible:<model>`
  - `anthropic:<model>`
  - `ollama:<model>`
  - Bare model names when custom `OPENAI_BASE_URL` or `BASE_URL` is set.
- `OPENAI_BASE_URL` / `BASE_URL`: Base URL for OpenAI-compatible inference backends (vLLM, LM Studio, Ollama, OpenRouter, etc.).
- `OPENAI_API_KEY` / `API_KEY`: API key. Automatically defaults to `"dummy"` when custom base URL is used and no key is provided, preventing client validation failures on auth-free local models.

## Development & Test Conventions
- Tests use pytest with asyncio support.
- Exercise templates in `stages/*/exercises/` are paired with solutions in `stages/*/solutions/`.
- Provider contract tests (`stages/stage_2_dag/tests/test_provider_contract.py`) mock HTTP endpoints locally to verify request formatting and schema compliance without spending API tokens.

