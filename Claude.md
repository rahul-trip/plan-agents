# Claude.md - Project Guide & Assistant Integration

## Project Overview
This repository contains `planagents`, a modular framework for plan-based agent architectures implemented in Python across progressive stages:
- **Stage 1 (Linear Loop)**: Planner -> Executor -> Replanner sequential execution.
- **Stage 2 (DAG)**: Dependency-directed acyclic graph task execution with concurrency and dependency tracking.
- **Stage 3 (Isolation)**: Worker context isolation, blackboard architecture, token budget guards, and subgraphs.
- **Stage 4 (Critic)**: Two-tier validation, circuit breaker, anti-thrashing negative constraints, and replanning.
- **Stage 5 (Durability)**: Event log hydration, LangGraph checkpointing, human-in-the-loop review/gates, and time-travel replay/forking.

## LLM Configuration & Providers (`planagents.llm`)
The central provider interface is located at `plan1/src/planagents/llm.py`. All LLM initialization is routed through `get_llm()`.

Supported environment variables in `.env`:
- `LLM_MODEL`: Provider and model name in `provider:model` format (e.g. `anthropic:claude-opus-5-5`, `openai:gpt-4o`, `openai-compatible:<model-name>`, `ollama:<model-name>`). Bare model names (e.g. `meta-llama/Llama-3-8b`) are also supported when `OPENAI_BASE_URL` or `BASE_URL` is set.
- `OPENAI_BASE_URL` / `BASE_URL` / `OPENAI_API_BASE`: Endpoint URL for OpenAI and any OpenAI-compatible inference provider (e.g. vLLM, LM Studio, Ollama, LocalAI, OpenRouter, Groq, Together).
- `OPENAI_API_KEY` / `API_KEY`: API key for the OpenAI-compatible service. If connecting to a local service that doesn't need credentials, this can be omitted or left blank (it defaults to `"dummy"` to satisfy client validation).
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude.
- `LLM_STRUCTURED_METHOD`: Structured output method (`json_schema` by default; can be set to `function_calling` for servers lacking strict JSON schema support).

## Running Tests
- Regression & Provider Contract Tests:
  ```bash
  cd plan1 && PYTHONPATH="src:.venv/lib/python3.11/site-packages" python3 -m pytest stages/stage_2_dag/tests/test_provider_contract.py
  ```
- Solutions vs Exercises:
  Stage tests default to `--target=exercises`. To test against reference solutions:
  ```bash
  cd plan1 && PYTHONPATH="src:.venv/lib/python3.11/site-packages" python3 -m pytest stages/stage_1_linear --target=solutions
  ```

