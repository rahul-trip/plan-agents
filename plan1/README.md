# Plan-Based Multi-Agent Systems: a Hands-On Curriculum

Build plan-based agents from a sequential task list up to a durable, self-healing DAG orchestrator, once in plain Python and once in LangGraph. Based on [docs/source-plan.pdf](docs/source-plan.pdf).

**Start with [PLAN.md](PLAN.md)**: roadmap, progress checklist and PDF coverage.

## Setup

Needs [uv](https://docs.astral.sh/uv/). Docker is only needed for Stage 5.

```bash
uv sync
uv run pytest stages/stage_1_linear/tests/test_ex2_reducer.py   # fails until you solve it
```

No API key is needed. By default, tests use a scripted fake LLM and make no external calls, so they're fast, free and deterministic. Live tests only run with `-m live`.

## Layout

```
PLAN.md             Master plan
stages/stage_N_*/
  README.md         One page: mechanics, gotchas, reading, exercises
  exercises/        Your work: replace each NotImplementedError
  solutions/        Reference implementations
  tests/            One suite, run against either folder
src/planagents/     Shared schemas, fake LLM, LLM config, tracing, DAG kit
```

## Commands

```bash
uv run pytest stages/stage_2_dag                      # your exercises
uv run pytest stages/stage_2_dag --target=solutions   # reference solutions
docker compose up -d --wait                           # Postgres for Stage 5
uv run pytest -m "not postgres and not live"          # skip anything needing services
uv run pytest -m live --target=solutions              # real LLM and public APIs
uv run python scripts/check_links.py                  # re-check the reading links
```

Any `-m` you pass replaces the default, so always include `not live` in your own filters.

A few tests pass from the start, such as `test_blueprint_gotchas.py` and `test_provider_contract.py`. They pin library behaviour rather than checking your code.

`scripts/make_exercises.py` regenerates exercises from solutions. It leaves existing files alone unless you pass `--force`, which **overwrites your work**.

## Using a real LLM

```bash
cp .env.example .env    # then set LLM_MODEL and any key it needs
```

`LLM_MODEL` takes `provider:model`, for example `anthropic:claude-opus-5-5`, `openai:<model>`, `openai-compatible:<model>` or `ollama:<model>`. `src/planagents/llm.py` is the only file that knows about providers.

To connect to any OpenAI-compatible inference service (vLLM, LM Studio, Ollama, LocalAI, OpenRouter, Groq, Together, etc.), specify `OPENAI_BASE_URL` (or `BASE_URL`) and `OPENAI_API_KEY` (or `API_KEY`) in `.env`. For local endpoints that do not require an API key, `OPENAI_API_KEY` can be left blank (it defaults to `dummy` so the client runs without authentication errors).

For Claude, that file also turns on server-side refusal fallbacks (`fallbacks: "default"`), so a safety-classifier refusal is retried on a suitable model. Remove the two keys from `_ANTHROPIC_DEFAULTS` to opt out.

Support differs by provider. Ollama has run the labs for real. Claude and OpenAI have only been tested through their real clients against a local stub (`test_provider_contract.py`), which checks our requests and how we read replies, not that their APIs accept them. Expect to find out on your first live run.

Live runs cost money on paid providers. Stage 3's code-generation worker executes what the LLM writes, in a temp directory that is **not a sandbox**, so read that stage's safety note first.
