# Project Memory

## Key Context
- Workspace repository: `/Users/cnc/sources/plan based agent`
- Active codebase: `plan1/`
- Python version: 3.11 with dependencies in `.venv/`

## Recent Changes & Decisions
1. **OpenAI-Compatible Inference Service Support**:
   - Updated `plan1/src/planagents/llm.py` so that `get_llm()` transparently supports any OpenAI-compatible inference service.
   - Supported environment variables:
     - `OPENAI_BASE_URL` or `BASE_URL` (or legacy `OPENAI_API_BASE` / `LLM_BASE_URL`) for specifying the service endpoint.
     - `OPENAI_API_KEY` or `API_KEY` (or `LLM_API_KEY`).
   - Automatically provides a fallback `"dummy"` key when a custom base URL is used and no API key is specified, allowing auth-free local inference engines (vLLM, LM Studio, Ollama, LocalAI) to run without error.
   - Accepts prefixes `openai:`, `openai-compatible:`, `openai_compatible:`, or bare model names when `BASE_URL` / `OPENAI_BASE_URL` is set.
   - Updated `.env.example` and created `.env` with documentation for base URL and API key configuration.
   - Added automated provider contract regression tests in `stages/stage_2_dag/tests/test_provider_contract.py` validating base URL and environment variable handling.

