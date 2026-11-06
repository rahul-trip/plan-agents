# Master Plan: Plan-Based Multi-Agent Systems

Goal: go from a sequential task list to a self-healing, durable DAG orchestrator with isolated workers, and build each mechanism twice: once in plain Python so you understand the scheduling, and once in LangGraph for production durability.

This plan assumes you already know agents, tools and ReAct. Stage notes cover mechanics and gotchas only.

## How to work through it

1. Read the stage `README.md` (one page), then the "Read first" links.
2. Open `exercises/exN_*.py`, replace each `NotImplementedError`, and run that exercise's tests:
   `uv run pytest stages/<stage>/tests/test_exN_*.py`
3. Compare with `solutions/exN_*.py`, or run the same tests on it with `--target=solutions`.
4. Finish with the stage lab, which is that stage's capstone.

Do the stages in order: each one builds on the mechanisms from the one before it. From Stage 2's `ex3` onward, exercises import reference versions of earlier pieces from `src/planagents/dag.py`, so treat that file as a spoiler until you finish Stage 2's `ex2`.

Where this repo deliberately differs from the PDF, the stage README says why. The main one: LLMs write a `PlanDraft`, and only code creates the PDF's `TaskNode`, which holds runtime state.

## Roadmap

Time estimates are rough. The PDF only gives a figure for the pure-Python engine (2–3 hours).

| Stage | Focus | LangGraph | Custom Python | Capstone | Est. |
|---|---|---|---|---|---|
| [1](stages/stage_1_linear/README.md) | Linear plan-and-solve | `StateGraph`, `with_structured_output`, custom reducers | `collections.deque` loop, Pydantic state | Due-diligence agent that generates, executes and mutates a step list | 4–6 h |
| [2](stages/stage_2_dag/README.md) | Dynamic DAG, parallel fan-out | `Send`, map-reduce, superstep sync | `graphlib.TopologicalSorter`, `asyncio.gather` | Dependency-aware scheduler dispatching independent nodes concurrently | 5–7 h |
| [3](stages/stage_3_isolation/README.md) | Context isolation, subgraphs | Compiled subgraphs, private worker schemas, state mapping | Scoped workers with immutable state slices | SQL + Python + Search system with zero context leakage | 6–8 h |
| [4](stages/stage_4_critic/README.md) | Critic gates, loop dampening | `Command`, negative prompt injection | Retry and circuit-breaker decorators, `Evaluator` | Pipeline that catches hallucinations and replans without thrashing | 5–7 h |
| [5](stages/stage_5_durability/README.md) | Durability, HITL, time travel | `AsyncPostgresSaver`, `interrupt()`, forking and replay | Append-only Postgres event log, state hydration | Orchestrator with plan approval and mid-graph crash recovery | 6–8 h |

## Progress checklist

- [ ] **Stage 1:** ex1 deque loop · ex2 reducer · ex3 LangGraph loop · lab due diligence
- [ ] **Stage 2:** ex1 pure engine · ex2 validator and frontier · ex3 blueprint graph · lab 6-node DAG · stretch eager scheduler
- [ ] **Stage 3:** ex1 scoped workers · ex2 blackboard · ex3 budget guard · ex4 subgraphs · lab multi-domain
- [ ] **Stage 4:** ex1 breaker and evaluator · ex2 two-tier recovery · ex3 anti-thrashing · ex4 critic and `Command` · lab self-correcting pipeline
- [ ] **Stage 5:** ex1 event log · ex2 crash recovery · ex3 HITL · ex4 time travel · lab production orchestrator

## PDF coverage matrix

Every item in the source plan ([docs/source-plan.pdf](docs/source-plan.pdf)) maps to code and a test. Stage paths are under `stages/`.

| PDF item | Where |
|---|---|
| S1: Plan and Step schemas | `src/planagents/schemas.py`; `stage_1_linear/ex1` |
| S1: global vs scratchpad state | `stage_1_linear/ex1`, `test_scratchpad_is_discarded...` |
| S1: custom `while plan_queue` loop | `stage_1_linear/ex1` |
| S1: planner, executor, replanner graph | `stage_1_linear/ex3` |
| S1: reducer updating in place by id | `stage_1_linear/ex2` |
| S1: lab, due diligence with a dynamic step | `stage_1_linear/lab_due_diligence` |
| S2: `depends_on` schema | `src/planagents/schemas.py` |
| S2: custom dispatcher edge emitting `Send` | `stage_2_dag/ex3` |
| S2: `TopologicalSorter` + `asyncio.gather` | `stage_2_dag/ex1` |
| S2: scheduler barrier, convergence via reducer | `stage_2_dag/ex3` |
| S2: lab, 6-node DAG proven by timestamps | `stage_2_dag/lab_parallel_dag` |
| S3: blackboard, minimal `WorkerInputState` | `stage_3_isolation/ex2` |
| S3: immutable state slices | `stage_3_isolation/ex1` |
| S3: hierarchical subgraphs | `stage_3_isolation/ex4` |
| S3: lab, CodeGenerator and DataAnalyst subgraphs | `stage_3_isolation/ex4`, `lab_multi_domain` |
| Capstone S3: SQL + Python + Search, no leakage | `stage_3_isolation/lab_multi_domain` (adds Search) |
| S4: local retry vs global replan | `stage_4_critic/ex2` |
| S4: `failed_plans`, negative constraints, `max_replans` | `stage_4_critic/ex3` |
| S4: `Command` from the critic | `stage_4_critic/ex4` |
| S4: retry and circuit breaker, `Evaluator` | `stage_4_critic/ex1` |
| S4: two-tier recovery working together | `stage_4_critic/lab_self_correcting` (retry, breaker, `Evaluator`, critic, replan) |
| S4: lab, 403 fault with a fallback tool | `stage_4_critic/lab_self_correcting` |
| Capstone S4: catches hallucinations | `stage_4_critic/lab_self_correcting` (groundedness scenario) |
| S5: `AsyncPostgresSaver` checkpointing | `stage_5_durability/ex2`, `lab_production` |
| S5: `interrupt()` plan review: inspect, edit, prune | `stage_5_durability/ex3` |
| S5: `get_state_history`, `update_state`, fork | `stage_5_durability/ex4` |
| S5: custom append-only log and hydration | `stage_5_durability/ex1` |
| S5: lab, gate before destructive action, edit mid-flight | `stage_5_durability/ex3`, `lab_production` |
| Capstone S5: mid-graph crash recovery | `stage_5_durability/ex2`, `lab_production` |
| Starter blueprint | `stage_2_dag/blueprint_pdf.py` (as published), `ex3` (completed) |
| Study workflow 1: pure-Python engine first | `stage_2_dag/ex1` |
| Study workflow 2: token budget fixture | `stage_3_isolation/ex3`; the `budget_guard` fixture is provided in `tests/conftest.py` and proven inside a `worker_executor` |
| Study workflow 3: trace superstep boundaries | `stage_2_dag/ex3`, `test_superstep_trace...` |
