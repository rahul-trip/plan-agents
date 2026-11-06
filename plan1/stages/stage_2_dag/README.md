# Stage 2: Dynamic DAG and Parallel Fan-Out

Linear plans bottleneck on latency. Model the plan as a DAG: Task C waits for A and B, while A and B run in parallel. Upgrade the task schema with `depends_on: list[str]`.

## Mechanics

- **The LangGraph gotcha.** `Send` dispatches parallel nodes within a single superstep, but it knows nothing about your plan's `depends_on`. You bridge that with a **custom dispatcher edge** that emits `Send` only for the unblocked frontier.
- **Barrier and fan-in.** All workers converge on a `scheduler` node. It does nothing itself: it is the point where the reducer has merged every delta, and the router can compute the next frontier.
- **`graphlib` is the same idea without the framework.** `get_ready()`, `asyncio.gather`, `done()`, repeat.

## Gotchas (each is pinned in `tests/test_blueprint_gotchas.py`)

| The PDF blueprint... | What really happens | Fix |
|---|---|---|
| Only checks for cycles | `TopologicalSorter` accepts an unknown ID as a new node | Validate IDs first (ex2) |
| Returns an empty `Send` list when nothing is ready | The run stops with **no error and no output** | Raise `DeadlockError` (ex2, ex3) |
| Merges only by add or overwrite | Nothing can be removed from the plan | `status="cancelled"` (Stages 1 and 5) |
| Has a stub replanner | Never fixes the failed task, so it burns all 3 replans | Real replanner (Stage 4) |
| Lets worker exceptions escape | One failure aborts the whole run | Catch and return `status="failed"` (ex3) |

Also:
- The planner returns a `PlanDraft`, not an `ExecutionPlan`. The PDF's plan schema lets an LLM set `status` and `result`, so a plan could mark a task finished with an invented result, and nothing would run. `make_plan` also re-asks on an invalid plan. `args` is a JSON string because providers' structured-output modes restrict free-form dicts (Claude's rewrite empties them).
- A raw `TopologicalSorter` loop can spin forever: a task you never mark `done()` keeps `is_active()` true while `get_ready()` returns nothing (ex1).
- Superstep barriers wait for the slowest task in a batch. That cost buys one atomic state per step, which is what makes checkpointing simple. Measure it in the stretch exercise.
- The LangGraph docs now recommend `stream_events(version="v3")` for new streaming code. This stage uses `astream(stream_mode="updates")` as the PDF does, since it still works and shows each node's delta plainly.

## Read first

- [`graphlib`](https://docs.python.org/3/library/graphlib.html)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents): the orchestrator-worker section
- [LLMCompiler](https://arxiv.org/abs/2312.04511): this stage's architecture as a research paper

## Go deeper

- [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/): where supersteps come from
- [asyncio tasks](https://docs.python.org/3/library/asyncio-task.html): `gather`, `wait`, `TaskGroup`
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

## Exercises

Do them in this order. **ex1 comes before any LangGraph.**

| File | You build | Track |
|---|---|---|
| `ex1_pure_engine` | ~60-line engine with `TopologicalSorter` and `gather` (study workflow #1, 2–3 h) | Custom |
| `ex2_frontier` | Plan validator, frontier, deadlock check | Both |
| `ex3_blueprint_graph` | Completed blueprint with superstep tracing (study workflow #3) | LangGraph |
| `lab_parallel_dag` | **Capstone:** 6-node DAG, parallelism proven from timestamps | LangGraph |
| `stretch_eager_scheduler` | Optional: `FIRST_COMPLETED` dispatch vs the barrier | Custom |

`ex3` imports the reference kit `planagents.dag`, which holds the answers to ex2. Finish ex2 before opening it. No replanner appears here on purpose; failures degrade gracefully until Stage 4.

**Done when** all tests pass and you can explain why the trace shows `[[task_1, task_2], [task_3]]`.
