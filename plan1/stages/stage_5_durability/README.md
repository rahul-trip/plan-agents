# Stage 5: Durability, HITL and Time-Travel Debugging

In production, a 15-step DAG can't restart from scratch because step 12 failed or needs a human sign-off.

## Setup

```bash
docker compose up -d --wait      # Postgres on localhost:55432
uv run pytest stages/stage_5_durability
```

Tests marked `postgres` skip with a clear message if the database isn't running. Most HITL and time-travel tests run twice, in memory and on Postgres.

## Mechanics

- **Checkpointing.** `AsyncPostgresSaver` persists every superstep's state. Call `await saver.setup()` once to create its tables.
- **Durability modes.** Pass `durability=` when invoking. `"exit"` saves only when the run exits, so a crash mid-run can't be recovered. `"async"` saves while the next step runs, so a crash may lose some checkpoints. `"sync"` saves before the next step starts, which is the safest and slowest, and is what this stage uses.
- **HITL.** `interrupt()` in a `plan_review` node lets a person or API caller inspect the DAG as JSON, edit arguments or prune nodes, then continue with `Command(resume=...)`.
- **Time travel.** `aget_state_history` finds the checkpoint before a failure, `aupdate_state` patches it onto a new fork, and `ainvoke(None, fork_config)` resumes from there.
- **Custom equivalent.** An append-only event log. State is never stored, only derived by replaying events.

## Gotchas (tested here unless marked)

- **A resumed node restarts from its top**, so code before `interrupt()` runs again. Plan in a separate node (or every resume calls the LLM) and put side effects after the interrupt.
- **Never raise after an `interrupt()`.** LangGraph saves the resume value, so it replays on every later resume, and one typo locks the thread for good, even across restarts. Validate the answer, and on a bad one route back and ask again. The docs recommend this same pattern.
- **Never wrap `interrupt()` in a broad `try/except`.** It pauses by raising, so `except Exception` swallows the pause. Wrap only your validation.
- **A malformed approval isn't a rejection.** Treating it as one silently cancels work a person meant to approve. Ask again.
- **Gates belong to code.** If the LLM flags what's destructive, not flagging a task skips the gate. Here a worker registry decides, and a reviewer can add a gate during plan review.
- **Parallel interrupts resume together**, with a map of `{interrupt.id: value}`, so an API must expose those ids. A sibling that already finished doesn't re-run.
- **Recovery is at-least-once, not exactly-once.** A task in flight at the crash runs again, as the crash test shows. Real side effects need an idempotency key derived from durable state.
- **A fork can silently do nothing.** In a `Command`-routed graph, `update_state` without `as_node` can leave a fork with nothing to run, and the resume returns with no error. Set `as_node` and check `.next`.
- **After a fork, the thread's latest state is the fork.** The failed history still exists, but keep its config if you want to come back to it.
- **A fork re-runs parallel siblings.** The checkpoint before a failed superstep predates a sibling that succeeded in it, so that sibling runs again.
- **Replay isn't free.** `ainvoke(None, snapshot.config)` re-runs every later node, including LLM calls and interrupts. (The test covers workers; LLM calls and interrupts are from the docs.)
- **Subgraph limits.** A default subgraph is one superstep to its parent, so you can't time travel inside it without `checkpointer=True` on that subgraph.
- **Pydantic in state needs an allowlist.** Without one, LangGraph warns on load that unregistered types will be blocked in a future version. `planagents.dag.checkpoint_serde()` registers `TaskNode`. Define checkpointed models in an importable module: `__main__` or dynamically loaded classes make poor allowlist entries.
- **Checkpoints hold your data in cleartext** and grow over time. Plan a retention policy. (Not tested.)

The docs now recommend `stream_events(version="v3")`, where interrupts appear on `stream.interrupts`. This stage uses `ainvoke` and `result["__interrupt__"]`, which the docs say still work.

## Read first

- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers): durability modes and pending writes
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance): retries and idempotency keys

## Go deeper

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution), [`langgraph-checkpoint-postgres`](https://pypi.org/project/langgraph-checkpoint-postgres/)
- [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [Temporal workflow execution](https://docs.temporal.io/workflow-execution): durable execution without a graph model
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

## Exercises

| File | You build | Track |
|---|---|---|
| `ex1_event_log` | Append-only Postgres log, hydration, resume | Custom |
| `ex2_crash_recovery` | `AsyncPostgresSaver`, survives a real `os._exit` mid-DAG | LangGraph |
| `ex3_hitl` | Plan review (approve, edit, prune) and destructive-action gate | LangGraph |
| `ex4_time_travel` | Find, patch, fork and resume | LangGraph |
| `lab_production` | **Capstone:** orchestrator with approval, mid-flight edits and crash recovery | LangGraph |

`crash_runner.py` is a test helper that runs a DAG in its own process, so the crash is real.

**Done when** all tests pass, Postgres ones included, and you can explain why `c` runs twice but `b` runs once in `test_real_process_crash_resumes_without_redoing_finished_work`.
