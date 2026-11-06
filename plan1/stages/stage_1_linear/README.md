# Stage 1: Linear Plan-and-Solve

**Loop:** Plan → Execute Next Step → Update State → Replan or Finish. Get this right before adding concurrency.

## Mechanics

- **Deterministic plan schema.** A `Plan` is a list of `Step` objects with `id`, `description`, `assigned_worker` and `status`, produced with structured output.
- **Global vs scratchpad state.** Global state (objective, plan, results) lasts the whole run. Scratchpad state (the active step's tool history) dies with the step. This is what stops a long run's prompts growing without bound.
- **Reducers.** In LangGraph, nodes return deltas. `operator.add` would append a second copy of a completed step, so use a reducer that replaces by `id`.

## Gotchas

- Replanning can grow a queue forever, so always cap it (`max_steps`, `recursion_limit`). The default recursion limit is 10,007, far too high to catch this quickly.
- A reducer can't delete. A step the replanner drops must become `status="cancelled"`, or it would run anyway.
- Type worker names as a `Literal`. With a plain `str`, a real local model invented `assigned_worker="Security Specialist"` in this lab and crashed the run with a `KeyError`. The scripted tests couldn't catch that, which is why you should run each lab against a real LLM at least once.
- Read replies with `reply_text(reply)`, not `reply.content`. Claude returns a list of blocks when it thinks. The fake LLM does the same on purpose, so the mistake fails here rather than live.
- Don't let the LLM write runtime state. `Plan` holds `PlannedStep`s, and `to_steps()` makes `Step`s that start pending, so a reply can't mark a step completed. A replanner can also repeat a finished step's ID, so drop those too.
- In an audit, "we didn't look" must never read as "none found".
- Don't leave business-critical triggers to the LLM. In the lab, code decides *that* a remediation step is needed, and the LLM only words it.
- Use `structured(llm, Schema)` from `planagents.llm`. It passes `method="json_schema"`, because the `ChatAnthropic` default of forced tool calling isn't supported on current Claude models.

## Read first

- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091): sections 1–3 are enough
- [LangChain: Plan-and-Execute Agents](https://blog.langchain.com/planning-agents/)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): state, reducers, nodes
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

## Go deeper

- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents): when a plan loop isn't worth it
- [ReWOO](https://arxiv.org/abs/2305.18323): plan first, then execute without replanning
- [Graph API how-to](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- Lab APIs: [GitHub licenses](https://docs.github.com/en/rest/licenses/licenses), [OSV](https://google.github.io/osv.dev/api/)

## Exercises

| File | You build | Track |
|---|---|---|
| `ex1_custom_loop` | `while plan_queue` loop on a `deque`, with disposable scratchpads | Custom |
| `ex2_reducer` | `merge_steps`, which updates in place by id | LangGraph |
| `ex3_langgraph_loop` | planner, executor and replanner as a `StateGraph` | LangGraph |
| `lab_due_diligence` | **Capstone:** GitHub audit agent that adds a remediation step on a critical vulnerability | LangGraph |

```bash
uv run pytest stages/stage_1_linear                      # your exercises
uv run pytest stages/stage_1_linear --target=solutions   # reference solutions
```

**Done when** all tests pass and you can explain why `test_scratchpad_is_discarded_after_each_step` plants a canary string.

Optional: `uv run pytest stages/stage_1_linear -m live --target=solutions` calls the real GitHub and OSV APIs. The lab's `main` also needs `LLM_MODEL`.
