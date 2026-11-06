# Stage 4: Critic Gates, Loop Dampening and Command Routing

Plans break when APIs time out, data is missing or workers hallucinate. You need deterministic verification gates and bounded replanning.

## Mechanics

- **Two-tier recovery.**
  1. *Local retry:* a schema check or transient error retries in the worker, up to N times, without touching the planner.
  2. *Global replan:* a task that can't succeed, such as one needing a data source that doesn't exist, routes back to the replanner to mutate the rest of the DAG.
- **Anti-thrashing.** Keep `failed_plans: list[FailedAttempt]` in global state, inject it into the replanner prompt as negative constraints, and enforce a hard `max_replans`.
- **`Command`.** `Command(update={...}, goto="replanner")` updates state and routes atomically, so a separate edge function can't see stale state.

## Gotchas

- **Negative prompts aren't enforcement.** An LLM can regenerate the failed path anyway. Add a deterministic guard that rejects it before anything runs.
- **Place the critic at the single post-fan-in barrier.** With one critic per superstep, two parallel failures produce one replan.
- **`Command` and static edges both fire.** Don't add an edge out of a node that routes with `Command`.
- **Annotate destinations**, e.g. `Command[Literal["replanner", "synthesizer"]]`. LangGraph uses this to know the graph's edges.
- **The recursion limit won't save you.** It defaults to 10,007 supersteps. Set `recursion_limit` explicitly, and keep `max_replans` as the real control.
- **Hallucinations raise no error.** Output can be well-formed and wrong, so success status proves nothing. The groundedness check here requires a verbatim quote from a source. That's a narrow check: it catches fabricated quotes, not faulty reasoning from a real one.
- **A replan must not discard finished work.** New tasks may depend on completed ones.
- **Treat a replan as untrusted input.** Validate it, reject a bad one, and re-ask, with a bound. A reused task ID is a real trap: `failed_plans` is keyed by ID, so that task's next failure would look already recorded and never trigger a replan.
- **A bad output isn't a broken tool.** A 403 should block the tool for the rest of the run. A hallucination shouldn't, or one bad answer would permanently remove your only research tool. `FailedAttempt.kind` keeps these apart.
- **A planner can't choose a tool it was never told about.** Pass a worker registry, so the prompt lists the tools and invented ones are rejected. Without one, a real model invented workers here, called no API at all, and reported success.
- **Never let the LLM write runtime state.** If the plan schema has `status` and `result`, a planner can mark a task completed with an invented result, and nothing runs. That's why LLMs get `PlanDraft`, and only code creates `TaskNode`s.
- **An empty replan is giving up, not success.** Mark it degraded.
- **Breakers see overlapping calls.** A slow call from before the outage must not close an open circuit when it finally succeeds.
- **`failed_plans` vs `failed_traces`:** the PDF uses both names for one thing. This repo uses `failed_plans`.

## Read first

- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657): a failure taxonomy, and a good lens for this stage
- [Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Timeouts, retries and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)

## Go deeper

- [Reflexion](https://arxiv.org/abs/2303.11366) and [Self-Refine](https://arxiv.org/abs/2303.17651): LLM critics, as a contrast with deterministic gates
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495): verifying claims
- [Magentic-One](https://arxiv.org/abs/2411.04468): ledger-based replanning
- [Tenacity](https://tenacity.readthedocs.io/en/latest/): the production retry library, once you've written your own

## Exercises

| File | You build | Track |
|---|---|---|
| `ex1_breaker` | Retry and circuit-breaker decorators, `Evaluator` | Custom |
| `ex2_two_tier` | Local retry vs replan, proven by planner call counts | LangGraph |
| `ex3_anti_thrashing` | Negative constraints, deterministic guard, `max_replans` | Both |
| `ex4_critic_command` | Critic node routing with `Command` | LangGraph |
| `lab_self_correcting` | **Capstone:** 403 fault with reroute, plus hallucination recovery | LangGraph |

**Done when** all tests pass and you can say why `test_replanner_that_repeats_the_blocked_path_is_rejected` still passes against an LLM that ignores its prompt.
