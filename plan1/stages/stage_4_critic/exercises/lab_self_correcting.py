"""Stage 4 lab and capstone: a self-correcting pipeline. Every layer of this stage, working together.

A worker's failure now climbs a ladder:
1. Local retry (ex2): a transient error retries here, with no planner call.
2. Circuit breaker (ex1): an endpoint that keeps failing stops being called.
3. Critic and replanner (ex3, ex4): whatever still fails is recorded, and the DAG reroutes around it.

Scenario 1, fault: a mock tool returns 403 on one endpoint, and a fallback exists. The critic catches
the failure, records the broken path in failed_plans, and the DAG reroutes around the blocked tool.

Scenario 2, hallucination: a worker cites a quote its sources don't contain. The result looks fine
and raises no error, so only a groundedness check catches it.

The two differ in one important way. A 403 means the tool is broken, so the guard blocks it for the
rest of the run. A hallucination is a bad output from a tool that may be fine, so it can be retried.

This lab imports your ex1 to ex4, so finish those first.

Live run (needs LLM_MODEL in .env):
    uv run python stages/stage_4_critic/solutions/lab_self_correcting.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from planagents.loader import sibling
from planagents.schemas import ExecutionPlan, TaskNode, WorkerSpec

# The planner and replanner are shown this list, and can't plan a worker that isn't on it.
TOOLS = {
    "primary_api": WorkerSpec(description="Fetches the data from the primary endpoint."),
    "fallback_api": WorkerSpec(description="Fetches the same data from a secondary endpoint."),
    "reader": WorkerSpec(description="Reads a source document."),
    "claimer": WorkerSpec(description="Writes a claim with a verbatim supporting quote from its sources."),
    "reporter": WorkerSpec(description="Writes a report from its dependencies' results."),
}


class ForbiddenError(Exception):
    """HTTP 403. Retrying can't help, so this must go to the replanner."""


class TaskGaveUp(Exception):
    """Local retry is exhausted, so this failure moves up to the critic."""


@dataclass
class MockAPI:
    """Fault injection. Records every call that reaches an endpoint.

    blocked: raises ForbiddenError (permanent). down: always raises TransientError.
    flaky: {endpoint: n} raises TransientError for the next n calls, then recovers.
    """

    blocked: set[str] = field(default_factory=set)
    down: set[str] = field(default_factory=set)
    flaky: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def get(self, endpoint: str) -> str:
        self.calls.append(endpoint)
        transient = sibling(__file__, "ex2_two_tier").TransientError
        if endpoint in self.blocked:
            raise ForbiddenError(f"403 Forbidden: {endpoint}")
        if endpoint in self.down:
            raise transient(f"503 Unavailable: {endpoint}")
        if self.flaky.get(endpoint, 0) > 0:
            self.flaky[endpoint] -= 1
            raise transient(f"503 Unavailable: {endpoint}")
        return f"data from {endpoint}"


def make_fault_worker(api: MockAPI, failure_threshold: int = 2):
    """Return an async worker(task, upstream) for the mock tools.

    - 'primary_api' and 'fallback_api' call api.get(<their name>) through a per-endpoint
      CircuitBreaker from ex1, created once with failure_threshold.
    - 'reporter' returns 'report: ' plus its upstream results, and raises ValueError if it has none,
      so an empty report can't pass as finished work.
    - Anything else raises ValueError.
    """
    breaker_type = sibling(__file__, "ex1_breaker").CircuitBreaker
    raise NotImplementedError("keep a dict of breakers by endpoint; write the async worker described above")


def with_local_retry(worker, max_attempts: int = 3):
    """Wrap worker(task, upstream) with tier 1: ex2's run_with_local_retry.

    If it gives up, raise TaskGaveUp with the failure's detail, and the critic takes over.
    """
    run_with_local_retry = sibling(__file__, "ex2_two_tier").run_with_local_retry
    raise NotImplementedError("return an async wrapper; adapt the worker to run_with_local_retry's (task, feedback) signature")


class Claim(BaseModel):
    """What a research worker returns: a statement and the verbatim quote that supports it."""

    statement: str
    quote: str


def claim_json(statement: str, quote: str) -> str:
    return json.dumps({"statement": statement, "quote": quote})


def groundedness_check(task: TaskNode, upstream: dict[str, str]) -> str | None:
    """Critic check for tasks with worker 'claimer'. Others pass automatically (return None).

    Parse task.result as a Claim. Return a reason containing the word 'Hallucination' if the quote
    is blank or doesn't appear verbatim in an upstream result, or a different reason if the JSON is
    invalid. Otherwise None. Deterministic on purpose: it can't be talked out of its answer.
    """
    raise NotImplementedError("implement the check described above")


def critic_check(task: TaskNode, upstream: dict[str, str]) -> str | None:
    """The check the pipeline gives the critic: ex1's Evaluator first, then groundedness_check.

    A blank result fails the Evaluator, whichever worker produced it. Return its reason.
    """
    evaluator = sibling(__file__, "ex1_breaker").Evaluator()
    raise NotImplementedError("evaluate task.result; return verdict.reason if it failed, otherwise groundedness_check")


def reuses_blocked_tool(plan: ExecutionPlan, failed_plans: list) -> str | None:
    """Plan guard for ex4: reject plans using a tool that failed with kind='tool'.

    Use ex3's repeated_failures. Verification failures must NOT block their tool.
    """
    repeated_failures = sibling(__file__, "ex3_anti_thrashing").repeated_failures
    raise NotImplementedError("filter to kind == 'tool', call repeated_failures, return a reason naming the task ids or None")


def build_pipeline(llm, worker, max_replans: int = 2, workers=TOOLS, max_attempts: int = 3):
    """Your ex4 graph, with the worker wrapped by with_local_retry, critic_check as its check,
    reuses_blocked_tool as its plan guard, and this tool registry."""
    build_graph = sibling(__file__, "ex4_critic_command").build_graph
    raise NotImplementedError("return build_graph(...) with all of the above")


async def main() -> None:
    from planagents.llm import get_llm

    api = MockAPI(blocked={"primary_api"})
    graph = build_pipeline(get_llm(), make_fault_worker(api))
    result = await graph.ainvoke(
        {"objective": "Fetch the customer data, preferring the primary endpoint, then write a report on it."},
        {"recursion_limit": 60},
    )
    print("API calls:", api.calls)
    print("Replans:", result["replan_count"])
    print(result["final_output"])


if __name__ == "__main__":
    asyncio.run(main())
