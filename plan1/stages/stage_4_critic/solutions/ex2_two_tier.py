"""Stage 4, ex2: two-tier recovery.

Tier 1 (local retry): transient errors and schema failures retry in the worker, up to N times.
This never touches the planner, which is the expensive part.
Tier 2 (global replan): a task that can't succeed goes back to the planner. Here a replan just
counts; ex4 makes it rewrite the DAG.

You could also retry with LangGraph's RetryPolicy on add_node. A hand-written loop is used here
because schema failures must feed the validation error back to the LLM on the next attempt.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError

from planagents.dag import make_plan, merge_tasks, ready_frontier, validate_plan
from planagents.schemas import TaskNode

FailureKind = Literal["transient", "schema", "permanent"]


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def classify_failure(error: Exception) -> FailureKind:
    """TransientError and asyncio/built-in TimeoutError: transient. ValidationError: schema.
    Everything else: permanent. Unknown errors must not be retried blindly."""
    # >>> TODO: map exception types to a FailureKind as described
    if isinstance(error, (TransientError, TimeoutError, asyncio.TimeoutError)):
        return "transient"
    if isinstance(error, ValidationError):
        return "schema"
    return "permanent"
    # <<<


class TaskFailure(BaseModel):
    kind: FailureKind
    attempts: int
    detail: str


Worker = Callable[[TaskNode, str | None], Awaitable[str]]


async def run_with_local_retry(
    task: TaskNode, worker: Worker, max_attempts: int = 3
) -> tuple[str | None, TaskFailure | None]:
    """Call worker(task, feedback) and return (result, None) or (None, failure).

    Retry transient and schema failures up to max_attempts in total. On a schema failure, pass the
    error text as `feedback` on the next attempt. A permanent failure stops at once.
    """
    # >>> TODO: loop up to max_attempts using classify_failure; only schema errors set feedback
    feedback = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await worker(task, feedback), None
        except Exception as exc:
            kind = classify_failure(exc)
            if kind == "permanent" or attempt == max_attempts:
                return None, TaskFailure(kind=kind, attempts=attempt, detail=f"{type(exc).__name__}: {exc}")
            feedback = str(exc) if kind == "schema" else feedback
    raise AssertionError("unreachable")
    # <<<


class TwoTierState(TypedDict, total=False):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    replan_requests: int
    final_output: str


def build_graph(llm, worker: Worker, max_attempts: int = 3):
    """planner -> scheduler -> worker fan-out -> scheduler.

    The planner uses make_plan from planagents.dag and sets replan_requests to 0. The worker node
    uses run_with_local_retry. A failure gives status='failed', with the failure's detail as result.
    The router sends failed tasks to 'request_replan' (which adds 1 to replan_requests, then ends)
    and otherwise fans out, finishing when nothing is pending.
    """
    # >>> TODO: compile the graph: only the planner calls the planning LLM, retries stay in the worker node
    async def planner(state: TwoTierState) -> dict:
        plan = await make_plan(llm, f"Plan a DAG for: {state['objective']}")
        return {"tasks": validate_plan(plan), "replan_requests": 0}

    async def scheduler(state: TwoTierState) -> dict:
        return {}

    async def worker_node(payload: dict) -> dict:
        task = payload["task"]
        result, failure = await run_with_local_retry(task, worker, max_attempts)
        update = (
            {"status": "failed", "result": failure.detail}
            if failure
            else {"status": "completed", "result": result}
        )
        return {"tasks": {task.id: task.model_copy(update=update)}}

    async def request_replan(state: TwoTierState) -> dict:
        return {"replan_requests": state["replan_requests"] + 1}

    def route(state: TwoTierState):
        tasks = state["tasks"]
        if any(t.status == "failed" for t in tasks.values()):
            return "request_replan"
        ready = ready_frontier(tasks)
        return [Send("worker", {"task": task}) for task in ready] if ready else END

    builder = StateGraph(TwoTierState)
    builder.add_node("planner", planner)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker", worker_node)
    builder.add_node("request_replan", request_replan)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "scheduler")
    builder.add_conditional_edges("scheduler", route, ["worker", "request_replan", END])
    builder.add_edge("worker", "scheduler")
    builder.add_edge("request_replan", END)
    return builder.compile()
    # <<<
