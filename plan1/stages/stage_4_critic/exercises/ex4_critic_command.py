"""Stage 4, ex4: the critic node. State update and routing in one atomic Command.

The critic replaces the scheduler barrier from Stage 2. Because exactly one critic runs per
superstep, after fan-in, parallel workers can't race to route. It decides everything:

    new failure, budget left -> record it, Command(update=..., goto='replanner')
    new failure, budget gone -> Command(update=..., goto='synthesizer'), with degraded=True
    nothing pending          -> Command(goto='synthesizer')
    otherwise                -> Command(goto=[Send('worker', ...), ...]) for the frontier

Don't also add a static edge out of the critic: static edges run in addition to Command routing.
"""

from __future__ import annotations

import operator
from typing import Annotated, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel

from planagents.dag import (
    PlanValidationError,
    Workers,
    cancel_with_dependents,
    check_deadlock,
    describe_workers,
    make_plan,
    merge_tasks,
    ready_frontier,
    tasks_from_draft,
    validate_plan,
)
from planagents.llm import structured
from planagents.schemas import ExecutionPlan, PlanDraft, TaskNode


class FailedAttempt(BaseModel):
    """kind='tool': the worker raised, so the tool itself is suspect.
    kind='verification': the worker ran but its output failed a check. The tool may be fine."""

    task_id: str
    tool: str
    reason: str
    kind: Literal["tool", "verification"] = "tool"


class CriticState(TypedDict, total=False):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    failed_plans: Annotated[list[FailedAttempt], operator.add]
    replan_count: int
    degraded: bool
    final_output: str


Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]
# Verifies a completed result; returns None if fine, else why not. It runs at every barrier, so
# keep it pure and cheap.
Check = Callable[[TaskNode, dict[str, str]], str | None]
# Inspects a proposed replan; returns None if acceptable, else the rejection reason for the LLM.
PlanGuard = Callable[[ExecutionPlan, list[FailedAttempt]], str | None]


def replan_problem(plan: ExecutionPlan, tasks: dict[str, TaskNode]) -> str | None:
    """Why this replan can't be merged, or None if it can.

    - It may not reuse an existing task ID. failed_plans is keyed by ID, so a reused ID's next
      failure would look already recorded, and would never be replanned.
    - Completed tasks plus the new tasks must form a valid plan, so a new task can depend on
      finished work but not on anything that was cancelled or failed.
    """
    raise NotImplementedError("return a reason string for reused IDs; validate completed + new tasks with validate_plan, returning any PlanValidationError as text")


def build_graph(
    llm,
    worker: Worker,
    max_replans: int = 2,
    check: Check | None = None,
    plan_guard: PlanGuard | None = None,
    max_guard_rejections: int = 2,
    workers: Workers | None = None,
):
    """planner -> critic -> (workers | replanner | synthesizer), with workers and the replanner
    both returning to the critic.

    planner: make_plan(llm, prompt, workers) from planagents.dag, plus replan_count=0, degraded=False.

    critic: run `check` on completed tasks; a rejection marks the task failed, with the reason as
    its result and FailedAttempt kind='verification'. Worker failures are kind='tool'. Then route
    per the module docstring. When the budget is gone, cancel everything still pending.

    replanner: ask structured(llm, PlanDraft) and convert with tasks_from_draft(draft, workers). The
    prompt needs the objective, completed results, the unfinished tasks being replaced, failed_plans,
    and, if workers were given, describe_workers for those without a kind='tool' failure: it can't
    reroute to a tool it was never told about, and shouldn't be offered a broken one. Reject
    a plan failing tasks_from_draft, replan_problem or plan_guard by re-asking with the reason
    appended, up to max_guard_rejections. On success: failed and pending tasks become cancelled, new
    tasks merge in, replan_count + 1, back to the critic. An empty plan means giving up: like running
    out of rejections, that cancels the remainder, sets degraded=True and replan_count + 1, and goes
    to the synthesizer, so an unrecovered failure can't read as a success.

    synthesizer: final_output has a line per task naming its status and result, and starts with the
    word DEGRADED when degraded is true.
    """
    raise NotImplementedError("write planner, critic, worker, replanner and synthesizer; annotate Command destinations")
