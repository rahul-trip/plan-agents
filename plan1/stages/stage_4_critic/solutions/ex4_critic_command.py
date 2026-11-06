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
    # >>> TODO: return a reason string for reused IDs; validate completed + new tasks with validate_plan, returning any PlanValidationError as text
    reused = {task.id for task in plan.tasks} & set(tasks)
    if reused:
        return f"Task IDs already used: {sorted(reused)}. Use new IDs."
    completed = [t for t in tasks.values() if t.status == "completed"]
    try:
        validate_plan(ExecutionPlan(tasks=[*completed, *plan.tasks]))
    except PlanValidationError as exc:
        return str(exc)
    return None
    # <<<


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
    # >>> TODO: write planner, critic, worker, replanner and synthesizer; annotate Command destinations
    async def planner(state: CriticState) -> dict:
        plan = await make_plan(llm, f"Plan a task DAG for: {state['objective']}", workers)
        return {"tasks": validate_plan(plan), "replan_count": 0, "degraded": False}

    async def critic(state: CriticState) -> Command[Literal["worker", "replanner", "synthesizer"]]:
        tasks = dict(state["tasks"])
        recorded = {attempt.task_id for attempt in state.get("failed_plans", [])}
        rejected: dict[str, TaskNode] = {}
        attempts: list[FailedAttempt] = []

        if check:
            for task in tasks.values():
                if task.status == "completed":
                    upstream = {dep: tasks[dep].result or "" for dep in task.depends_on}
                    reason = check(task, upstream)
                    if reason:
                        rejected[task.id] = task.model_copy(update={"status": "failed", "result": reason})
                        attempts.append(
                            FailedAttempt(task_id=task.id, tool=task.worker, reason=reason, kind="verification")
                        )
            tasks.update(rejected)

        for task in tasks.values():
            if task.status == "failed" and task.id not in recorded and task.id not in rejected:
                attempts.append(FailedAttempt(task_id=task.id, tool=task.worker, reason=task.result or "unknown"))

        update: dict = {}
        if rejected:
            update["tasks"] = rejected
        if attempts:
            update["failed_plans"] = attempts
            if state["replan_count"] < max_replans:
                return Command(update=update, goto="replanner")
            pending = [t.id for t in tasks.values() if t.status == "pending"]
            update["tasks"] = {**rejected, **cancel_with_dependents(tasks, pending)}
            update["degraded"] = True
            return Command(update=update, goto="synthesizer")

        if not any(t.status == "pending" for t in tasks.values()):
            return Command(update=update, goto="synthesizer")

        check_deadlock(tasks)
        sends = [
            Send("worker", {"task": t, "upstream": {d: tasks[d].result or "" for d in t.depends_on}})
            for t in ready_frontier(tasks)
        ]
        return Command(update=update, goto=sends)

    async def worker_node(payload: dict) -> dict:
        task = payload["task"]
        try:
            update = {"status": "completed", "result": await worker(task, payload["upstream"])}
        except Exception as exc:
            update = {"status": "failed", "result": f"{type(exc).__name__}: {exc}"}
        return {"tasks": {task.id: task.model_copy(update=update)}}

    async def replanner(state: CriticState) -> Command[Literal["critic", "synthesizer"]]:
        tasks = state["tasks"]
        failed_plans = state["failed_plans"]
        stale = {
            t.id: t.model_copy(update={"status": "cancelled"})
            for t in tasks.values()
            if t.status in ("failed", "pending")
        }
        history = "\n".join(f"- {a.tool} on {a.task_id}: {a.reason}" for a in failed_plans)
        done = {t.id: t.result for t in tasks.values() if t.status == "completed"}
        unfinished = "\n".join(f"- {t.id}: {t.instruction}" for t in tasks.values() if t.status == "pending")
        broken = {a.tool for a in failed_plans if a.kind == "tool"}
        usable = {name: spec for name, spec in (workers or {}).items() if name not in broken}
        available = f"Use only these workers:\n{describe_workers(usable)}\n" if workers else ""
        prompt = (
            f"Objective: {state['objective']}\nCompleted tasks: {done}\n"
            f"Unfinished tasks being replaced:\n{unfinished or '(none)'}\n"
            f"These paths failed; do not repeat them:\n{history}\n{available}"
            "Return only the new tasks, with new IDs. In depends_on, use only other new task IDs or these "
            f"completed IDs: {sorted(done)}. Return no tasks to give up."
        )
        give_up = Command(
            update={"tasks": stale, "replan_count": state["replan_count"] + 1, "degraded": True},
            goto="synthesizer",
        )
        for _ in range(max_guard_rejections + 1):
            draft = await structured(llm, PlanDraft).ainvoke(prompt)
            try:
                plan = tasks_from_draft(draft, workers)
                problem = replan_problem(plan, tasks) or (plan_guard and plan_guard(plan, failed_plans))
            except PlanValidationError as exc:
                problem = str(exc)
            if not problem:
                if not plan.tasks:
                    return give_up
                fresh = {task.id: task for task in plan.tasks}
                return Command(
                    update={"tasks": {**stale, **fresh}, "replan_count": state["replan_count"] + 1},
                    goto="critic",
                )
            prompt += f"\nREJECTED: {problem}"
        return give_up

    async def synthesizer(state: CriticState) -> dict:
        lines = [f"{t.id} [{t.status}]: {t.result}" for t in state["tasks"].values()]
        prefix = "DEGRADED RESULT\n" if state.get("degraded") else ""
        return {"final_output": prefix + "\n".join(lines)}

    builder = StateGraph(CriticState)
    builder.add_node("planner", planner)
    builder.add_node("critic", critic)
    builder.add_node("worker", worker_node)
    builder.add_node("replanner", replanner)
    builder.add_node("synthesizer", synthesizer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "critic")
    builder.add_edge("worker", "critic")
    builder.add_edge("synthesizer", END)
    return builder.compile()
    # <<<
