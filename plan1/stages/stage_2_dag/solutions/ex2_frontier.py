"""Stage 2, ex2: the plan validator and frontier logic that LangGraph doesn't give you.

Send dispatches whatever you hand it. It has no idea about depends_on, so your dispatcher has to
compute the unblocked frontier and catch the failure modes test_blueprint_gotchas.py pins down.
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from planagents.schemas import ExecutionPlan, TaskNode


class PlanValidationError(ValueError):
    """The plan can never run correctly."""


class DeadlockError(Exception):
    """Tasks are pending, nothing is ready, and nothing is left to unblock them."""


def validate_plan(plan: ExecutionPlan) -> dict[str, TaskNode]:
    """Return tasks keyed by id, or raise PlanValidationError on duplicate ids, dangling
    depends_on references (name the unknown id in the message), or a cycle. Remember
    TopologicalSorter accepts unknown ids."""
    # >>> TODO: check duplicates, then dangling ids, then run TopologicalSorter(...).prepare() and convert CycleError
    ids = [task.id for task in plan.tasks]
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    if duplicates:
        raise PlanValidationError(f"Duplicate task IDs: {sorted(duplicates)}")
    known = set(ids)
    for task in plan.tasks:
        dangling = set(task.depends_on) - known
        if dangling:
            raise PlanValidationError(f"{task.id} depends on unknown tasks: {sorted(dangling)}")
    try:
        TopologicalSorter({t.id: set(t.depends_on) for t in plan.tasks}).prepare()
    except CycleError as exc:
        raise PlanValidationError(f"Cyclic dependency: {exc.args[1]}") from exc
    return {task.id: task for task in plan.tasks}
    # <<<


def merge_tasks(existing: dict[str, TaskNode] | None, updates: dict[str, TaskNode]) -> dict[str, TaskNode]:
    """Reducer that merges parallel workers' deltas. Handle empty initial state; don't mutate."""
    # >>> TODO: return a new dict: existing overlaid with updates
    return {**(existing or {}), **updates}
    # <<<


def ready_frontier(tasks: dict[str, TaskNode]) -> list[TaskNode]:
    """Pending tasks whose dependencies are all completed. A failed parent never satisfies one."""
    # >>> TODO: collect completed ids, return pending tasks whose depends_on is a subset of them
    completed = {task_id for task_id, task in tasks.items() if task.status == "completed"}
    return [t for t in tasks.values() if t.status == "pending" and set(t.depends_on) <= completed]
    # <<<


def check_deadlock(tasks: dict[str, TaskNode]) -> None:
    """Raise DeadlockError if tasks are pending but the frontier is empty. Barrier use only."""
    # >>> TODO: raise DeadlockError naming the stuck ids when pending exists and ready_frontier is empty
    pending = [task.id for task in tasks.values() if task.status == "pending"]
    if pending and not ready_frontier(tasks):
        raise DeadlockError(f"Pending but unreachable: {sorted(pending)}")
    # <<<
