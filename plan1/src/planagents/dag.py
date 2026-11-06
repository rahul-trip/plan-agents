"""Reference DAG kit used from Stage 2's ex3 onward. SPOILER for Stage 2: finish ex2 first."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Iterable, Mapping

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from planagents.llm import structured
from planagents.schemas import ExecutionPlan, PlanDraft, PlanValidationError, TaskNode, WorkerSpec

__all__ = [
    "DeadlockError",
    "PlanValidationError",
    "cancel_with_dependents",
    "check_deadlock",
    "checkpoint_serde",
    "describe_workers",
    "make_plan",
    "merge_tasks",
    "ready_frontier",
    "tasks_from_draft",
    "validate_plan",
]

Workers = Mapping[str, WorkerSpec]


class DeadlockError(RuntimeError):
    """Tasks are pending, nothing is ready, and nothing is left to unblock them."""


def validate_plan(plan: ExecutionPlan) -> dict[str, TaskNode]:
    """Return the tasks keyed by ID, or raise PlanValidationError."""
    ids = [task.id for task in plan.tasks]
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    if duplicates:
        raise PlanValidationError(f"Duplicate task IDs: {sorted(duplicates)}")
    # TopologicalSorter silently accepts unknown IDs as new nodes, so check them first.
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


def describe_workers(workers: Workers) -> str:
    """The list a planner prompt needs. An LLM can't choose a tool it was never told about."""
    return "\n".join(f"- {name}: {spec.description}" for name, spec in workers.items())


def tasks_from_draft(draft: PlanDraft, workers: Workers | None = None) -> ExecutionPlan:
    """Turn what the LLM wrote into runtime tasks: pending, with parsed args.

    With a registry, unknown workers are rejected, and `destructive` comes from the registry.
    """
    tasks = []
    for planned in draft.tasks:
        if workers is not None and planned.worker not in workers:
            raise PlanValidationError(f"{planned.id} uses unknown worker {planned.worker!r}. Choose from {sorted(workers)}.")
        tasks.append(
            TaskNode(
                id=planned.id,
                worker=planned.worker,
                instruction=planned.instruction,
                depends_on=planned.depends_on,
                args=planned.parsed_args(),
                destructive=bool(workers and workers[planned.worker].destructive),
            )
        )
    return ExecutionPlan(tasks=tasks)


async def make_plan(llm, request: str, workers: Workers | None = None, max_rejections: int = 2) -> ExecutionPlan:
    """Ask for a plan and validate it. A bad plan is re-asked with the reason, up to max_rejections
    times; after that the last PlanValidationError is raised."""
    prompt = request
    if workers is not None:
        prompt += f"\nUse only these workers:\n{describe_workers(workers)}"
    problem: PlanValidationError | None = None
    for _ in range(max_rejections + 1):
        draft = await structured(llm, PlanDraft).ainvoke(prompt)
        try:
            plan = tasks_from_draft(draft, workers)
            if not plan.tasks:
                raise PlanValidationError("The plan has no tasks.")
            validate_plan(plan)
            return plan
        except PlanValidationError as exc:
            problem = exc
            prompt += f"\nREJECTED: {exc}"
    raise problem


def merge_tasks(existing: dict[str, TaskNode] | None, updates: dict[str, TaskNode]) -> dict[str, TaskNode]:
    """Reducer: parallel workers merge deltas. Removal is status='cancelled', never a delete."""
    return {**(existing or {}), **updates}


def ready_frontier(tasks: dict[str, TaskNode]) -> list[TaskNode]:
    """Pending tasks whose dependencies have all completed."""
    completed = {task_id for task_id, task in tasks.items() if task.status == "completed"}
    return [t for t in tasks.values() if t.status == "pending" and set(t.depends_on) <= completed]


def check_deadlock(tasks: dict[str, TaskNode]) -> None:
    """Call only at a barrier, when no worker is running."""
    pending = [task.id for task in tasks.values() if task.status == "pending"]
    if pending and not ready_frontier(tasks):
        raise DeadlockError(f"Pending but unreachable: {sorted(pending)}")


def cancel_with_dependents(tasks: dict[str, TaskNode], task_ids: Iterable[str]) -> dict[str, TaskNode]:
    """Delta cancelling these tasks and everything downstream of them.

    A dependent can never run once its parent is gone, so leaving it pending would deadlock.
    """
    doomed = set(task_ids)
    changed = True
    while changed:
        changed = False
        for task in tasks.values():
            if task.id not in doomed and doomed & set(task.depends_on):
                doomed.add(task.id)
                changed = True
    return {
        task_id: tasks[task_id].model_copy(update={"status": "cancelled"})
        for task_id in doomed
        if task_id in tasks
    }


def checkpoint_serde(*extra: tuple[str, str]) -> JsonPlusSerializer:
    """Serializer that may rebuild our Pydantic models from a checkpoint.

    Without an allowlist LangGraph warns on every load that unregistered types will be blocked
    in a future version. Allowlisting is a security control: only name types you own.
    """
    allowed = [("planagents.schemas", "TaskNode"), *extra]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)
