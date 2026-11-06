"""Stage 2, ex1: the DAG engine in about 60 lines of plain Python. Do this before any LangGraph.

Once get_ready() / done() are in your hands, LangGraph's superstep barrier stops being magic.
"""

from __future__ import annotations

import asyncio
from graphlib import TopologicalSorter
from typing import Awaitable, Callable

from planagents.schemas import ExecutionPlan, TaskNode

# A worker gets its task and only its direct parents' results, and returns the task's result.
Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


async def run_plan(plan: ExecutionPlan, worker: Worker) -> dict[str, TaskNode]:
    """Run the plan level by level and return the final tasks keyed by id.

    Loop while sorter.is_active(): run every id from get_ready() with asyncio.gather, then call
    done() for the ones that succeeded.

    A raising worker marks its task 'failed', with the error text as its result. Siblings still
    finish, and dependents stay 'pending'. Beware: a task never marked done keeps the sorter
    active forever, while get_ready() returns nothing, so you must detect that and stop.
    A cyclic plan should let graphlib.CycleError propagate.
    """
    # >>> TODO: build TopologicalSorter from depends_on, prepare(), loop with gather(return_exceptions=True), done() only the successes
    tasks = {task.id: task for task in plan.tasks}
    sorter = TopologicalSorter({task.id: set(task.depends_on) for task in plan.tasks})
    sorter.prepare()

    while sorter.is_active():
        ready = sorter.get_ready()
        if not ready:
            break  # Only failed tasks' dependents remain, and they can never become ready.
        calls = []
        for task_id in ready:
            task = tasks[task_id]
            upstream = {dep: tasks[dep].result for dep in task.depends_on}
            calls.append(worker(task, upstream))
        outcomes = await asyncio.gather(*calls, return_exceptions=True)

        succeeded = []
        for task_id, outcome in zip(ready, outcomes):
            # gather captures BaseException too. Cancellation isn't a task failure.
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, Exception):
                tasks[task_id] = tasks[task_id].model_copy(
                    update={"status": "failed", "result": str(outcome)}
                )
            else:
                tasks[task_id] = tasks[task_id].model_copy(
                    update={"status": "completed", "result": outcome}
                )
                succeeded.append(task_id)
        sorter.done(*succeeded)
    return tasks
    # <<<
