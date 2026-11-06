"""Stage 5, ex4: time travel. Go back to before a failed worker, patch state, resume on a fork.

Two moves that look alike:
- Replay: ainvoke(None, snapshot.config) re-runs everything after that checkpoint.
- Fork: aupdate_state(snapshot.config, ...) writes a NEW checkpoint, then you resume from it.

The original history stays, but the thread's latest state becomes the fork. Keep the original
snapshot's config if you need to look at the failed run again.
"""

from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from planagents.dag import check_deadlock, merge_tasks, ready_frontier, validate_plan
from planagents.schemas import ExecutionPlan, TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


class TravelState(TypedDict, total=False):
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    final_output: str


def build_graph(plan: ExecutionPlan, worker: Worker, checkpointer):
    """Provided: the DAG graph from ex2, so this exercise can focus on the time travel itself."""
    tasks = validate_plan(plan)

    async def load_plan(state: TravelState) -> dict:
        return {"tasks": tasks}

    async def scheduler(state: TravelState) -> dict:
        return {}

    def route(state: TravelState) -> list[Send] | Literal["synthesize"]:
        current = state["tasks"]
        if any(t.status == "failed" for t in current.values()):
            return "synthesize"
        if not any(t.status == "pending" for t in current.values()):
            return "synthesize"
        check_deadlock(current)
        return [
            Send("worker", {"task": t, "upstream": {d: current[d].result or "" for d in t.depends_on}})
            for t in ready_frontier(current)
        ]

    async def worker_node(payload: dict) -> dict:
        task = payload["task"]
        try:
            update = {"status": "completed", "result": await worker(task, payload["upstream"])}
        except Exception as exc:
            update = {"status": "failed", "result": f"{type(exc).__name__}: {exc}"}
        return {"tasks": {task.id: task.model_copy(update=update)}}

    async def synthesize(state: TravelState) -> dict:
        return {"final_output": "\n".join(f"{t.id} [{t.status}]" for t in state["tasks"].values())}

    builder = StateGraph(TravelState)
    builder.add_node("load_plan", load_plan)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker", worker_node)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "load_plan")
    builder.add_edge("load_plan", "scheduler")
    builder.add_conditional_edges("scheduler", route, ["worker", "synthesize"])
    builder.add_edge("worker", "scheduler")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)


async def find_before_failure(graph, config: dict, task_id: str):
    """Return the newest checkpoint in which task_id is still pending: the moment before it ran.

    aget_state_history yields newest first. Skip snapshots with no tasks yet.
    """
    # >>> TODO: iterate aget_state_history and return the first snapshot where this task's status is 'pending'
    async for snapshot in graph.aget_state_history(config):
        task = snapshot.values.get("tasks", {}).get(task_id)
        if task is not None and task.status == "pending":
            return snapshot
    raise LookupError(f"No checkpoint with {task_id} pending")
    # <<<


async def patch_and_fork(graph, snapshot, task_id: str, args: dict[str, Any]) -> dict:
    """Write a fork with this task's args replaced, and return its config.

    Pass as_node='worker' so the fork looks like a worker just finished, which makes the scheduler
    the next node to run, and it re-dispatches the frontier with the patched args.
    """
    # >>> TODO: model_copy the task with new args, then return await graph.aupdate_state(snapshot.config, {...}, as_node='worker')
    patched = snapshot.values["tasks"][task_id].model_copy(update={"args": args})
    return await graph.aupdate_state(snapshot.config, {"tasks": {task_id: patched}}, as_node="worker")
    # <<<


async def fix_and_resume(graph, config: dict, task_id: str, args: dict[str, Any]) -> dict:
    """The whole move: find the checkpoint, fork it with the fix, and run the fork to the end."""
    # >>> TODO: combine find_before_failure and patch_and_fork, then ainvoke(None, fork_config)
    snapshot = await find_before_failure(graph, config, task_id)
    fork_config = await patch_and_fork(graph, snapshot, task_id, args)
    return await graph.ainvoke(None, fork_config)
    # <<<
