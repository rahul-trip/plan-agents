"""Stage 5, ex2: crash recovery with AsyncPostgresSaver.

A 15-step DAG can't restart from scratch because step 12 died. The checkpointer persists every
superstep, and it also saves each finished task's writes within a superstep. The test kills a real
process with os._exit mid-DAG, then resumes it from a brand-new one.
"""

from __future__ import annotations

from typing import Annotated, Awaitable, Callable, Literal, TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from planagents.dag import check_deadlock, checkpoint_serde, merge_tasks, ready_frontier, validate_plan
from planagents.schemas import ExecutionPlan, TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


class DurableState(TypedDict, total=False):
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    final_output: str


def build_graph(plan: ExecutionPlan, worker: Worker, checkpointer):
    """load_plan -> scheduler -> worker fan-out -> scheduler -> ... -> synthesize, compiled with
    the checkpointer. Worker exceptions become status='failed'; a failed task goes to synthesize."""
    # >>> TODO: the Stage 2 graph shape, compiled with checkpointer=checkpointer
    tasks = validate_plan(plan)

    async def load_plan(state: DurableState) -> dict:
        return {"tasks": tasks}

    async def scheduler(state: DurableState) -> dict:
        return {}

    def route(state: DurableState) -> list[Send] | Literal["synthesize"]:
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

    async def synthesize(state: DurableState) -> dict:
        return {"final_output": "\n".join(f"{t.id}: {t.result}" for t in state["tasks"].values())}

    builder = StateGraph(DurableState)
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
    # <<<


async def run_or_resume(url: str, thread_id: str, plan: ExecutionPlan, worker: Worker, resume: bool = False) -> dict:
    """Open AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()), await setup(), and
    invoke the graph with durability='sync'. Input is {} for a new run, None to resume a thread.

    durability='sync' waits for each checkpoint write before continuing. It is the slowest mode and
    the one that gives you the strongest guarantee if the process dies at any moment.
    """
    # >>> TODO: open the saver, setup(), build the graph, ainvoke with thread_id config, a recursion_limit and durability='sync'
    async with AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()) as saver:
        await saver.setup()
        graph = build_graph(plan, worker, saver)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        return await graph.ainvoke(None if resume else {}, config, durability="sync")
    # <<<
