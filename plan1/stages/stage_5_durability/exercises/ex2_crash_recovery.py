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
    raise NotImplementedError("the Stage 2 graph shape, compiled with checkpointer=checkpointer")


async def run_or_resume(url: str, thread_id: str, plan: ExecutionPlan, worker: Worker, resume: bool = False) -> dict:
    """Open AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()), await setup(), and
    invoke the graph with durability='sync'. Input is {} for a new run, None to resume a thread.

    durability='sync' waits for each checkpoint write before continuing. It is the slowest mode and
    the one that gives you the strongest guarantee if the process dies at any moment.
    """
    raise NotImplementedError("open the saver, setup(), build the graph, ainvoke with thread_id config, a recursion_limit and durability='sync'")
