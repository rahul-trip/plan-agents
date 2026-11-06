"""Stage 3, ex2: the blackboard pattern in LangGraph.

Workers never read graph state. They post an Artifact to the blackboard, with a short `summary`
for other workers and a `raw` body (tool output, stack traces) for audit only. The dispatcher
resolves each task's references and injects just the summaries into a minimal WorkerInputState.
"""

from __future__ import annotations

from typing import Annotated, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from planagents.dag import check_deadlock, merge_tasks, ready_frontier, validate_plan
from planagents.schemas import ExecutionPlan, TaskNode


class Artifact(BaseModel):
    summary: str
    raw: str = ""


class WorkerInputState(TypedDict):
    """The whole of what a worker receives. Deliberately has no 'tasks' or 'artifacts' key."""

    task_id: str
    instruction: str
    inputs: dict[str, str]


def merge_artifacts(existing: dict[str, Artifact] | None, updates: dict[str, Artifact]) -> dict[str, Artifact]:
    return {**(existing or {}), **updates}


class BlackboardState(TypedDict, total=False):
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    artifacts: Annotated[dict[str, Artifact], merge_artifacts]


Worker = Callable[[WorkerInputState], Awaitable[Artifact]]


def resolve_inputs(task: TaskNode, artifacts: dict[str, Artifact]) -> WorkerInputState:
    """Build the minimal worker input: for each dep in depends_on, only that artifact's summary."""
    raise NotImplementedError("return a WorkerInputState with inputs mapping each dependency to artifacts[dep].summary")


def build_graph(plan: ExecutionPlan, worker: Worker):
    """load_plan -> scheduler -> fan out to worker_node(WorkerInputState) -> scheduler -> END.

    worker_node returns two deltas: the task marked completed (with result=the summary) and
    artifacts[task_id]. If the worker raises, the task is marked failed instead, with the error as
    its result, so one failure can't abort the run. The router goes to END on any failed task or when
    nothing is pending; otherwise it runs check_deadlock and sends resolve_inputs(...) per frontier task.
    """
    raise NotImplementedError("implement load_plan, scheduler, the router and worker_node using resolve_inputs; compile")
