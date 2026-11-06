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
    # >>> TODO: return a WorkerInputState with inputs mapping each dependency to artifacts[dep].summary
    return WorkerInputState(
        task_id=task.id,
        instruction=task.instruction,
        inputs={dep: artifacts[dep].summary for dep in task.depends_on},
    )
    # <<<


def build_graph(plan: ExecutionPlan, worker: Worker):
    """load_plan -> scheduler -> fan out to worker_node(WorkerInputState) -> scheduler -> END.

    worker_node returns two deltas: the task marked completed (with result=the summary) and
    artifacts[task_id]. If the worker raises, the task is marked failed instead, with the error as
    its result, so one failure can't abort the run. The router goes to END on any failed task or when
    nothing is pending; otherwise it runs check_deadlock and sends resolve_inputs(...) per frontier task.
    """
    # >>> TODO: implement load_plan, scheduler, the router and worker_node using resolve_inputs; compile
    tasks = validate_plan(plan)

    async def load_plan(state: BlackboardState) -> dict:
        return {"tasks": tasks, "artifacts": {}}

    async def scheduler(state: BlackboardState) -> dict:
        return {}

    def route(state: BlackboardState) -> list[Send] | Literal["__end__"]:
        current = state["tasks"]
        if any(t.status == "failed" for t in current.values()):
            return END
        if not any(t.status == "pending" for t in current.values()):
            return END
        check_deadlock(current)
        return [
            Send("worker_node", resolve_inputs(task, state["artifacts"]))
            for task in ready_frontier(current)
        ]

    async def worker_node(worker_input: WorkerInputState) -> dict:
        task_id = worker_input["task_id"]
        try:
            artifact = await worker(worker_input)
        except Exception as exc:
            failed = tasks[task_id].model_copy(update={"status": "failed", "result": f"{type(exc).__name__}: {exc}"})
            return {"tasks": {task_id: failed}}
        done = tasks[task_id].model_copy(update={"status": "completed", "result": artifact.summary})
        return {"tasks": {task_id: done}, "artifacts": {task_id: artifact}}

    builder = StateGraph(BlackboardState)
    builder.add_node("load_plan", load_plan)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker_node", worker_node)
    builder.add_edge(START, "load_plan")
    builder.add_edge("load_plan", "scheduler")
    builder.add_conditional_edges("scheduler", route, ["worker_node", END])
    builder.add_edge("worker_node", "scheduler")
    return builder.compile()
    # <<<
