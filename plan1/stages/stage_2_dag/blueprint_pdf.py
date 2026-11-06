"""The source PDF's starter blueprint, with the same logic and mock planner.

The only structural change: the mock plan and the worker can be passed in, so the tests can swap
them. It runs on the happy path, and test_blueprint_gotchas.py pins down where it breaks. Do not
edit it: your completed version is ex3_blueprint_graph.py.
"""

import operator  # noqa: F401  (unused in the PDF too)
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field


class TaskNode(BaseModel):
    id: str = Field(description="Unique task identifier, e.g., 'task_1'")
    worker: Literal["researcher", "coder", "analyst"]
    instruction: str
    depends_on: list[str] = Field(default_factory=list, description="IDs of prerequisite tasks")
    status: Literal["pending", "completed", "failed"] = "pending"
    result: str | None = None


class ExecutionPlan(BaseModel):
    tasks: list[TaskNode]


def merge_tasks(existing: dict[str, TaskNode], updates: dict[str, TaskNode]) -> dict[str, TaskNode]:
    """Custom reducer: merges parallel worker task updates into the central DAG dictionary."""
    merged = existing.copy() if existing else {}
    merged.update(updates)
    return merged


class OrchestratorState(TypedDict):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    replan_count: int
    final_output: str | None


class WorkerPayload(TypedDict):
    """Scoped state passed to an isolated worker via Send()."""

    task: TaskNode
    upstream_context: dict[str, str]


# Not in the PDF: lets the tests swap the mock plan. The PDF hard-codes MOCK_PLAN inline.
MOCK_PLAN = ExecutionPlan(
    tasks=[
        TaskNode(id="task_1", worker="researcher", instruction="Fetch API specs", depends_on=[]),
        TaskNode(id="task_2", worker="analyst", instruction="Query DB schema", depends_on=[]),
        TaskNode(
            id="task_3",
            worker="coder",
            instruction="Write integration script",
            depends_on=["task_1", "task_2"],
        ),
    ]
)


def make_planner(plan: ExecutionPlan):
    async def planner_node(state: OrchestratorState) -> Command[Literal["scheduler"]]:
        """Generates a validated DAG from the user objective."""
        dag = {t.id: set(t.depends_on) for t in plan.tasks}
        try:
            TopologicalSorter(dag).prepare()
        except CycleError as e:
            raise ValueError(f"Planner produced a cyclic dependency graph: {e}")
        task_map = {t.id: t for t in plan.tasks}
        return Command(update={"tasks": task_map}, goto="scheduler")

    return planner_node


async def scheduler_node(state: OrchestratorState) -> dict:
    """Synchronization barrier between parallel supersteps."""
    return {}


def route_ready_tasks(state: OrchestratorState) -> list[Send] | Literal["replanner", "synthesizer"]:
    """Evaluates DAG dependencies and fans out ready tasks in parallel via Send."""
    tasks = state["tasks"]

    if any(t.status == "failed" for t in tasks.values()):
        if state.get("replan_count", 0) < 3:
            return "replanner"
        return "synthesizer"

    if all(t.status == "completed" for t in tasks.values()):
        return "synthesizer"

    completed_ids = {t_id for t_id, t in tasks.items() if t.status == "completed"}
    ready_tasks = [
        t
        for t in tasks.values()
        if t.status == "pending" and set(t.depends_on).issubset(completed_ids)
    ]

    dispatches = []
    for task in ready_tasks:
        scoped_context = {dep_id: tasks[dep_id].result for dep_id in task.depends_on}
        dispatches.append(Send("worker_executor", {"task": task, "upstream_context": scoped_context}))
    return dispatches


async def worker_executor(payload: WorkerPayload) -> dict:
    """Executes a single bounded task in isolation and commits only its delta back to global state."""
    task = payload["task"]
    context = payload["upstream_context"]
    updated_task = task.model_copy(
        update={
            "status": "completed",
            "result": f"[{task.worker.upper()} OUTPUT for {task.id} | Context keys: {list(context.keys())}]",
        }
    )
    return {"tasks": {task.id: updated_task}}


async def replanner_node(state: OrchestratorState) -> Command[Literal["scheduler"]]:
    """Mutates the remaining DAG when a worker node fails."""
    return Command(update={"replan_count": state.get("replan_count", 0) + 1}, goto="scheduler")


async def synthesizer_node(state: OrchestratorState) -> dict:
    """Aggregates terminal DAG nodes into the final response."""
    results = {t_id: t.result for t_id, t in state["tasks"].items()}
    return {"final_output": f"Completed {len(results)} tasks: {results}"}


def build_graph(plan: ExecutionPlan = MOCK_PLAN, worker=worker_executor):
    builder = StateGraph(OrchestratorState)
    builder.add_node("planner", make_planner(plan))
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("worker_executor", worker)
    builder.add_node("replanner", replanner_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "scheduler", route_ready_tasks, ["worker_executor", "replanner", "synthesizer"]
    )
    builder.add_edge("worker_executor", "scheduler")
    builder.add_edge("synthesizer", END)
    return builder.compile()
