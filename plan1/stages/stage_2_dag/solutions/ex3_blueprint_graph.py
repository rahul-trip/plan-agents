"""Stage 2, ex3: complete the PDF blueprint: a real planner, a custom dispatcher edge, and
Send fan-out converging on a scheduler barrier.

Differences from blueprint_pdf.py (each fixes something test_blueprint_gotchas.py pinned down):
- The planner runs the full validator, not only a cycle check.
- The router raises DeadlockError instead of silently ending the run.
- The worker catches exceptions, so one failure can't abort its siblings.
- There's no replanner yet. A failure goes to the synthesizer (graceful degradation), and
  Stage 4 replaces this with a critic and a real replanner.
"""

from __future__ import annotations

from typing import Annotated, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from planagents.dag import check_deadlock, make_plan, merge_tasks, ready_frontier, validate_plan
from planagents.schemas import TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


class OrchestratorState(TypedDict, total=False):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    final_output: str


class WorkerPayload(TypedDict):
    """Scoped state a worker receives through Send: its task and its direct parents' results."""

    task: TaskNode
    upstream_context: dict[str, str]


def route_ready_tasks(state: OrchestratorState) -> list[Send] | Literal["synthesizer"]:
    """The custom dispatcher edge.

    - Any failed task, or nothing left pending: 'synthesizer'.
    - Otherwise check_deadlock(), then one Send('worker_executor', WorkerPayload) per frontier
      task, with upstream_context holding ONLY that task's depends_on results.
    """
    # >>> TODO: implement the routing rules above using ready_frontier and check_deadlock
    tasks = state["tasks"]
    if any(t.status == "failed" for t in tasks.values()):
        return "synthesizer"
    if not any(t.status == "pending" for t in tasks.values()):
        return "synthesizer"
    check_deadlock(tasks)
    return [
        Send(
            "worker_executor",
            {
                "task": task,
                "upstream_context": {dep: tasks[dep].result for dep in task.depends_on},
            },
        )
        for task in ready_frontier(tasks)
    ]
    # <<<


def build_graph(llm, worker: Worker):
    """Compile planner -> scheduler -> (fan out worker_executor) -> scheduler -> ... -> synthesizer.

    - planner: await make_plan(llm, prompt) from planagents.dag, then
      Command(update={'tasks': validate_plan(plan)}, goto='scheduler'). make_plan asks for a
      PlanDraft, so the LLM can only write the plan itself, never a task's status or result. It
      re-asks on an invalid plan, then raises PlanValidationError.
    - scheduler: returns {}. It is purely the barrier that fan-in converges on.
    - worker_executor(payload): awaits worker(task, upstream_context) and returns only
      {'tasks': {id: updated_task}}. Exceptions become status='failed' with the error as result.
    - synthesizer: final_output naming each task's status and result.
    """
    # >>> TODO: write the four nodes and wire the edges; annotate planner's return as Command[Literal['scheduler']]
    async def planner(state: OrchestratorState) -> Command[Literal["scheduler"]]:
        plan = await make_plan(
            llm, f"Plan a task DAG for this objective. Use depends_on for ordering.\n{state['objective']}"
        )
        return Command(update={"tasks": validate_plan(plan)}, goto="scheduler")

    async def scheduler(state: OrchestratorState) -> dict:
        return {}

    async def worker_executor(payload: WorkerPayload) -> dict:
        task = payload["task"]
        try:
            result = await worker(task, payload["upstream_context"])
            update = {"status": "completed", "result": result}
        except Exception as exc:
            update = {"status": "failed", "result": f"{type(exc).__name__}: {exc}"}
        return {"tasks": {task.id: task.model_copy(update=update)}}

    async def synthesizer(state: OrchestratorState) -> dict:
        lines = [f"{t.id} [{t.status}]: {t.result}" for t in state["tasks"].values()]
        return {"final_output": "\n".join(lines)}

    builder = StateGraph(OrchestratorState)
    builder.add_node("planner", planner)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker_executor", worker_executor)
    builder.add_node("synthesizer", synthesizer)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("scheduler", route_ready_tasks, ["worker_executor", "synthesizer"])
    builder.add_edge("worker_executor", "scheduler")
    builder.add_edge("synthesizer", END)
    return builder.compile()
    # <<<
