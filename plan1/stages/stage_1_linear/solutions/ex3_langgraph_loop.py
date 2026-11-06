"""Stage 1, ex3: the same loop as a StateGraph: planner -> executor -> replanner -> ...

The graph owns the loop now. State updates are deltas, merged by the reducer you wrote in ex2.
"""

from __future__ import annotations

from typing import Annotated, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from planagents.llm import structured
from planagents.schemas import Plan, ReplanDecision, Step


def merge_steps(existing: list[Step] | None, updates: list[Step]) -> list[Step]:
    merged = {step.id: step for step in existing or []}
    for step in updates:
        merged[step.id] = step
    return list(merged.values())


class PlanState(TypedDict, total=False):
    objective: str
    steps: Annotated[list[Step], merge_steps]
    final_output: str


Worker = Callable[[Step], Awaitable[str]]


def next_pending(steps: list[Step]) -> Step | None:
    return next((step for step in steps if step.status == "pending"), None)


def build_graph(llm, workers: dict[str, Worker], recursion_limit: int = 50):
    """Compile planner, executor and replanner.

    - planner: structured(llm, Plan) writes the initial steps (plan.to_steps()).
    - executor: runs the first pending step and returns only that step, as a delta.
    - replanner: structured(llm, ReplanDecision) returns the new remaining queue (to_steps()).
      Pending steps it leaves out must become status='cancelled', because the reducer can't
      delete. Ignore any it returns with a completed id, or the reducer would wipe finished work.
    - After the replanner, loop to the executor while a step is pending, otherwise END.
    - Return the compiled graph with .with_config(recursion_limit=recursion_limit), so a replanner
      that keeps adding steps is stopped.
    """
    # >>> TODO: write the three nodes, add edges START->planner->executor->replanner, a conditional edge from replanner, and compile
    structured_plan = structured(llm, Plan)
    structured_replan = structured(llm, ReplanDecision)

    async def planner(state: PlanState) -> dict:
        plan = await structured_plan.ainvoke(f"Plan sequential steps for: {state['objective']}")
        return {"steps": plan.to_steps()}

    async def executor(state: PlanState) -> dict:
        step = next_pending(state["steps"])
        result = await workers[step.assigned_worker](step)
        return {"steps": [step.model_copy(update={"status": "completed", "result": result})]}

    async def replanner(state: PlanState) -> dict:
        steps = state["steps"]
        done = "\n".join(f"- {s.id}: {s.result}" for s in steps if s.status == "completed")
        pending = [s for s in steps if s.status == "pending"]
        remaining = "\n".join(f"- {s.id}: {s.description}" for s in pending)
        decision = await structured_replan.ainvoke(
            f"Objective: {state['objective']}\nResults:\n{done}\n"
            f"Remaining:\n{remaining or '(none)'}\n"
            "Return the full remaining queue. Return an empty list to finish."
        )
        finished = {s.id for s in steps if s.status == "completed"}
        fresh = [step for step in decision.to_steps() if step.id not in finished]
        kept = {step.id for step in fresh}
        dropped = [
            step.model_copy(update={"status": "cancelled"})
            for step in pending
            if step.id not in kept
        ]
        return {"steps": [*fresh, *dropped]}

    def route(state: PlanState) -> str:
        return "executor" if next_pending(state["steps"]) else END

    builder = StateGraph(PlanState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("replanner", replanner)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges("replanner", route, ["executor", END])
    return builder.compile().with_config(recursion_limit=recursion_limit)
    # <<<
