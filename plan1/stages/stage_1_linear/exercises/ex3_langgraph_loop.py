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
    raise NotImplementedError("write the three nodes, add edges START->planner->executor->replanner, a conditional edge from replanner, and compile")
