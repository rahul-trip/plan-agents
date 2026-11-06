"""Stage 1, ex1: the plan-execute-replan loop in plain Python.

Plan -> Execute Next Step -> Update State -> Replan or Finish, with no framework.

Keep two kinds of state apart:
- GlobalState survives the whole run: objective, remaining queue, step results.
- Scratchpad holds one step's tool-calling history and is thrown away with that step.
"""

from __future__ import annotations

from collections import deque
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from planagents.llm import structured
from planagents.schemas import Plan, ReplanDecision, Step


class StepLimitExceeded(Exception):
    """The replanner kept growing the queue past max_steps."""


class Scratchpad(BaseModel):
    """Temporary tool history of the active step. Workers may append to it freely."""

    step_id: str
    tool_history: list[str] = Field(default_factory=list)


class GlobalState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    objective: str
    plan_queue: deque[Step] = Field(default_factory=deque)
    completed: list[Step] = Field(default_factory=list)


Worker = Callable[[Step, Scratchpad], Awaitable[str]]


def render_results(state: GlobalState) -> str:
    """What the replanner may see: step results only, never a scratchpad."""
    return "\n".join(f"- {step.id}: {step.result}" for step in state.completed) or "(none)"


async def make_plan(llm, objective: str) -> deque[Step]:
    """Ask the LLM for a Plan and return its steps as a queue. Plan.to_steps() gives runtime steps."""
    # >>> TODO: call structured(llm, Plan) with a prompt containing the objective; return a deque of plan.to_steps()
    plan = await structured(llm, Plan).ainvoke(
        f"Break this objective into sequential steps.\nObjective: {objective}"
    )
    return deque(plan.to_steps())
    # <<<


async def replan(llm, state: GlobalState) -> deque[Step]:
    """Let the LLM rewrite the remaining queue from the results so far.

    The LLM may repeat a step that already finished. Drop those, so finished work never runs twice.
    """
    # >>> TODO: call structured(llm, ReplanDecision) with objective, render_results(state) and the remaining steps; return the new queue minus completed ids
    remaining = "\n".join(f"- {step.id}: {step.description}" for step in state.plan_queue)
    decision = await structured(llm, ReplanDecision).ainvoke(
        f"Objective: {state.objective}\n"
        f"Completed results:\n{render_results(state)}\n"
        f"Remaining steps:\n{remaining or '(none)'}\n"
        "Return the full remaining queue. Return an empty list to finish."
    )
    done = {step.id for step in state.completed}
    return deque(step for step in decision.to_steps() if step.id not in done)
    # <<<


async def run_plan_loop(
    objective: str, llm, workers: dict[str, Worker], max_steps: int = 20
) -> GlobalState:
    """Run until the queue is empty, replanning after every step.

    Each step gets a fresh Scratchpad, which is dropped when the step ends. Replanning can grow
    the queue forever, so raise StepLimitExceeded rather than run more than max_steps steps.
    """
    # >>> TODO: while plan_queue: popleft, run the worker with a new Scratchpad, record the result, replan; enforce the step cap
    state = GlobalState(objective=objective, plan_queue=await make_plan(llm, objective))
    executed = 0
    while state.plan_queue:
        if executed >= max_steps:
            raise StepLimitExceeded(f"Exceeded max_steps={max_steps}")
        step = state.plan_queue.popleft()
        result = await workers[step.assigned_worker](step, Scratchpad(step_id=step.id))
        state.completed.append(step.model_copy(update={"status": "completed", "result": result}))
        executed += 1
        state.plan_queue = await replan(llm, state)
    return state
    # <<<
