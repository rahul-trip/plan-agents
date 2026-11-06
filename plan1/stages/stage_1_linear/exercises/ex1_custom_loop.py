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
    raise NotImplementedError("call structured(llm, Plan) with a prompt containing the objective; return a deque of plan.to_steps()")


async def replan(llm, state: GlobalState) -> deque[Step]:
    """Let the LLM rewrite the remaining queue from the results so far.

    The LLM may repeat a step that already finished. Drop those, so finished work never runs twice.
    """
    raise NotImplementedError("call structured(llm, ReplanDecision) with objective, render_results(state) and the remaining steps; return the new queue minus completed ids")


async def run_plan_loop(
    objective: str, llm, workers: dict[str, Worker], max_steps: int = 20
) -> GlobalState:
    """Run until the queue is empty, replanning after every step.

    Each step gets a fresh Scratchpad, which is dropped when the step ends. Replanning can grow
    the queue forever, so raise StepLimitExceeded rather than run more than max_steps steps.
    """
    raise NotImplementedError("while plan_queue: popleft, run the worker with a new Scratchpad, record the result, replan; enforce the step cap")
