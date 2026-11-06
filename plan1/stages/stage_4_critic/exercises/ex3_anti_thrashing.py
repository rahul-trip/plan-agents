"""Stage 4, ex3: anti-thrashing guards.

Thrashing is a replanner regenerating a path that already failed. There are three layers, because
none is enough alone:
- Negative prompts tell the LLM what failed. It may ignore them.
- A deterministic guard rejects any plan that repeats a failed path. It can't be ignored.
- A hard max_replans ends the run whatever happens.
"""

from __future__ import annotations

from pydantic import BaseModel

from planagents.dag import tasks_from_draft
from planagents.llm import structured
from planagents.schemas import ExecutionPlan, PlanDraft, TaskNode


class FailedAttempt(BaseModel):
    """One broken path. `tool` is the worker or tool that failed; reason is the evidence."""

    task_id: str
    tool: str
    reason: str


class ReplanBudgetExhausted(Exception):
    pass


def render_negative_constraints(failed_plans: list[FailedAttempt]) -> str:
    """Turn the log into explicit DO NOT instructions, one line per attempt, each naming the tool
    and the reason. Empty log gives an empty string, so a first plan sees no noise."""
    raise NotImplementedError("return '' for an empty log; otherwise a header and a line per FailedAttempt naming tool and reason")


def repeated_failures(tasks: list[TaskNode], failed_plans: list[FailedAttempt]) -> list[str]:
    """IDs of pending tasks that would use a failed tool. Finished tasks don't count, because
    they aren't going to run again."""
    raise NotImplementedError("collect failed tools; return ids of pending tasks whose worker is one of them")


async def guarded_replan(
    llm,
    objective: str,
    failed_plans: list[FailedAttempt],
    replan_count: int,
    max_replans: int = 3,
    max_guard_rejections: int = 2,
) -> ExecutionPlan:
    """Ask structured(llm, PlanDraft) for a new plan, with negative constraints in the prompt, and
    convert the reply with tasks_from_draft from planagents.dag.

    - Raise ReplanBudgetExhausted at once if replan_count >= max_replans, without any LLM call.
    - If a plan repeats a failure, ask again with the rejection added to the prompt, up to
      max_guard_rejections rejections in total. Past that, raise ReplanBudgetExhausted.
    """
    raise NotImplementedError("check the budget, build the prompt with render_negative_constraints, loop on repeated_failures")
