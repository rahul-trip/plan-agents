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
    # >>> TODO: return '' for an empty log; otherwise a header and a line per FailedAttempt naming tool and reason
    if not failed_plans:
        return ""
    lines = [f"- DO NOT use {attempt.tool} (task {attempt.task_id}): {attempt.reason}" for attempt in failed_plans]
    return "These paths already failed. Never plan them again:\n" + "\n".join(lines)
    # <<<


def repeated_failures(tasks: list[TaskNode], failed_plans: list[FailedAttempt]) -> list[str]:
    """IDs of pending tasks that would use a failed tool. Finished tasks don't count, because
    they aren't going to run again."""
    # >>> TODO: collect failed tools; return ids of pending tasks whose worker is one of them
    blocked = {attempt.tool for attempt in failed_plans}
    return [task.id for task in tasks if task.status == "pending" and task.worker in blocked]
    # <<<


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
    # >>> TODO: check the budget, build the prompt with render_negative_constraints, loop on repeated_failures
    if replan_count >= max_replans:
        raise ReplanBudgetExhausted(f"Used {replan_count} of {max_replans} replans")
    prompt = f"Objective: {objective}\n{render_negative_constraints(failed_plans)}\nReturn a new plan."
    for _ in range(max_guard_rejections + 1):
        plan = tasks_from_draft(await structured(llm, PlanDraft).ainvoke(prompt))
        repeated = repeated_failures(plan.tasks, failed_plans)
        if not repeated:
            return plan
        prompt += f"\nREJECTED: tasks {repeated} reuse a failed tool. Choose a different tool."
    raise ReplanBudgetExhausted(f"Plan kept repeating failed paths after {max_guard_rejections} rejections")
    # <<<
