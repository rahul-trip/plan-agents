"""Plan schemas shared across stages.

Two kinds of model, kept apart on purpose:
- What the LLM may write (PlannedStep, PlannedTask): the plan only. Fields are all required, with
  no defaults or free-form dicts, because provider structured-output modes restrict those.
- What the runtime tracks (Step, TaskNode): the plan plus status and result. If the LLM could write
  these, it could mark a task completed with an invented result, and the task would never run.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pending", "completed", "failed", "cancelled"]


class PlanValidationError(ValueError):
    """The plan can never run correctly: duplicate IDs, dangling dependencies, a cycle, and so on."""


class PlannedStep(BaseModel):
    """One item of a linear plan, as the LLM writes it (Stage 1)."""

    id: str = Field(description="Unique step identifier, e.g. 'step_1'")
    description: str
    assigned_worker: str


class Step(PlannedStep):
    """A planned step plus its runtime state."""

    status: Status = "pending"
    result: str | None = None


def _to_steps(planned: list[PlannedStep]) -> list[Step]:
    return [Step(id=p.id, description=p.description, assigned_worker=p.assigned_worker) for p in planned]


class Plan(BaseModel):
    steps: list[PlannedStep]

    def to_steps(self) -> list[Step]:
        """Runtime steps, always starting pending, whatever the reply contained."""
        return _to_steps(self.steps)


class ReplanDecision(BaseModel):
    """The replanner's answer: the full remaining queue. Empty means finish."""

    remaining_steps: list[PlannedStep] = Field(default_factory=list)

    def to_steps(self) -> list[Step]:
        return _to_steps(self.remaining_steps)


class PlannedTask(BaseModel):
    """One DAG node, as the LLM writes it (Stage 2 onward)."""

    id: str = Field(description="Unique task identifier, e.g. 'task_1'")
    worker: str
    instruction: str
    depends_on: list[str] = Field(description="IDs of prerequisite tasks. Empty if none.")
    args_json: str = Field(description="The task's arguments as a JSON object in a string. '{}' if none.")

    def parsed_args(self) -> dict[str, Any]:
        """Weaker models often write a Python-style dict or an empty string here. Their meaning is
        unambiguous, so accept them rather than fail the whole plan."""
        text = self.args_json.strip() or "{}"
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise PlanValidationError(f"{self.id}: args_json is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise PlanValidationError(f"{self.id}: args_json must be a JSON object")
        return value


class PlanDraft(BaseModel):
    """What the LLM returns when planning. See planagents.dag.make_plan."""

    tasks: list[PlannedTask]


class TaskNode(BaseModel):
    """One node of a DAG plan as the runtime tracks it. Graph state holds these."""

    id: str = Field(description="Unique task identifier, e.g. 'task_1'")
    worker: str
    instruction: str
    depends_on: list[str] = Field(default_factory=list, description="IDs of prerequisite tasks")
    status: Status = "pending"
    result: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    destructive: bool = False


class ExecutionPlan(BaseModel):
    tasks: list[TaskNode]

    def to_draft(self) -> PlanDraft:
        """The same plan as the LLM would write it. Runtime fields don't survive this."""
        return PlanDraft(
            tasks=[
                PlannedTask(
                    id=t.id,
                    worker=t.worker,
                    instruction=t.instruction,
                    depends_on=t.depends_on,
                    args_json=json.dumps(t.args),
                )
                for t in self.tasks
            ]
        )


class WorkerSpec(BaseModel):
    """A worker your code provides. The registry is how the planner learns what exists, and it's
    code, not the LLM, that decides what counts as destructive."""

    description: str
    destructive: bool = False
