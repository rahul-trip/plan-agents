"""Stage 3, ex1: scoped workers that receive immutable state slices. Custom track.

A worker can't leak or corrupt what it never receives. `frozen=True` alone isn't enough:
it stops field reassignment, but a nested dict stays mutable.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, field_validator

from planagents.schemas import TaskNode


class StateSlice(BaseModel):
    """Everything a worker may see. Immutable all the way down."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str
    instruction: str
    upstream: Mapping[str, str]

    @field_validator("upstream", mode="after")
    @classmethod
    def _freeze_upstream(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        """Copy, then wrap in a read-only view, so neither side can change the other's data."""
        # >>> TODO: return a MappingProxyType over a copy of value
        return MappingProxyType(dict(value))
        # <<<


def make_slice(task: TaskNode, tasks: Mapping[str, TaskNode]) -> StateSlice:
    """Build a task's slice: its instruction plus its direct parents' results. Nothing else."""
    # >>> TODO: construct a StateSlice whose upstream maps each dep in task.depends_on to its result
    upstream = {dep: tasks[dep].result or "" for dep in task.depends_on}
    return StateSlice(task_id=task.id, instruction=task.instruction, upstream=upstream)
    # <<<


class ScopedWorker:
    """Base class for domain workers. Subclasses implement `handle`, which sees only the slice."""

    name = "base"

    async def handle(self, state: StateSlice) -> str:
        raise NotImplementedError

    async def run(self, task: TaskNode, tasks: Mapping[str, TaskNode]) -> TaskNode:
        """Slice the state, call handle, and return an updated COPY of the task.

        A failing handler gives status='failed' with the error as result. Never touch `tasks`.
        """
        # >>> TODO: make_slice, await handle, return task.model_copy with status and result; catch Exception as failed
        state = make_slice(task, tasks)
        try:
            return task.model_copy(update={"status": "completed", "result": await self.handle(state)})
        except Exception as exc:
            return task.model_copy(update={"status": "failed", "result": f"{type(exc).__name__}: {exc}"})
        # <<<
