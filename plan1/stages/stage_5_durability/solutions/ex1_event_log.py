"""Stage 5, ex1: durability without a framework: an append-only event log plus state hydration.

State is never stored, only derived: replay the events and you get it back. This is what a
checkpointer gives you from the other direction, and it's easier to audit.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from planagents.dag import check_deadlock, ready_frontier, validate_plan
from planagents.schemas import ExecutionPlan, TaskNode

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS run_events (
    run_id  TEXT   NOT NULL,
    seq     BIGINT NOT NULL,
    type    TEXT   NOT NULL,
    payload JSONB  NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE OR REPLACE FUNCTION run_events_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'run_events is append-only';
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER run_events_no_rewrite
    BEFORE UPDATE OR DELETE ON run_events
    FOR EACH ROW EXECUTE FUNCTION run_events_append_only();
"""


class Event(BaseModel):
    seq: int
    type: str
    payload: dict[str, Any]


class RunState(BaseModel):
    tasks: dict[str, TaskNode] = {}
    last_seq: int = -1


class ConcurrentAppend(Exception):
    """Another writer already took this sequence number."""


def apply_event(state: RunState, event: Event) -> RunState:
    """Pure fold step. Returns a NEW state. Handles these event types:

    plan_created   payload {'tasks': [TaskNode dumps]}: sets the tasks
    task_completed payload {'task_id', 'result'}: marks completed
    task_failed    payload {'task_id', 'error'}: marks failed, with the error as result

    Always sets last_seq. An unknown type raises ValueError: silently skipping one would
    hydrate a wrong state.
    """
    # >>> TODO: copy the tasks dict, apply the event with model_copy, return a new RunState with last_seq=event.seq
    tasks = dict(state.tasks)
    if event.type == "plan_created":
        tasks = {data["id"]: TaskNode.model_validate(data) for data in event.payload["tasks"]}
    elif event.type == "task_completed":
        task = tasks[event.payload["task_id"]]
        tasks[task.id] = task.model_copy(update={"status": "completed", "result": event.payload["result"]})
    elif event.type == "task_failed":
        task = tasks[event.payload["task_id"]]
        tasks[task.id] = task.model_copy(update={"status": "failed", "result": event.payload["error"]})
    else:
        raise ValueError(f"Unknown event type: {event.type}")
    return RunState(tasks=tasks, last_seq=event.seq)
    # <<<


def hydrate(events: list[Event]) -> RunState:
    """Rebuild state by folding apply_event over the events in order."""
    # >>> TODO: fold apply_event starting from RunState()
    state = RunState()
    for event in events:
        state = apply_event(state, event)
    return state
    # <<<


class EventLog:
    """Postgres-backed log. Uses an autocommit connection, so each append is durable on return."""

    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    async def setup(self) -> None:
        await self.conn.execute(SETUP_SQL)

    async def append(self, run_id: str, seq: int, type: str, payload: dict[str, Any]) -> None:
        """Insert one event. `seq` is the optimistic lock: a duplicate raises ConcurrentAppend."""
        # >>> TODO: INSERT using Jsonb(payload); convert psycopg.errors.UniqueViolation to ConcurrentAppend
        try:
            await self.conn.execute(
                "INSERT INTO run_events (run_id, seq, type, payload) VALUES (%s, %s, %s, %s)",
                (run_id, seq, type, Jsonb(payload)),
            )
        except psycopg.errors.UniqueViolation as exc:
            raise ConcurrentAppend(f"{run_id} seq {seq} already exists") from exc
        # <<<

    async def read(self, run_id: str) -> list[Event]:
        """Return this run's events ordered by seq."""
        # >>> TODO: SELECT seq, type, payload ORDER BY seq and build Event objects
        cursor = await self.conn.execute(
            "SELECT seq, type, payload FROM run_events WHERE run_id = %s ORDER BY seq", (run_id,)
        )
        return [Event(seq=row[0], type=row[1], payload=row[2]) for row in await cursor.fetchall()]
        # <<<


Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


async def run_durable(log: EventLog, run_id: str, plan: ExecutionPlan, worker: Worker) -> RunState:
    """Run a plan, or resume it. Hydrate first; tasks with a task_completed event never re-run.

    If the log is empty, validate and append plan_created. Then loop: run the ready frontier with
    gather(return_exceptions=True), appending one event per result. A worker exception is a
    task_failed event, not a crash, and it stops the run after that superstep. A task that crashed
    the process has no event, so it runs again: this gives at-least-once execution.
    """
    # >>> TODO: hydrate, append plan_created if new, loop over ready_frontier appending events, stop on failure or when done
    state = hydrate(await log.read(run_id))
    if state.last_seq < 0:
        validate_plan(plan)
        await log.append(run_id, 0, "plan_created", {"tasks": [t.model_dump() for t in plan.tasks]})
        state = hydrate(await log.read(run_id))

    while True:
        if any(t.status == "failed" for t in state.tasks.values()):
            return state
        frontier = ready_frontier(state.tasks)
        if not frontier:
            check_deadlock(state.tasks)
            return state
        calls = [
            worker(task, {dep: state.tasks[dep].result or "" for dep in task.depends_on})
            for task in frontier
        ]
        outcomes = await asyncio.gather(*calls, return_exceptions=True)
        for task, outcome in zip(frontier, outcomes):
            # gather captures BaseException too. Cancellation or shutdown isn't a task failure.
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                raise outcome
            seq = state.last_seq + 1
            if isinstance(outcome, Exception):
                event = Event(seq=seq, type="task_failed", payload={"task_id": task.id, "error": str(outcome)})
            else:
                event = Event(seq=seq, type="task_completed", payload={"task_id": task.id, "result": outcome})
            await log.append(run_id, seq, event.type, event.payload)
            state = apply_event(state, event)
    # <<<
