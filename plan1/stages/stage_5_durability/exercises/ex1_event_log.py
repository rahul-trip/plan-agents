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
    raise NotImplementedError("copy the tasks dict, apply the event with model_copy, return a new RunState with last_seq=event.seq")


def hydrate(events: list[Event]) -> RunState:
    """Rebuild state by folding apply_event over the events in order."""
    raise NotImplementedError("fold apply_event starting from RunState()")


class EventLog:
    """Postgres-backed log. Uses an autocommit connection, so each append is durable on return."""

    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    async def setup(self) -> None:
        await self.conn.execute(SETUP_SQL)

    async def append(self, run_id: str, seq: int, type: str, payload: dict[str, Any]) -> None:
        """Insert one event. `seq` is the optimistic lock: a duplicate raises ConcurrentAppend."""
        raise NotImplementedError("INSERT using Jsonb(payload); convert psycopg.errors.UniqueViolation to ConcurrentAppend")

    async def read(self, run_id: str) -> list[Event]:
        """Return this run's events ordered by seq."""
        raise NotImplementedError("SELECT seq, type, payload ORDER BY seq and build Event objects")


Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


async def run_durable(log: EventLog, run_id: str, plan: ExecutionPlan, worker: Worker) -> RunState:
    """Run a plan, or resume it. Hydrate first; tasks with a task_completed event never re-run.

    If the log is empty, validate and append plan_created. Then loop: run the ready frontier with
    gather(return_exceptions=True), appending one event per result. A worker exception is a
    task_failed event, not a crash, and it stops the run after that superstep. A task that crashed
    the process has no event, so it runs again: this gives at-least-once execution.
    """
    raise NotImplementedError("hydrate, append plan_created if new, loop over ready_frontier appending events, stop on failure or when done")
