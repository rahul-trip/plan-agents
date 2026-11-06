"""Test helper (not an exercise): runs a durable DAG in its own process, so a crash is real.

The worker records each start in a ledger file, standing in for an external side effect. When it
reaches --crash-on it calls os._exit, which skips all cleanup, as a power cut or OOM kill would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from planagents.fakes import ScriptedLLM
from planagents.loader import import_stage_module
from planagents.schemas import ExecutionPlan, TaskNode

CRASH_EXIT_CODE = 42
STAGE_DIR = Path(__file__).parent


def demo_plan() -> ExecutionPlan:
    """a -> (b, c) -> d. Parallel b and c let the test crash a superstep halfway through."""

    def node(id: str, *deps: str) -> TaskNode:
        return TaskNode(id=id, worker="w", instruction=id, depends_on=list(deps))

    return ExecutionPlan(tasks=[node("a"), node("b", "a"), node("c", "a"), node("d", "b", "c")])


def make_worker(ledger: Path, crash_on: str | None):
    async def worker(task: TaskNode, upstream: dict[str, str]) -> str:
        with ledger.open("a") as file:
            file.write(task.id + "\n")
            file.flush()
            os.fsync(file.fileno())
        if task.id == crash_on:
            await asyncio.sleep(0.5)  # Long enough for a sibling's writes to reach Postgres.
            os._exit(CRASH_EXIT_CODE)
        return f"out-{task.id}"

    return worker


async def run_exercise(args: argparse.Namespace) -> dict:
    module = import_stage_module(STAGE_DIR, args.target, "ex2_crash_recovery")
    worker = make_worker(Path(args.ledger), args.crash_on)
    return await module.run_or_resume(args.url, args.thread, demo_plan(), worker, resume=args.resume)


async def run_lab(args: argparse.Namespace) -> dict:
    """Resume an existing lab thread (already past plan review) in this process."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from planagents.dag import checkpoint_serde

    module = import_stage_module(STAGE_DIR, args.target, "lab_production")
    worker = make_worker(Path(args.ledger), args.crash_on)
    async with AsyncPostgresSaver.from_conn_string(args.url, serde=checkpoint_serde()) as saver:
        orchestrator = module.Orchestrator(ScriptedLLM({}), worker, saver)
        if args.resume:
            return await orchestrator.recover(args.thread)
        return await orchestrator.resume(args.thread, {"action": "approve"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exercise", "lab"], default="exercise")
    parser.add_argument("--url", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--crash-on")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run = run_lab if args.mode == "lab" else run_exercise
    result = asyncio.run(run(args))
    statuses = {task_id: task.status for task_id, task in result["tasks"].items()}
    print(json.dumps(statuses))


if __name__ == "__main__":
    main()
