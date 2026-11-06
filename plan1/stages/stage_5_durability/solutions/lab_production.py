"""Stage 5 lab and capstone: a production orchestrator.

The ex3 graph on a real checkpointer, behind the API a service would expose: start a run, see what
it's waiting on, inspect its state, resume it with a decision, and recover it after a crash.

Try it locally (docker compose up -d --wait, and LLM_MODEL in .env):
    uv run python stages/stage_5_durability/solutions/lab_production.py "Clean up stale users"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from planagents.dag import checkpoint_serde
from planagents.loader import sibling
from planagents.schemas import WorkerSpec

# Which actions need a person is decided here, in code, not by the planner.
DEMO_WORKERS = {
    "read_data": WorkerSpec(description="Reads data from the database."),
    "write_report": WorkerSpec(description="Writes a report from its dependencies' results."),
    "delete_data": WorkerSpec(description="Permanently deletes data from the database.", destructive=True),
}


class Orchestrator:
    """Every method takes a thread_id, and the run's state lives entirely in the checkpointer. So
    any process can pick up any thread: nothing about a run is held in this object."""

    def __init__(self, llm, worker, checkpointer, workers=None, recursion_limit: int = 100):
        build_graph = sibling(__file__, "ex3_hitl").build_graph
        self.graph = build_graph(llm, worker, checkpointer, workers=workers)
        self.recursion_limit = recursion_limit

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

    async def start(self, objective: str, thread_id: str | None = None) -> tuple[str, dict]:
        """Start a run. Returns (thread_id, result), where result pauses at plan review."""
        # >>> TODO: generate a uuid thread_id if none was given; ainvoke with {'objective': ...} and durability='sync'
        thread_id = thread_id or str(uuid.uuid4())
        result = await self.graph.ainvoke({"objective": objective}, self._config(thread_id), durability="sync")
        return thread_id, result
        # <<<

    async def resume(self, thread_id: str, decision: Any) -> dict:
        """Answer what's pending. One pending interrupt takes a decision. Several, as with two gates
        in one superstep, take a map of {interrupt_id: decision} using the ids from pending()."""
        # >>> TODO: ainvoke Command(resume=decision) with durability='sync'
        return await self.graph.ainvoke(Command(resume=decision), self._config(thread_id), durability="sync")
        # <<<

    async def recover(self, thread_id: str) -> dict:
        """Continue a run that died mid-flight, from its last checkpoint."""
        # >>> TODO: ainvoke None with durability='sync'
        return await self.graph.ainvoke(None, self._config(thread_id), durability="sync")
        # <<<

    async def pending(self, thread_id: str) -> list[dict]:
        """What the run is waiting on, from aget_state(...).interrupts: each payload plus its
        'interrupt_id', which a caller needs to answer several at once."""
        # >>> TODO: return [{'interrupt_id': i.id, **i.value} for i in the snapshot's interrupts]
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return [{"interrupt_id": item.id, **item.value} for item in snapshot.interrupts]
        # <<<

    async def inspect(self, thread_id: str) -> dict[str, dict]:
        """The current tasks, as JSON-ready dicts: what an API would return to a reviewer."""
        # >>> TODO: return {id: task.model_dump(mode='json')} from aget_state
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return {task_id: task.model_dump(mode="json") for task_id, task in snapshot.values["tasks"].items()}
        # <<<


HELP = {
    "plan_review": '{"action": "approve"}, {"action": "reject"} or {"action": "edit", "edits": {...}, "prune": [...]}',
    "destructive_action": '{"approved": true}, {"approved": false} or {"approved": true, "args": {...}}',
}


def ask_for_decision(item: dict) -> Any:
    print(json.dumps(item, indent=2))
    while True:
        try:
            return json.loads(input(f"Decision as JSON, e.g. {HELP[item['type']]}: "))
        except json.JSONDecodeError as exc:
            print(f"That isn't valid JSON: {exc}")


async def main(objective: str) -> None:
    from planagents.llm import get_llm

    async def worker(task, upstream):
        return f"(demo) did: {task.instruction} with {task.args}"

    url = os.environ["POSTGRES_URL"]
    async with AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()) as saver:
        await saver.setup()
        orchestrator = Orchestrator(get_llm(), worker, saver, workers=DEMO_WORKERS)
        thread_id, result = await orchestrator.start(objective)
        while "__interrupt__" in result:
            items = await orchestrator.pending(thread_id)
            answers = {item["interrupt_id"]: ask_for_decision(item) for item in items}
            result = await orchestrator.resume(thread_id, next(iter(answers.values())) if len(answers) == 1 else answers)
        print(result["final_output"])


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main(sys.argv[1]))
