"""Stage 2 lab and capstone: prove a 6-node DAG really runs in parallel.

Docs, database and changelog fetches run in one superstep, and synthesis waits for everything
upstream. This lab imports your ex3 graph, so finish that first.

Live run with a real planner (needs LLM_MODEL in .env):
    uv run python stages/stage_2_dag/solutions/lab_parallel_dag.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from planagents.loader import sibling
from planagents.schemas import ExecutionPlan, TaskNode
from planagents.tracing import Timeline, trace_supersteps

ROOTS = ("fetch_api_docs", "query_database", "scrape_changelog")
OBJECTIVE = (
    "Assess an API upgrade. Fetch API docs, query the database and scrape the changelog "
    "independently, then extract breaking changes from the docs and changelog, compare the "
    "schema using the database and docs, and finally synthesize a report from those two."
)


def six_node_plan() -> ExecutionPlan:
    """The target shape: the 3 ROOTS run first, then

    extract_breaking_changes <- fetch_api_docs, scrape_changelog
    compare_schema           <- query_database, fetch_api_docs
    synthesize_report        <- extract_breaking_changes, compare_schema
    """
    # >>> TODO: return an ExecutionPlan with exactly these 6 TaskNodes (ids above)
    def node(id: str, *deps: str) -> TaskNode:
        return TaskNode(id=id, worker="researcher", instruction=id.replace("_", " "), depends_on=list(deps))

    return ExecutionPlan(
        tasks=[
            *(node(root) for root in ROOTS),
            node("extract_breaking_changes", "fetch_api_docs", "scrape_changelog"),
            node("compare_schema", "query_database", "fetch_api_docs"),
            node("synthesize_report", "extract_breaking_changes", "compare_schema"),
        ]
    )
    # <<<


@dataclass
class LabRun:
    timeline: Timeline
    batches: list[list[str]]


async def run_lab(llm, task_seconds: float = 0.05) -> LabRun:
    """Build your ex3 graph with a worker that records a Timeline span around a sleep, trace it
    with trace_supersteps, and return both."""
    build_graph = sibling(__file__, "ex3_blueprint_graph").build_graph
    # >>> TODO: write the timed worker, build the graph, call trace_supersteps with recursion_limit=25, return a LabRun
    timeline = Timeline()

    async def worker(task: TaskNode, upstream: dict[str, str]) -> str:
        with timeline.span(task.id):
            await asyncio.sleep(task_seconds)
        return f"{task.id} done"

    batches = await trace_supersteps(
        build_graph(llm, worker), {"objective": OBJECTIVE}, config={"recursion_limit": 25}
    )
    return LabRun(timeline=timeline, batches=batches)
    # <<<


if __name__ == "__main__":
    from planagents.llm import get_llm

    run = asyncio.run(run_lab(get_llm(), task_seconds=0.2))
    for number, batch in enumerate(run.batches, start=1):
        print(f"superstep batch {number}: {sorted(batch)}")
