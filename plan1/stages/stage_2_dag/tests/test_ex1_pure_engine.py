import asyncio
from graphlib import CycleError

import pytest

from planagents.schemas import ExecutionPlan, TaskNode
from planagents.tracing import Timeline


def task(id, *deps):
    return TaskNode(id=id, worker="w", instruction="x", depends_on=list(deps))


DIAMOND = ExecutionPlan(tasks=[task("a"), task("b"), task("c", "a", "b"), task("d", "c")])


@pytest.fixture
def run_plan(load):
    return load("ex1_pure_engine").run_plan


async def test_independent_tasks_overlap_and_dependents_wait(run_plan):
    timeline = Timeline()

    async def worker(task, upstream):
        with timeline.span(task.id):
            await asyncio.sleep(0.05)
        return task.id

    tasks = await run_plan(DIAMOND, worker)
    assert all(t.status == "completed" for t in tasks.values())
    assert timeline.overlap("a", "b")
    assert timeline.starts_after("c", "a", "b")
    assert timeline.starts_after("d", "c")


async def test_worker_receives_only_direct_parents(run_plan):
    seen = {}

    async def worker(task, upstream):
        seen[task.id] = dict(upstream)
        return f"out-{task.id}"

    await run_plan(DIAMOND, worker)
    assert seen["a"] == {}
    assert seen["c"] == {"a": "out-a", "b": "out-b"}
    assert seen["d"] == {"c": "out-c"}, "grandparents must not appear"


async def test_failure_stops_dependents_without_hanging(run_plan):
    ran = []

    async def worker(task, upstream):
        ran.append(task.id)
        if task.id == "a":
            raise ValueError("boom")
        return "ok"

    tasks = await asyncio.wait_for(run_plan(DIAMOND, worker), timeout=2)
    assert tasks["a"].status == "failed" and "boom" in tasks["a"].result
    assert tasks["b"].status == "completed", "a sibling must still finish"
    assert tasks["c"].status == "pending" and tasks["d"].status == "pending"
    assert sorted(ran) == ["a", "b"]


async def test_cycle_raises(run_plan):
    async def worker(task, upstream):
        return "ok"

    with pytest.raises(CycleError):
        await run_plan(ExecutionPlan(tasks=[task("a", "b"), task("b", "a")]), worker)
