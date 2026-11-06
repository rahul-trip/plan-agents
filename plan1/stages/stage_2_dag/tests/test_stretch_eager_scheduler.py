import asyncio

import pytest

from planagents.schemas import ExecutionPlan, TaskNode
from planagents.tracing import Timeline

DURATIONS = {"slow": 0.30, "fast": 0.02, "after_fast": 0.02}
PLAN = ExecutionPlan(
    tasks=[
        TaskNode(id="slow", worker="w", instruction="x"),
        TaskNode(id="fast", worker="w", instruction="x"),
        TaskNode(id="after_fast", worker="w", instruction="x", depends_on=["fast"]),
    ]
)


@pytest.fixture
def run_plan_eager(load):
    return load("stretch_eager_scheduler").run_plan_eager


def timed_worker(timeline):
    async def worker(task, upstream):
        with timeline.span(task.id):
            await asyncio.sleep(DURATIONS[task.id])
        return f"out-{task.id}"

    return worker


async def test_barrier_delays_unrelated_work_and_eager_does_not(run_plan_eager, load):
    """The measurement: the same plan under both schedulers."""
    eager, barrier = Timeline(), Timeline()

    tasks = await run_plan_eager(PLAN, timed_worker(eager))
    await load("ex1_pure_engine").run_plan(PLAN, timed_worker(barrier))

    assert all(t.status == "completed" for t in tasks.values())
    assert eager.starts_after("after_fast", "fast"), "dependencies still hold"
    assert eager.spans["after_fast"].start < eager.spans["slow"].end, "eager doesn't wait for slow"
    assert barrier.starts_after("after_fast", "slow"), "the barrier does, though the two are unrelated"
    print(
        f"after_fast finished at {eager.spans['after_fast'].end - eager.spans['fast'].start:.2f}s eager, "
        f"{barrier.spans['after_fast'].end - barrier.spans['fast'].start:.2f}s at a barrier"
    )


async def test_upstream_results_are_passed(run_plan_eager):
    seen = {}

    async def worker(task, upstream):
        seen[task.id] = dict(upstream)
        return f"out-{task.id}"

    await run_plan_eager(PLAN, worker)
    assert seen["after_fast"] == {"fast": "out-fast"}
