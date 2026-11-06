import asyncio

import pytest

from planagents.dag import DeadlockError, PlanValidationError
from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode
from planagents.tracing import Timeline, trace_supersteps

CONFIG = {"recursion_limit": 25}


def task(id, *deps, status="pending"):
    return TaskNode(id=id, worker="w", instruction="x", depends_on=list(deps), status=status)


PLAN = ExecutionPlan(tasks=[task("task_1"), task("task_2"), task("task_3", "task_1", "task_2")])


def llm_for(plan=PLAN):
    return ScriptedLLM({"PlanDraft": [plan]})


async def echo(task, upstream):
    return f"out-{task.id}"


@pytest.fixture
def ex(load):
    return load("ex3_blueprint_graph")


async def test_completes_and_synthesizes(ex):
    result = await ex.build_graph(llm_for(), echo).ainvoke({"objective": "goal"}, CONFIG)
    assert all(t.status == "completed" for t in result["tasks"].values())
    assert "out-task_3" in result["final_output"]


async def test_workers_run_in_parallel_then_wait_at_the_barrier(ex):
    timeline = Timeline()

    async def worker(task, upstream):
        with timeline.span(task.id):
            await asyncio.sleep(0.05)
        return "ok"

    await ex.build_graph(llm_for(), worker).ainvoke({"objective": "goal"}, CONFIG)
    assert timeline.overlap("task_1", "task_2")
    assert timeline.starts_after("task_3", "task_1", "task_2")


async def test_superstep_trace_shows_parallel_batch_then_barrier(ex):
    """Study workflow #3: task_1 and task_2 in one batch, the barrier, then task_3."""
    batches = await trace_supersteps(
        ex.build_graph(llm_for(), echo), {"objective": "goal"}, config=CONFIG
    )
    assert [sorted(batch) for batch in batches] == [["task_1", "task_2"], ["task_3"]]


async def test_upstream_context_is_only_direct_parents(ex):
    seen = {}
    chain = ExecutionPlan(tasks=[task("a"), task("b", "a"), task("c", "b")])

    async def worker(task, upstream):
        seen[task.id] = dict(upstream)
        return f"out-{task.id}"

    await ex.build_graph(llm_for(chain), worker).ainvoke({"objective": "goal"}, CONFIG)
    assert seen["c"] == {"b": "out-b"}


async def test_dangling_plan_rejected_at_planning(ex):
    bad = ExecutionPlan(tasks=[task("a", "ghost")])
    llm = ScriptedLLM(lambda key, prompt: bad)
    with pytest.raises(PlanValidationError):
        await ex.build_graph(llm, echo).ainvoke({"objective": "goal"}, CONFIG)
    assert len(llm.calls) == 3, "make_plan re-asks twice before giving up"


async def test_llm_cannot_mark_a_task_finished(ex):
    """Regression: a plan carrying status and result skipped the worker and shipped the fake result."""
    ran = []

    async def worker(task, upstream):
        ran.append(task.id)
        return "real"

    fake = {
        "tasks": [
            {"id": "a", "worker": "w", "instruction": "x", "depends_on": [], "args_json": "{}",
             "status": "completed", "result": "HALLUCINATED"}
        ]
    }  # fmt: skip
    llm = ScriptedLLM({"PlanDraft": [fake]})
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert ran == ["a"]
    assert result["tasks"]["a"].result == "real"


def test_router_raises_on_deadlock_instead_of_ending_silently(ex):
    with pytest.raises(DeadlockError):
        ex.route_ready_tasks({"tasks": {"a": task("a", "cancelled_parent")}})


async def test_failure_is_contained_and_degrades_gracefully(ex):
    async def worker(task, upstream):
        if task.id == "task_1":
            raise RuntimeError("boom")
        return "ok"

    result = await ex.build_graph(llm_for(), worker).ainvoke({"objective": "goal"}, CONFIG)
    tasks = result["tasks"]
    assert tasks["task_1"].status == "failed" and "boom" in tasks["task_1"].result
    assert tasks["task_2"].status == "completed", "the sibling's work must survive"
    assert tasks["task_3"].status == "pending"
    assert "failed" in result["final_output"]
