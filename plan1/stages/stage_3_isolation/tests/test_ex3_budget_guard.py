from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from planagents.dag import merge_tasks, ready_frontier
from planagents.schemas import TaskNode


def task(id="child", *deps, instruction="short"):
    return TaskNode(id=id, worker="w", instruction=instruction, depends_on=list(deps))


@pytest.fixture
def ex(load):
    return load("ex3_budget_guard")


def test_non_parent_context_is_a_leak(ex, budget_guard):
    guard = budget_guard(expect_violations=True)
    guard.check(task("child", "parent"), {"parent": "ok"})
    with pytest.raises(ex.ContextLeak, match="grandparent"):
        guard.check(task("child", "parent"), {"parent": "ok", "grandparent": "leaked"})


def test_canary_is_detected_even_under_a_valid_key(ex, budget_guard):
    guard = budget_guard(canaries=["CANARY-123"], expect_violations=True)
    guard.check(task("child", "parent"), {"parent": "clean"})
    with pytest.raises(ex.ContextLeak, match="CANARY-123"):
        guard.check(task("child", "parent"), {"parent": "raw trace CANARY-123"})


def test_budget_is_enforced_and_largest_payload_tracked(ex, budget_guard):
    guard = budget_guard(max_tokens=10, expect_violations=True)
    guard.check(task("child", "parent"), {"parent": "x" * 16})
    with pytest.raises(ex.TokenBudgetExceeded):
        guard.check(task("child", "parent"), {"parent": "x" * 400})
    assert guard.checked == 1
    assert guard.largest_payload >= 4
    assert len(guard.violations) == 1


async def test_wrap_guards_a_worker(ex, budget_guard):
    calls = []

    async def worker(task, upstream):
        calls.append(task.id)
        return "ok"

    guard = budget_guard(canaries=["BAD"], expect_violations=True)
    guarded = guard.wrap(worker)
    assert await guarded(task("a"), {}) == "ok"
    with pytest.raises(ex.ContextLeak):
        await guarded(task("b", "p"), {"p": "BAD"})
    assert calls == ["a"], "the worker must not run on a rejected payload"


class _State(TypedDict, total=False):
    tasks: Annotated[dict[str, TaskNode], merge_tasks]


def _graph(tasks, worker, dispatch):
    """The Stage 2 shape: fan-out through Send, with a worker_executor that catches exceptions."""

    async def load_plan(state):
        return {"tasks": tasks}

    async def scheduler(state):
        return {}

    def route(state):
        current = state["tasks"]
        ready = ready_frontier(current)
        if not ready:
            return END
        return [Send("worker_executor", {"task": t, "upstream": dispatch(t, current)}) for t in ready]

    async def worker_executor(payload):
        target = payload["task"]
        try:
            update = {"status": "completed", "result": await worker(target, payload["upstream"])}
        except Exception as exc:
            update = {"status": "failed", "result": f"{type(exc).__name__}: {exc}"}
        return {"tasks": {target.id: target.model_copy(update=update)}}

    builder = StateGraph(_State)
    builder.add_node("load_plan", load_plan)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker_executor", worker_executor)
    builder.add_edge(START, "load_plan")
    builder.add_edge("load_plan", "scheduler")
    builder.add_conditional_edges("scheduler", route, ["worker_executor", END])
    builder.add_edge("worker_executor", "scheduler")
    return builder.compile()


def parents_only(task, tasks):
    return {dep: tasks[dep].result or "" for dep in task.depends_on}


def everything_so_far(task, tasks):
    return {t.id: t.result or "" for t in tasks.values() if t.status == "completed"}


CHAIN = {t.id: t for t in [task("a"), task("b", "a"), task("c", "b")]}


async def worker(task, upstream):
    return f"out-{task.id}"


async def test_guard_inside_worker_executor_passes_a_clean_dispatcher(budget_guard):
    guard = budget_guard()
    result = await _graph(CHAIN, guard.wrap(worker), parents_only).ainvoke({}, {"recursion_limit": 25})
    assert all(t.status == "completed" for t in result["tasks"].values())
    assert guard.checked == 3


async def test_guard_catches_a_leak_the_graph_swallowed(ex, budget_guard):
    """Regression: the worker's except turned the leak into a failed task, and the test still passed."""
    guard = budget_guard(expect_violations=True)
    result = await _graph(CHAIN, guard.wrap(worker), everything_so_far).ainvoke({}, {"recursion_limit": 25})
    assert result["tasks"]["c"].status == "failed", "no exception ever reached the caller"
    with pytest.raises(ex.ContextLeak, match="non-parent"):
        guard.assert_clean()
