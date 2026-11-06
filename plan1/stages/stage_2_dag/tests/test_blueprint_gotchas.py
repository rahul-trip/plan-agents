"""Pins down the PDF blueprint's real behaviour. These run against blueprint_pdf.py, so they
pass under either --target. Read them before ex2: each one is a bug you will design out.
"""

import importlib.util
import sys
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "blueprint_pdf.py"
_spec = importlib.util.spec_from_file_location("_planagents_blueprint_pdf", _path)
bp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bp
_spec.loader.exec_module(bp)

CONFIG = {"recursion_limit": 25}


def _task(id, depends_on=()):
    return bp.TaskNode(id=id, worker="coder", instruction="x", depends_on=list(depends_on))


async def test_happy_path_runs_as_published():
    result = await bp.build_graph().ainvoke({"objective": "demo"}, CONFIG)
    assert all(t.status == "completed" for t in result["tasks"].values())
    assert "Completed 3 tasks" in result["final_output"]


def test_topological_sorter_accepts_a_dangling_dependency():
    """Gotcha 1: an unknown ID becomes a new node, so the cycle check says nothing."""
    sorter = TopologicalSorter({"task_1": {"task_99"}})
    sorter.prepare()  # No error.
    assert sorter.get_ready() == ("task_99",)


async def test_dangling_dependency_ends_the_run_silently():
    """Gotcha 2: nothing is ready, the router returns [], and the run stops with no output."""
    plan = bp.ExecutionPlan(tasks=[_task("task_1", ["task_99"])])
    result = await bp.build_graph(plan).ainvoke({"objective": "demo"}, CONFIG)
    assert result.get("final_output") is None
    assert result["tasks"]["task_1"].status == "pending"


def test_merge_tasks_cannot_remove_a_task():
    """Gotcha 3: the reducer only adds or overwrites, so a prune can't be expressed."""
    merged = bp.merge_tasks({"a": _task("a"), "b": _task("b")}, {"a": _task("a")})
    assert set(merged) == {"a", "b"}


async def test_stub_replanner_burns_the_whole_budget_without_fixing_anything():
    """Gotcha 4: the failed task stays failed, so all 3 replans are spent for nothing."""

    async def failing_worker(payload):
        task = payload["task"].model_copy(update={"status": "failed"})
        return {"tasks": {task.id: task}}

    result = await bp.build_graph(worker=failing_worker).ainvoke({"objective": "demo"}, CONFIG)
    assert result["replan_count"] == 3
    assert result["tasks"]["task_1"].status == "failed"


async def test_uncaught_worker_exception_aborts_the_whole_run():
    """Gotcha 9: without a try/except in the worker, one failure kills every sibling's run."""

    async def exploding_worker(payload):
        if payload["task"].id == "task_1":
            raise RuntimeError("boom")
        task = payload["task"].model_copy(update={"status": "completed"})
        return {"tasks": {task.id: task}}

    with pytest.raises(RuntimeError, match="boom"):
        await bp.build_graph(worker=exploding_worker).ainvoke({"objective": "demo"}, CONFIG)
