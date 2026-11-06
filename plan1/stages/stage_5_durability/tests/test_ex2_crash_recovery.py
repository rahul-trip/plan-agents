import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from planagents.dag import checkpoint_serde
from planagents.schemas import ExecutionPlan, TaskNode

RUNNER = Path(__file__).parents[1] / "crash_runner.py"
CRASH_EXIT_CODE = 42
PLAN = ExecutionPlan(
    tasks=[
        TaskNode(id="a", worker="w", instruction="a"),
        TaskNode(id="b", worker="w", instruction="b", depends_on=["a"]),
    ]
)


@pytest.fixture
def ex(load):
    return load("ex2_crash_recovery")


async def echo(task, upstream):
    return f"out-{task.id}"


def config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}


async def test_graph_runs_and_checkpoints_each_superstep(ex):
    saver = InMemorySaver(serde=checkpoint_serde())
    graph = ex.build_graph(PLAN, echo, saver)
    cfg = config()
    result = await graph.ainvoke({}, cfg)
    assert all(t.status == "completed" for t in result["tasks"].values())
    assert len([s async for s in graph.aget_state_history(cfg)]) > 4


async def test_serde_allowlist_rebuilds_pydantic_without_warning(ex, caplog):
    """Gotcha: without an allowlist LangGraph warns that this load will be blocked in future."""
    saver = InMemorySaver(serde=checkpoint_serde())
    graph = ex.build_graph(PLAN, echo, saver)
    cfg = config()
    with caplog.at_level(logging.WARNING):
        await graph.ainvoke({}, cfg)
        state = await graph.aget_state(cfg)
    assert isinstance(state.values["tasks"]["a"], TaskNode), "must come back as the model, not a dict"
    assert "unregistered" not in caplog.text


async def test_default_serde_warns_about_our_models(ex, caplog):
    """Pins the behaviour the allowlist exists to fix. Remove if LangGraph starts blocking outright."""
    graph = ex.build_graph(PLAN, echo, InMemorySaver())
    cfg = config()
    with caplog.at_level(logging.WARNING):
        await graph.ainvoke({}, cfg)
        await graph.aget_state(cfg)
    assert "unregistered" in caplog.text


def run_process(url, thread, ledger, target, crash_on=None, resume=False):
    command = [sys.executable, str(RUNNER), "--url", url, "--thread", thread, "--ledger", str(ledger), "--target", target]
    if crash_on:
        command += ["--crash-on", crash_on]
    if resume:
        command.append("--resume")
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


@pytest.mark.postgres
def test_real_process_crash_resumes_without_redoing_finished_work(ex, postgres_url, tmp_path, request):
    """a -> (b, c) -> d. The process dies during c, after its sibling b finished."""
    target = request.config.getoption("--target")
    thread, ledger = f"crash-{uuid.uuid4()}", tmp_path / "ledger.txt"

    crashed = run_process(postgres_url, thread, ledger, target, crash_on="c")
    assert crashed.returncode == CRASH_EXIT_CODE, crashed.stderr
    assert "d" not in ledger.read_text().split()

    resumed = run_process(postgres_url, thread, ledger, target, resume=True)
    assert resumed.returncode == 0, resumed.stderr
    assert set(json.loads(resumed.stdout.splitlines()[-1]).values()) == {"completed"}

    runs = ledger.read_text().split()
    assert runs.count("a") == 1, "a finished in an earlier superstep"
    assert runs.count("b") == 1, "b finished in the crashed superstep: its saved writes must be reused"
    assert runs.count("c") == 2, "c was mid-flight, so it runs again: at-least-once, not exactly-once"
    assert runs.count("d") == 1
