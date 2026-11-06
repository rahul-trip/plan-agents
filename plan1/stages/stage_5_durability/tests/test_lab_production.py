import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from planagents.dag import checkpoint_serde
from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode, WorkerSpec

WORKERS = {"w": WorkerSpec(description="Safe"), "danger": WorkerSpec(description="Destructive", destructive=True)}

RUNNER = Path(__file__).parents[1] / "crash_runner.py"
CRASH_EXIT_CODE = 42


def task(id, *deps, destructive=False, args=None):
    return TaskNode(id=id, worker="danger" if destructive else "w", instruction=id, depends_on=list(deps), args=args or {})


PLAN = ExecutionPlan(
    tasks=[
        task("find_stale"),
        task("notify", "find_stale"),
        task("optional_audit", "find_stale"),
        task("delete_users", "notify", destructive=True, args={"limit": 1000}),
        task("report", "delete_users"),
    ]
)


class Recorder:
    def __init__(self):
        self.calls = {}

    async def __call__(self, task, upstream):
        self.calls.setdefault(task.id, []).append(task.args)
        return f"out-{task.id}"


@pytest.fixture
def lab(load):
    return load("lab_production")


def llm():
    return ScriptedLLM({"PlanDraft": [PLAN]})


async def test_full_flow_approve_edit_prune_gate_complete(lab, saver):
    worker = Recorder()
    orchestrator = lab.Orchestrator(llm(), worker, saver, workers=WORKERS)

    thread_id, _ = await orchestrator.start("Clean up stale users")
    review = await orchestrator.pending(thread_id)
    assert review[0]["type"] == "plan_review"
    assert worker.calls == {}

    state = await orchestrator.inspect(thread_id)
    json.dumps(state)  # The reviewer's view must be plain JSON.
    assert state["delete_users"]["args"] == {"limit": 1000}

    await orchestrator.resume(thread_id, {"action": "edit", "prune": ["optional_audit"]})
    gate = await orchestrator.pending(thread_id)
    assert gate[0]["task_id"] == "delete_users"
    assert "delete_users" not in worker.calls

    # A human catches the limit and lowers it mid-flight.
    result = await orchestrator.resume(thread_id, {"approved": True, "args": {"limit": 10}})
    assert worker.calls["delete_users"] == [{"limit": 10}]
    assert result["tasks"]["optional_audit"].status == "cancelled"
    assert result["tasks"]["report"].status == "completed"
    assert await orchestrator.pending(thread_id) == []


async def test_a_new_orchestrator_can_pick_up_a_paused_thread(lab, saver):
    """Nothing about the run lives in the object, as a restarted service would need."""
    first = lab.Orchestrator(llm(), Recorder(), saver, workers=WORKERS)
    thread_id, _ = await first.start("Clean up stale users")

    worker = Recorder()
    second = lab.Orchestrator(ScriptedLLM({}), worker, saver, workers=WORKERS)
    assert (await second.pending(thread_id))[0]["type"] == "plan_review"
    await second.resume(thread_id, {"action": "approve"})
    assert worker.calls["find_stale"] == [{}]


async def test_parallel_gates_can_be_answered_through_the_orchestrator(lab, saver):
    """Regression: pending() dropped the interrupt ids, so two gates at once couldn't be answered."""
    worker = Recorder()
    plan = ExecutionPlan(tasks=[task("drop_a", destructive=True), task("drop_b", destructive=True)])
    orchestrator = lab.Orchestrator(ScriptedLLM({"PlanDraft": [plan]}), worker, saver, workers=WORKERS)
    thread_id, _ = await orchestrator.start("Clean up")
    await orchestrator.resume(thread_id, {"action": "approve"})

    gates = await orchestrator.pending(thread_id)
    answers = {gate["interrupt_id"]: {"approved": gate["task_id"] == "drop_a"} for gate in gates}
    result = await orchestrator.resume(thread_id, answers)

    assert result["tasks"]["drop_a"].status == "completed"
    assert result["tasks"]["drop_b"].status == "cancelled"
    assert list(worker.calls) == ["drop_a"]


async def test_wrong_decision_form_at_review_does_not_lock_the_run(lab, saver):
    """Regression: {"approved": true}, which the CLI's prompt offered, locked the thread for good."""
    worker = Recorder()
    orchestrator = lab.Orchestrator(llm(), worker, saver, workers=WORKERS)
    thread_id, _ = await orchestrator.start("Clean up stale users")
    await orchestrator.resume(thread_id, {"approved": True})
    pending = await orchestrator.pending(thread_id)
    assert pending[0]["type"] == "plan_review" and pending[0]["error"]
    await orchestrator.resume(thread_id, {"action": "approve"})
    assert worker.calls["find_stale"] == [{}]


def run_process(url, thread, ledger, target, crash_on=None, resume=False):
    command = [
        sys.executable, str(RUNNER), "--mode", "lab", "--url", url, "--thread", thread,
        "--ledger", str(ledger), "--target", target,
    ]  # fmt: skip
    if crash_on:
        command += ["--crash-on", crash_on]
    if resume:
        command.append("--resume")
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


@pytest.mark.postgres
async def test_process_crash_after_approval_recovers_to_the_next_gate(lab, postgres_url, tmp_path, request):
    target = request.config.getoption("--target")
    ledger = tmp_path / "ledger.txt"

    # 1. This process plans the run and pauses at review.
    async with AsyncPostgresSaver.from_conn_string(postgres_url, serde=checkpoint_serde()) as saver:
        await saver.setup()
        orchestrator = lab.Orchestrator(llm(), Recorder(), saver, workers=WORKERS)
        thread_id, _ = await orchestrator.start("Clean up stale users", thread_id=f"lab-{uuid.uuid4()}")

    # 2. Another process approves and dies during 'notify', with its sibling running alongside.
    crashed = run_process(postgres_url, thread_id, ledger, target, crash_on="notify")
    assert crashed.returncode == CRASH_EXIT_CODE, crashed.stderr

    # 3. A third process recovers. The human's approval is not asked for again.
    resumed = run_process(postgres_url, thread_id, ledger, target, resume=True)
    assert resumed.returncode == 0, resumed.stderr

    runs = ledger.read_text().split()
    assert runs.count("find_stale") == 1, "completed before the crash"
    assert runs.count("notify") == 2, "in flight at the crash, so it runs again"
    assert "delete_users" not in runs, "the destructive task still waits at its gate"

    # 4. Back here, the run is waiting at the destructive gate, and a human can finish it.
    async with AsyncPostgresSaver.from_conn_string(postgres_url, serde=checkpoint_serde()) as saver:
        worker = Recorder()
        orchestrator = lab.Orchestrator(ScriptedLLM({}), worker, saver, workers=WORKERS)
        assert (await orchestrator.pending(thread_id))[0]["task_id"] == "delete_users"
        result = await orchestrator.resume(thread_id, {"approved": True, "args": {"limit": 10}})
    assert result["tasks"]["report"].status == "completed"
    assert worker.calls["delete_users"] == [{"limit": 10}]
