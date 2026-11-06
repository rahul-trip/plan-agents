import pytest

from planagents.schemas import ExecutionPlan, TaskNode

CONFIG = {"recursion_limit": 25}


def task(id, *deps):
    return TaskNode(id=id, worker="w", instruction=f"do {id}", depends_on=list(deps))


PLAN = ExecutionPlan(tasks=[task("task_1"), task("task_2"), task("task_3", "task_1")])


@pytest.fixture
def ex(load):
    return load("ex2_blackboard")


def test_resolve_inputs_injects_summary_only(ex):
    artifacts = {
        "task_1": ex.Artifact(summary="short", raw="CANARY-RAW"),
        "task_2": ex.Artifact(summary="CANARY-UNRELATED"),
    }
    worker_input = ex.resolve_inputs(task("task_3", "task_1"), artifacts)
    assert set(worker_input) == {"task_id", "instruction", "inputs"}
    assert worker_input["inputs"] == {"task_1": "short"}
    assert "CANARY" not in str(worker_input)


async def test_worker_input_never_contains_global_state(ex):
    received = []

    async def worker(worker_input):
        received.append(worker_input)
        return ex.Artifact(summary=f"sum-{worker_input['task_id']}", raw=f"CANARY-RAW-{worker_input['task_id']}")

    result = await ex.build_graph(PLAN, worker).ainvoke({}, CONFIG)

    assert len(received) == 3
    for worker_input in received:
        assert set(worker_input) == {"task_id", "instruction", "inputs"}
        assert "CANARY" not in str(worker_input), "raw output reached a worker"
    task_3 = next(w for w in received if w["task_id"] == "task_3")
    assert task_3["inputs"] == {"task_1": "sum-task_1"}, "only the referenced summary"
    assert all(t.status == "completed" for t in result["tasks"].values())


async def test_worker_failure_is_contained(ex):
    async def worker(worker_input):
        if worker_input["task_id"] == "task_1":
            raise RuntimeError("boom")
        return ex.Artifact(summary="ok")

    result = await ex.build_graph(PLAN, worker).ainvoke({}, CONFIG)
    assert result["tasks"]["task_1"].status == "failed" and "boom" in result["tasks"]["task_1"].result
    assert result["tasks"]["task_2"].status == "completed", "the sibling's work must survive"
    assert result["tasks"]["task_3"].status == "pending"


async def test_raw_output_stays_on_the_blackboard_for_audit(ex):
    async def worker(worker_input):
        return ex.Artifact(summary="s", raw=f"trace of {worker_input['task_id']}")

    result = await ex.build_graph(PLAN, worker).ainvoke({}, CONFIG)
    assert result["artifacts"]["task_1"].raw == "trace of task_1"
    assert "trace of" not in result["tasks"]["task_1"].result
