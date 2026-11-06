import pytest
from pydantic import ValidationError

from planagents.schemas import TaskNode


def task(id, *deps, status="pending", result=None):
    return TaskNode(id=id, worker="w", instruction=f"do {id}", depends_on=list(deps), status=status, result=result)


TASKS = {
    "grandparent": task("grandparent", status="completed", result="CANARY-GRANDPARENT"),
    "parent": task("parent", "grandparent", status="completed", result="parent output"),
    "unrelated": task("unrelated", status="completed", result="CANARY-UNRELATED"),
    "child": task("child", "parent"),
}


@pytest.fixture
def ex(load):
    return load("ex1_scoped_workers")


def test_slice_holds_only_direct_parents(ex):
    state = ex.make_slice(TASKS["child"], TASKS)
    assert dict(state.upstream) == {"parent": "parent output"}
    assert "CANARY" not in state.model_dump_json(exclude={"upstream"}) + str(dict(state.upstream))


def test_slice_fields_cannot_be_reassigned(ex):
    state = ex.make_slice(TASKS["child"], TASKS)
    with pytest.raises(ValidationError):
        state.instruction = "tampered"


def test_nested_upstream_cannot_be_mutated(ex):
    """The part frozen=True alone doesn't give you."""
    state = ex.make_slice(TASKS["child"], TASKS)
    with pytest.raises(TypeError):
        state.upstream["parent"] = "tampered"


def test_slice_is_detached_from_its_source(ex):
    source = {"parent": "original"}
    state = ex.StateSlice(task_id="t", instruction="i", upstream=source)
    source["parent"] = "changed later"
    assert state.upstream["parent"] == "original"


async def test_worker_returns_updated_copy_without_touching_state(ex):
    class Upper(ex.ScopedWorker):
        async def handle(self, state):
            return state.upstream["parent"].upper()

    before = {key: value.model_copy() for key, value in TASKS.items()}
    updated = await Upper().run(TASKS["child"], TASKS)
    assert updated.status == "completed" and updated.result == "PARENT OUTPUT"
    assert TASKS == before
    assert TASKS["child"].status == "pending"


async def test_worker_failure_becomes_a_failed_task(ex):
    class Broken(ex.ScopedWorker):
        async def handle(self, state):
            raise KeyError("nope")

    updated = await Broken().run(TASKS["child"], TASKS)
    assert updated.status == "failed" and "nope" in updated.result
