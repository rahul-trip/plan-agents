import pytest

from planagents.schemas import ExecutionPlan, TaskNode


def task(id, *deps, status="pending"):
    return TaskNode(id=id, worker="w", instruction="x", depends_on=list(deps), status=status)


def as_map(*tasks):
    return {t.id: t for t in tasks}


@pytest.fixture
def ex(load):
    return load("ex2_frontier")


def test_valid_plan_returns_task_map(ex):
    plan = ExecutionPlan(tasks=[task("a"), task("b", "a")])
    assert set(ex.validate_plan(plan)) == {"a", "b"}


@pytest.mark.parametrize(
    "tasks",
    [
        pytest.param([task("a"), task("a")], id="duplicate"),
        pytest.param([task("a", "b"), task("b", "a")], id="cycle"),
        pytest.param([task("a", "a")], id="self-dependency"),
    ],
)
def test_invalid_plans_are_rejected(ex, tasks):
    with pytest.raises(ex.PlanValidationError):
        ex.validate_plan(ExecutionPlan(tasks=tasks))


def test_dangling_dependency_is_rejected_and_named(ex):
    with pytest.raises(ex.PlanValidationError, match="ghost"):
        ex.validate_plan(ExecutionPlan(tasks=[task("a", "ghost")]))


def test_merge_tasks(ex):
    original = {"a": task("a")}
    merged = ex.merge_tasks(original, {"a": task("a", status="completed"), "b": task("b")})
    assert merged["a"].status == "completed" and set(merged) == {"a", "b"}
    assert original["a"].status == "pending", "must not mutate"
    assert set(ex.merge_tasks(None, {"a": task("a")})) == {"a"}


def test_frontier_only_contains_unblocked_pending_tasks(ex):
    tasks = as_map(
        task("done", status="completed"),
        task("ready", "done"),
        task("blocked", "ready"),
        task("root"),
    )
    assert {t.id for t in ex.ready_frontier(tasks)} == {"ready", "root"}


def test_failed_parent_does_not_unblock(ex):
    tasks = as_map(task("a", status="failed"), task("b", "a"))
    assert ex.ready_frontier(tasks) == []


def test_deadlock_detected_and_named(ex):
    tasks = as_map(task("a", status="failed"), task("stuck", "a"))
    with pytest.raises(ex.DeadlockError, match="stuck"):
        ex.check_deadlock(tasks)


def test_no_deadlock_when_finished_or_runnable(ex):
    ex.check_deadlock(as_map(task("a", status="completed")))
    ex.check_deadlock(as_map(task("a")))
