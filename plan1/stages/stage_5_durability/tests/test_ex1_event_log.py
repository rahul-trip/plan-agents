import uuid

import psycopg
import pytest

from planagents.schemas import ExecutionPlan, TaskNode


def task(id, *deps):
    return TaskNode(id=id, worker="w", instruction=id, depends_on=list(deps))


PLAN = ExecutionPlan(tasks=[task("a"), task("b"), task("c", "a", "b")])


@pytest.fixture
def ex(load):
    return load("ex1_event_log")


def events(ex):
    return [
        ex.Event(seq=0, type="plan_created", payload={"tasks": [t.model_dump() for t in PLAN.tasks]}),
        ex.Event(seq=1, type="task_completed", payload={"task_id": "a", "result": "A"}),
        ex.Event(seq=2, type="task_failed", payload={"task_id": "b", "error": "boom"}),
    ]


def test_hydrate_rebuilds_state_from_events(ex):
    state = ex.hydrate(events(ex))
    assert state.last_seq == 2
    assert state.tasks["a"].status == "completed" and state.tasks["a"].result == "A"
    assert state.tasks["b"].status == "failed" and state.tasks["b"].result == "boom"
    assert state.tasks["c"].status == "pending"


def test_apply_event_is_pure(ex):
    before = ex.hydrate(events(ex)[:1])
    after = ex.apply_event(before, events(ex)[1])
    assert before.tasks["a"].status == "pending"
    assert after.tasks["a"].status == "completed"


def test_unknown_event_type_is_an_error_not_a_skip(ex):
    with pytest.raises(ValueError):
        ex.apply_event(ex.RunState(), ex.Event(seq=0, type="mystery", payload={}))


@pytest.fixture
async def log(ex, postgres_url):
    async with await psycopg.AsyncConnection.connect(postgres_url, autocommit=True) as conn:
        event_log = ex.EventLog(conn)
        await event_log.setup()
        yield event_log


def run_id():
    return f"test-{uuid.uuid4()}"


@pytest.mark.postgres
async def test_append_read_and_optimistic_lock(ex, log):
    rid = run_id()
    await log.append(rid, 0, "plan_created", {"tasks": []})
    with pytest.raises(ex.ConcurrentAppend):
        await log.append(rid, 0, "plan_created", {"tasks": []})
    assert [e.seq for e in await log.read(rid)] == [0]


@pytest.mark.postgres
@pytest.mark.parametrize("statement", ["UPDATE run_events SET type = 'x'", "DELETE FROM run_events"])
async def test_log_is_append_only_at_the_database_level(ex, log, statement):
    rid = run_id()
    await log.append(rid, 0, "plan_created", {"tasks": []})
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        await log.conn.execute(f"{statement} WHERE run_id = %s", (rid,))


@pytest.mark.postgres
async def test_resume_after_a_crash_skips_completed_tasks(ex, log):
    rid = run_id()
    ran = []

    class Crash(BaseException):
        """A stand-in for the process dying. gather(return_exceptions=True) still captures a
        BaseException, so run_durable must re-raise it rather than log it as a result."""

    async def crashing(task, upstream):
        ran.append(task.id)
        if task.id == "c":
            raise Crash()
        return f"out-{task.id}"

    with pytest.raises(Crash):
        await ex.run_durable(log, rid, PLAN, crashing)
    assert sorted(ran) == ["a", "b", "c"]

    ran.clear()

    async def healthy(task, upstream):
        ran.append(task.id)
        assert upstream == {"a": "out-a", "b": "out-b"}, "upstream results come back from the log"
        return f"out-{task.id}"

    state = await ex.run_durable(log, rid, PLAN, healthy)
    assert ran == ["c"], "only the task with no completion event re-runs"
    assert all(t.status == "completed" for t in state.tasks.values())
    assert state.tasks == ex.hydrate(await log.read(rid)).tasks, "state is fully derivable from the log"


@pytest.mark.postgres
async def test_worker_exception_is_recorded_not_raised(ex, log):
    rid = run_id()

    async def worker(task, upstream):
        if task.id == "a":
            raise ValueError("boom")
        return "ok"

    state = await ex.run_durable(log, rid, PLAN, worker)
    assert state.tasks["a"].status == "failed"
    assert "task_failed" in [e.type for e in await log.read(rid)]
