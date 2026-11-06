import pytest
from pydantic import BaseModel, ValidationError

from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode

CONFIG = {"recursion_limit": 25}
TASK = TaskNode(id="a", worker="w", instruction="x")


class Answer(BaseModel):
    value: int


def schema_error() -> ValidationError:
    try:
        Answer.model_validate({"value": "many"})
    except ValidationError as exc:
        return exc


def scripted(*outcomes):
    """A worker that raises or returns each outcome in turn, recording the feedback it got."""
    remaining, feedback = list(outcomes), []

    async def worker(task, fb):
        feedback.append(fb)
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    worker.feedback = feedback
    return worker


@pytest.fixture
def ex(load):
    return load("ex2_two_tier")


def llm():
    return ScriptedLLM({"PlanDraft": [ExecutionPlan(tasks=[TASK])]})


def test_classify_failure(ex):
    assert ex.classify_failure(ex.TransientError()) == "transient"
    assert ex.classify_failure(TimeoutError()) == "transient"
    assert ex.classify_failure(schema_error()) == "schema"
    assert ex.classify_failure(ex.PermanentError()) == "permanent"
    assert ex.classify_failure(KeyError("x")) == "permanent", "unknown errors must not be retried"


async def test_schema_failure_feeds_the_error_back(ex):
    worker = scripted(schema_error(), "fixed")
    result, failure = await ex.run_with_local_retry(TASK, worker)
    assert (result, failure) == ("fixed", None)
    assert worker.feedback[0] is None
    assert "value" in worker.feedback[1]


async def test_permanent_failure_stops_immediately(ex):
    worker = scripted(ex.PermanentError("403"))
    result, failure = await ex.run_with_local_retry(TASK, worker, max_attempts=5)
    assert result is None
    assert (failure.kind, failure.attempts) == ("permanent", 1)
    assert "403" in failure.detail


async def test_retries_are_bounded(ex):
    worker = scripted(*[ex.TransientError() for _ in range(9)])
    _, failure = await ex.run_with_local_retry(TASK, worker, max_attempts=3)
    assert (failure.kind, failure.attempts) == ("transient", 3)


async def test_transient_failure_recovers_with_zero_planner_calls(ex):
    """The whole point of tier 1."""
    fake = llm()
    worker = scripted(ex.TransientError(), ex.TransientError(), "ok")
    result = await ex.build_graph(fake, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["a"].status == "completed"
    assert len(fake.calls_for("PlanDraft")) == 1, "only the initial plan"
    assert result["replan_requests"] == 0


async def test_impossible_task_escalates_to_a_global_replan(ex):
    worker = scripted(ex.PermanentError("source does not exist"))
    result = await ex.build_graph(llm(), worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["a"].status == "failed"
    assert "does not exist" in result["tasks"]["a"].result
    assert result["replan_requests"] == 1
