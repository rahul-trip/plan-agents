import pytest

from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode


def plan(worker):
    return ExecutionPlan(tasks=[TaskNode(id="fetch", worker=worker, instruction="fetch")])


@pytest.fixture
def ex(load):
    return load("ex3_anti_thrashing")


@pytest.fixture
def failed(ex):
    return [ex.FailedAttempt(task_id="fetch", tool="primary_api", reason="403 Forbidden")]


def test_constraints_name_the_tool_and_reason(ex, failed):
    text = ex.render_negative_constraints(failed)
    assert "primary_api" in text and "403 Forbidden" in text


def test_empty_log_adds_nothing_to_the_prompt(ex):
    assert ex.render_negative_constraints([]) == ""


def test_repeated_failures_ignores_tasks_that_already_finished(ex, failed):
    tasks = [
        TaskNode(id="again", worker="primary_api", instruction="x"),
        TaskNode(id="old", worker="primary_api", instruction="x", status="failed"),
        TaskNode(id="new", worker="fallback_api", instruction="x"),
    ]
    assert ex.repeated_failures(tasks, failed) == ["again"]


async def test_prompt_contains_negative_constraints(ex, failed):
    llm = ScriptedLLM({"PlanDraft": [plan("fallback_api")]})
    result = await ex.guarded_replan(llm, "goal", failed, replan_count=0)
    assert result.tasks[0].worker == "fallback_api"
    assert "403 Forbidden" in llm.calls[0].prompt


async def test_guard_rejects_an_llm_that_ignores_the_negative_prompt(ex, failed):
    llm = ScriptedLLM({"PlanDraft": [plan("primary_api"), plan("fallback_api")]})
    result = await ex.guarded_replan(llm, "goal", failed, replan_count=0)
    assert result.tasks[0].worker == "fallback_api"
    assert len(llm.calls) == 2
    assert llm.calls[1].prompt != llm.calls[0].prompt, "the retry must tell the LLM what was rejected"


async def test_stubborn_llm_is_cut_off(ex, failed):
    llm = ScriptedLLM(lambda key, prompt: plan("primary_api"))
    with pytest.raises(ex.ReplanBudgetExhausted):
        await ex.guarded_replan(llm, "goal", failed, replan_count=0, max_guard_rejections=2)
    assert len(llm.calls) == 3, "the first try plus 2 rejected retries"


async def test_max_replans_is_a_hard_stop_with_no_llm_call(ex, failed):
    llm = ScriptedLLM({})
    with pytest.raises(ex.ReplanBudgetExhausted):
        await ex.guarded_replan(llm, "goal", failed, replan_count=3, max_replans=3)
    assert llm.calls == []
