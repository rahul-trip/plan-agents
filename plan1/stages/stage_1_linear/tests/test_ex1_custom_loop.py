import pytest

from planagents.fakes import ScriptedLLM
from planagents.schemas import Plan, ReplanDecision, Step


def step(id: str, worker: str = "w") -> Step:
    return Step(id=id, description=f"do {id}", assigned_worker=worker)


async def echo(step, scratchpad):
    scratchpad.tool_history.append(f"called tool for {step.id}")
    return f"result of {step.id}"


@pytest.fixture
def ex(load):
    return load("ex1_custom_loop")


async def test_runs_steps_in_order(ex):
    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a"), step("b")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("b")]), ReplanDecision()],
        }
    )
    state = await ex.run_plan_loop("goal", llm, {"w": echo})
    assert [s.id for s in state.completed] == ["a", "b"]
    assert state.completed[1].status == "completed"
    assert state.completed[1].result == "result of b"
    assert not state.plan_queue


async def test_replanner_can_rewrite_the_queue(ex):
    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a"), step("b")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("new")]), ReplanDecision()],
        }
    )
    state = await ex.run_plan_loop("goal", llm, {"w": echo})
    assert [s.id for s in state.completed] == ["a", "new"]


async def test_replanner_sees_results_of_earlier_steps(ex):
    llm = ScriptedLLM({"Plan": [Plan(steps=[step("a")])], "ReplanDecision": [ReplanDecision()]})
    await ex.run_plan_loop("goal", llm, {"w": echo})
    assert "result of a" in llm.calls_for("ReplanDecision")[0].prompt


async def test_scratchpad_is_discarded_after_each_step(ex):
    seen = []

    async def worker(step, scratchpad):
        assert scratchpad.tool_history == [], "each step must start with a fresh scratchpad"
        scratchpad.tool_history.append(f"CANARY-RAW-TOOL-OUTPUT-{step.id}")
        seen.append(scratchpad)
        return "clean summary"

    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a"), step("b")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("b")]), ReplanDecision()],
        }
    )
    state = await ex.run_plan_loop("goal", llm, {"w": worker})
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert "CANARY" not in llm.prompts(), "tool history leaked into a prompt"
    assert "CANARY" not in state.model_dump_json(), "tool history leaked into global state"


async def test_max_steps_stops_a_runaway_replanner(ex):
    counter = iter(range(1000))
    llm = ScriptedLLM(
        lambda key, prompt: Plan(steps=[step("a")])
        if key == "Plan"
        else ReplanDecision(remaining_steps=[step(f"new_{next(counter)}")])
    )
    with pytest.raises(ex.StepLimitExceeded):
        await ex.run_plan_loop("goal", llm, {"w": echo}, max_steps=3)


async def test_replanner_cannot_rerun_a_finished_step(ex):
    ran = []

    async def worker(step, scratchpad):
        ran.append(step.id)
        return "ok"

    llm = ScriptedLLM(
        {"Plan": [Plan(steps=[step("a")])], "ReplanDecision": [ReplanDecision(remaining_steps=[step("a")])]}
    )
    state = await ex.run_plan_loop("goal", llm, {"w": worker})
    assert ran == ["a"]
    assert [s.id for s in state.completed] == ["a"]


async def test_llm_cannot_mark_a_step_finished(ex):
    """A reply carrying status and result must not skip the worker or plant a result."""
    reply = {"steps": [{"id": "a", "description": "d", "assigned_worker": "w", "status": "completed", "result": "FAKE"}]}
    llm = ScriptedLLM({"Plan": [reply], "ReplanDecision": [ReplanDecision()]})
    state = await ex.run_plan_loop("goal", llm, {"w": echo})
    assert state.completed[0].result == "result of a"
