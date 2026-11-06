import pytest
from langgraph.errors import GraphRecursionError

from planagents.fakes import ScriptedLLM
from planagents.schemas import Plan, ReplanDecision, Step


def step(id: str) -> Step:
    return Step(id=id, description=f"do {id}", assigned_worker="w")


async def echo(step):
    return f"result of {step.id}"


def by_id(result):
    return {s.id: s for s in result["steps"]}


@pytest.fixture
def ex(load):
    return load("ex3_langgraph_loop")


async def test_steps_complete_in_place_without_duplicates(ex):
    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a"), step("b")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("b")]), ReplanDecision()],
        }
    )
    result = await ex.build_graph(llm, {"w": echo}).ainvoke({"objective": "goal"})
    assert [s.id for s in result["steps"]] == ["a", "b"]
    assert all(s.status == "completed" for s in result["steps"])
    assert by_id(result)["b"].result == "result of b"


async def test_replanner_can_add_a_step(ex):
    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("extra")]), ReplanDecision()],
        }
    )
    result = await ex.build_graph(llm, {"w": echo}).ainvoke({"objective": "goal"})
    assert by_id(result)["extra"].status == "completed"


async def test_steps_the_replanner_drops_are_cancelled_not_run(ex):
    ran = []

    async def worker(step):
        ran.append(step.id)
        return "ok"

    llm = ScriptedLLM(
        {"Plan": [Plan(steps=[step("a"), step("b")])], "ReplanDecision": [ReplanDecision()]}
    )
    result = await ex.build_graph(llm, {"w": worker}).ainvoke({"objective": "goal"})
    assert ran == ["a"]
    assert by_id(result)["b"].status == "cancelled"


async def test_replanner_cannot_rerun_or_wipe_a_finished_step(ex):
    ran = []

    async def worker(step):
        ran.append(step.id)
        return "real result"

    llm = ScriptedLLM(
        {
            "Plan": [Plan(steps=[step("a")])],
            "ReplanDecision": [ReplanDecision(remaining_steps=[step("a")]), ReplanDecision()],
        }
    )
    result = await ex.build_graph(llm, {"w": worker}).ainvoke({"objective": "goal"})
    assert ran == ["a"]
    assert by_id(result)["a"].result == "real result"


async def test_recursion_limit_stops_a_runaway_replanner(ex):
    counter = iter(range(1000))

    def handler(key, prompt):
        if key == "Plan":
            return Plan(steps=[step("s0")])
        return ReplanDecision(remaining_steps=[step(f"s{next(counter) + 1}")])

    with pytest.raises(GraphRecursionError):
        await ex.build_graph(ScriptedLLM(handler), {"w": echo}, recursion_limit=9).ainvoke({"objective": "goal"})


async def test_replanner_sees_results(ex):
    llm = ScriptedLLM({"Plan": [Plan(steps=[step("a")])], "ReplanDecision": [ReplanDecision()]})
    await ex.build_graph(llm, {"w": echo}).ainvoke({"objective": "goal"})
    assert "result of a" in llm.calls_for("ReplanDecision")[0].prompt
