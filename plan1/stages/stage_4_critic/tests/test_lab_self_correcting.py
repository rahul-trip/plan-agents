import os

import pytest
from dotenv import load_dotenv

from planagents.fakes import ScriptedLLM
from planagents.llm import get_llm
from planagents.schemas import ExecutionPlan, TaskNode

CONFIG = {"recursion_limit": 60}
SOURCE = "Revenue grew 12% in Q3."


def task(id, worker, *deps):
    return TaskNode(id=id, worker=worker, instruction=id, depends_on=list(deps))


def primary_plan():
    return ExecutionPlan(tasks=[task("fetch", "primary_api"), task("report", "reporter", "fetch")])


def fallback_plan(suffix="v2"):
    return ExecutionPlan(
        tasks=[task(f"fetch_{suffix}", "fallback_api"), task(f"report_{suffix}", "reporter", f"fetch_{suffix}")]
    )


@pytest.fixture
def lab(load):
    return load("lab_self_correcting")


async def test_transient_blip_recovers_locally_with_no_planner_call(lab):
    """Tier 1 inside the capstone: this used to go straight to a replan."""
    api = lab.MockAPI(flaky={"primary_api": 1})
    llm = ScriptedLLM({"PlanDraft": [primary_plan()]})
    result = await lab.build_pipeline(llm, lab.make_fault_worker(api)).ainvoke({"objective": "Get the data"}, CONFIG)
    assert api.calls == ["primary_api", "primary_api"]
    assert result["replan_count"] == 0
    assert len(llm.calls) == 1
    assert result["tasks"]["report"].result == "report: data from primary_api"


async def test_persistent_outage_trips_the_breaker_then_replans(lab):
    """Tier 1 gives up early because the breaker opened, then tier 2 reroutes."""
    api = lab.MockAPI(down={"primary_api"})
    llm = ScriptedLLM({"PlanDraft": [primary_plan(), fallback_plan()]})
    worker = lab.make_fault_worker(api, failure_threshold=2)
    result = await lab.build_pipeline(llm, worker, max_attempts=3).ainvoke({"objective": "Get the data"}, CONFIG)
    assert api.calls.count("primary_api") == 2, "the third attempt must be refused by the open breaker"
    assert result["replan_count"] == 1
    assert result["tasks"]["report_v2"].status == "completed"


async def test_planner_sees_the_tools_and_cannot_invent_one(lab):
    """Regression: on a real model the planner invented workers, and no API was ever called."""
    api = lab.MockAPI()
    invented = ExecutionPlan(tasks=[task("x", "data_fetcher")])
    llm = ScriptedLLM({"PlanDraft": [invented, primary_plan()]})
    result = await lab.build_pipeline(llm, lab.make_fault_worker(api)).ainvoke({"objective": "Get the data"}, CONFIG)
    assert "fallback_api" in llm.calls[0].prompt
    assert "data_fetcher" in llm.calls[1].prompt, "the rejection names the invented worker"
    assert "x" not in result["tasks"]
    assert api.calls == ["primary_api"]


async def test_report_with_nothing_to_report_is_a_failure(lab):
    worker = lab.make_fault_worker(lab.MockAPI())
    with pytest.raises(ValueError):
        await worker(task("r", "reporter"), {})


async def test_blank_output_is_rejected_by_the_evaluator(lab):
    outputs = iter(["   ", SOURCE])

    async def worker(task, upstream):
        return next(outputs)

    llm = ScriptedLLM(
        {"PlanDraft": [ExecutionPlan(tasks=[task("a", "reader")]), ExecutionPlan(tasks=[task("b", "reader")])]}
    )
    result = await lab.build_pipeline(llm, worker).ainvoke({"objective": "Read it"}, CONFIG)
    assert result["failed_plans"][0].kind == "verification"
    assert result["tasks"]["b"].result == SOURCE


@pytest.mark.live
async def test_live_model_reaches_the_real_tools(lab):
    load_dotenv()
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is not set")
    api = lab.MockAPI(blocked={"primary_api"})
    graph = lab.build_pipeline(get_llm(), lab.make_fault_worker(api))
    objective = "Fetch the customer data, preferring the primary endpoint, then write a report on it."
    await graph.ainvoke({"objective": objective}, CONFIG)
    assert api.calls, "the planner must have used a real tool"
    assert api.calls.count("primary_api") <= 1, "the blocked tool must not be called twice"


async def test_reroutes_around_the_403_using_the_fallback(lab):
    api = lab.MockAPI(blocked={"primary_api"})
    llm = ScriptedLLM({"PlanDraft": [primary_plan(), fallback_plan()]})
    graph = lab.build_pipeline(llm, lab.make_fault_worker(api))
    result = await graph.ainvoke({"objective": "Get the data"}, CONFIG)

    tasks = result["tasks"]
    assert tasks["fetch"].status == "cancelled"
    assert tasks["report_v2"].status == "completed"
    assert tasks["report_v2"].result == "report: data from fallback_api"
    assert api.calls == ["primary_api", "fallback_api"]
    assert result["failed_plans"][0].tool == "primary_api"
    assert "403" in result["failed_plans"][0].reason
    assert result["replan_count"] == 1
    assert "403" in llm.calls[1].prompt, "negative constraint reached the replanner"


async def test_replanner_that_repeats_the_blocked_path_is_rejected(lab):
    """An LLM ignoring its negative prompt must not reach the failing API a second time."""
    api = lab.MockAPI(blocked={"primary_api"})
    repeat = ExecutionPlan(tasks=[task("fetch_again", "primary_api")])
    llm = ScriptedLLM({"PlanDraft": [primary_plan(), repeat, fallback_plan()]})
    graph = lab.build_pipeline(llm, lab.make_fault_worker(api))
    result = await graph.ainvoke({"objective": "Get the data"}, CONFIG)

    assert api.calls.count("primary_api") == 1, "thrashing: the blocked API was called again"
    assert result["replan_count"] == 1
    assert result["tasks"]["report_v2"].status == "completed"


async def test_no_way_forward_ends_degraded_without_thrashing(lab):
    api = lab.MockAPI(blocked={"primary_api", "fallback_api"})
    llm = ScriptedLLM({"PlanDraft": [primary_plan(), fallback_plan()]})
    graph = lab.build_pipeline(llm, lab.make_fault_worker(api), max_replans=1)
    result = await graph.ainvoke({"objective": "Get the data"}, CONFIG)
    assert result["degraded"] is True
    assert result["replan_count"] == 1
    assert api.calls == ["primary_api", "fallback_api"]
    assert not any(t.status == "pending" for t in result["tasks"].values())


def test_groundedness_check(lab):
    claim = task("c", "claimer", "s")
    upstream = {"s": SOURCE}

    def with_result(result):
        return claim.model_copy(update={"result": result})

    assert lab.groundedness_check(with_result(lab.claim_json("Up 12%", "grew 12%")), upstream) is None
    assert "Hallucination" in lab.groundedness_check(with_result(lab.claim_json("Up 40%", "grew 40%")), upstream)
    assert "Hallucination" in lab.groundedness_check(with_result(lab.claim_json("Up", "  ")), upstream)
    assert lab.groundedness_check(with_result("not json"), upstream)
    assert lab.groundedness_check(task("x", "reader"), upstream) is None, "other workers pass"


async def test_catches_a_hallucination_and_retries_the_same_tool(lab):
    """No exception is raised here. Without the check, the fabricated claim would just ship."""
    outputs = iter([lab.claim_json("Up 40%", "grew 40% in Q3"), lab.claim_json("Up 12%", "grew 12% in Q3")])

    async def worker(task, upstream):
        return SOURCE if task.worker == "reader" else next(outputs)

    first = ExecutionPlan(tasks=[task("source", "reader"), task("claim", "claimer", "source")])
    # The same 'claimer' tool: a verification failure doesn't mean the tool is broken.
    corrected = ExecutionPlan(tasks=[task("claim_v2", "claimer", "source")])
    llm = ScriptedLLM({"PlanDraft": [first, corrected]})

    result = await lab.build_pipeline(llm, worker).ainvoke({"objective": "Report revenue"}, CONFIG)

    tasks = result["tasks"]
    assert tasks["claim"].status == "cancelled"
    assert "Hallucination" in result["failed_plans"][0].reason
    assert result["failed_plans"][0].kind == "verification"
    assert tasks["claim_v2"].status == "completed"
    assert "12%" in tasks["claim_v2"].result
    assert result["degraded"] is False


async def test_repeated_hallucination_ends_degraded_and_never_ships_as_verified(lab):
    async def worker(task, upstream):
        return SOURCE if task.worker == "reader" else lab.claim_json("Made up", "not in the source")

    first = ExecutionPlan(tasks=[task("source", "reader"), task("claim", "claimer", "source")])
    retries = [ExecutionPlan(tasks=[task(f"claim_{i}", "claimer", "source")]) for i in range(2)]
    llm = ScriptedLLM({"PlanDraft": [first, *retries]})

    result = await lab.build_pipeline(llm, worker).ainvoke({"objective": "Report revenue"}, CONFIG)

    assert result["degraded"] is True
    assert result["replan_count"] == 2
    claims = [t for t in result["tasks"].values() if t.worker == "claimer"]
    assert not any(t.status == "completed" for t in claims)
