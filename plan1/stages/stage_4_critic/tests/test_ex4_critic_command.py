import pytest

from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode, WorkerSpec

CONFIG = {"recursion_limit": 40}


def task(id, worker="good", *deps):
    return TaskNode(id=id, worker=worker, instruction=id, depends_on=list(deps))


def plan(*tasks):
    return ExecutionPlan(tasks=list(tasks))


async def worker(task, upstream):
    if task.worker == "bad":
        raise PermissionError("403 Forbidden")
    return f"out-{task.id}"


@pytest.fixture
def ex(load):
    return load("ex4_critic_command")


async def test_happy_path_needs_no_replan(ex):
    llm = ScriptedLLM({"PlanDraft": [plan(task("a"), task("b", "good", "a"))]})
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["replan_count"] == 0
    assert result["degraded"] is False
    assert all(t.status == "completed" for t in result["tasks"].values())


async def test_failure_records_the_path_and_replans_around_it(ex):
    llm = ScriptedLLM(
        {
            "PlanDraft": [
                plan(task("fetch", "bad"), task("report", "good", "fetch")),
                plan(task("fetch_v2", "good"), task("report_v2", "good", "fetch_v2")),
            ]
        }
    )
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    tasks = result["tasks"]
    assert result["replan_count"] == 1
    assert tasks["fetch"].status == "cancelled" and tasks["report"].status == "cancelled"
    assert tasks["report_v2"].status == "completed"
    assert [(a.task_id, a.tool) for a in result["failed_plans"]] == [("fetch", "bad")]
    assert "403 Forbidden" in result["failed_plans"][0].reason
    assert "403 Forbidden" in llm.calls[1].prompt, "negative constraints reach the replanner"


async def test_replan_keeps_finished_work(ex):
    calls = []

    async def counting(task, upstream):
        calls.append(task.id)
        return await worker(task, upstream)

    llm = ScriptedLLM(
        {
            "PlanDraft": [
                plan(task("keep"), task("fetch", "bad", "keep")),
                plan(task("fetch_v2", "good", "keep")),
            ]
        }
    )
    result = await ex.build_graph(llm, counting).ainvoke({"objective": "goal"}, CONFIG)
    assert calls.count("keep") == 1, "completed work must not re-run"
    assert result["tasks"]["fetch_v2"].status == "completed"


async def test_budget_exhaustion_degrades_instead_of_looping(ex):
    llm = ScriptedLLM({"PlanDraft": [plan(task(f"t{i}", "bad")) for i in range(5)]})
    result = await ex.build_graph(llm, worker, max_replans=2).ainvoke({"objective": "goal"}, CONFIG)
    assert result["replan_count"] == 2
    assert result["degraded"] is True
    assert "DEGRADED" in result["final_output"]
    assert len(llm.calls) == 3, "the initial plan plus exactly 2 replans"


async def test_exhaustion_cancels_pending_work(ex):
    llm = ScriptedLLM(
        {"PlanDraft": [plan(task("root", "bad"), task("child", "good", "root"))]}
    )
    result = await ex.build_graph(llm, worker, max_replans=0).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["child"].status == "cancelled"


async def test_parallel_failures_trigger_only_one_replan(ex):
    """One critic after fan-in, so two failures in one superstep can't race."""
    llm = ScriptedLLM(
        {"PlanDraft": [plan(task("x", "bad"), task("y", "bad")), plan(task("z"))]}
    )
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["replan_count"] == 1
    assert {a.task_id for a in result["failed_plans"]} == {"x", "y"}


async def test_critic_check_rejects_a_completed_result(ex):
    def check(task, upstream):
        return "not grounded" if task.id == "claim" else None

    llm = ScriptedLLM({"PlanDraft": [plan(task("claim")), plan(task("claim_v2"))]})
    result = await ex.build_graph(llm, worker, check=check).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["claim"].status == "cancelled"
    assert result["tasks"]["claim_v2"].status == "completed"
    assert result["failed_plans"][0].reason == "not grounded"
    assert result["failed_plans"][0].kind == "verification"


def test_replan_problem_rules(ex):
    done = task("done").model_copy(update={"status": "completed"})
    dead = task("dead").model_copy(update={"status": "cancelled"})
    tasks = {"done": done, "dead": dead}
    assert ex.replan_problem(plan(task("new", "good", "done")), tasks) is None
    assert "done" in ex.replan_problem(plan(task("done")), tasks), "reused ID"
    assert "dead" in ex.replan_problem(plan(task("new", "good", "dead")), tasks), "cancelled parent"


async def test_invalid_replan_is_rejected_and_reasked_instead_of_crashing(ex):
    llm = ScriptedLLM(
        {
            "PlanDraft": [
                plan(task("fetch", "bad")),
                plan(task("v2", "good", "ghost")),
                plan(task("fetch", "good")),
                plan(task("v3", "good")),
            ]
        }
    )
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["v3"].status == "completed"
    assert result["replan_count"] == 1, "rejected attempts aren't separate replans"
    assert "ghost" in llm.calls[2].prompt, "the rejection's reason must be appended to the prompt"


async def test_empty_replan_after_a_failure_ends_degraded(ex):
    """Regression: giving up with an empty plan reported the unrecovered run as not degraded."""
    llm = ScriptedLLM({"PlanDraft": [plan(task("only", "bad")), plan()]})
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["degraded"] is True
    assert result["final_output"].startswith("DEGRADED")


async def test_replanner_cannot_plant_a_finished_task(ex):
    """Regression: a replan carrying status and result shipped a result nothing ever produced."""
    fake = {
        "tasks": [
            {"id": "b", "worker": "good", "instruction": "b", "depends_on": [], "args_json": "{}",
             "status": "completed", "result": "HALLUCINATED"}
        ]
    }  # fmt: skip
    llm = ScriptedLLM({"PlanDraft": [plan(task("a", "bad")), fake]})
    result = await ex.build_graph(llm, worker).ainvoke({"objective": "goal"}, CONFIG)
    assert result["tasks"]["b"].result == "out-b"


async def test_replanner_prompt_shows_what_it_is_replacing_and_the_workers_available(ex):
    workers = {"good": WorkerSpec(description="Does good work"), "bad": WorkerSpec(description="Fails")}
    llm = ScriptedLLM(
        {"PlanDraft": [plan(task("fails", "bad"), task("REMAINING", "good", "fails")), plan(task("n"))]}
    )
    await ex.build_graph(llm, worker, workers=workers).ainvoke({"objective": "goal"}, CONFIG)
    assert "Does good work" in llm.calls[0].prompt, "the planner must see the tools"
    assert "Does good work" in llm.calls[1].prompt, "and so must the replanner"
    assert "Fails" not in llm.calls[1].prompt, "a broken tool must not be offered as available"
    assert "REMAINING" in llm.calls[1].prompt


async def test_replan_with_an_unknown_worker_is_rejected(ex):
    workers = {"good": WorkerSpec(description="x"), "bad": WorkerSpec(description="y")}
    llm = ScriptedLLM({"PlanDraft": [plan(task("a", "bad")), plan(task("b", "invented")), plan(task("c"))]})
    result = await ex.build_graph(llm, worker, workers=workers).ainvoke({"objective": "goal"}, CONFIG)
    assert "b" not in result["tasks"]
    assert result["tasks"]["c"].status == "completed"


async def test_plan_guard_rejections_are_bounded_and_end_degraded(ex):
    llm = ScriptedLLM(
        lambda key, prompt: plan(task(f"try_{len(prompt)}", "bad"))
    )

    def guard(candidate, failed_plans):
        return "reuses a failed tool" if failed_plans else None

    graph = ex.build_graph(llm, worker, plan_guard=guard, max_guard_rejections=2)
    result = await graph.ainvoke({"objective": "goal"}, CONFIG)
    assert result["degraded"] is True
    assert len(llm.calls) == 4, "the initial plan, one try and 2 rejected retries"
    assert not any(t.status == "pending" for t in result["tasks"].values())
