import uuid
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from planagents.dag import PlanValidationError
from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode, WorkerSpec

WORKERS = {"w": WorkerSpec(description="Safe"), "danger": WorkerSpec(description="Destructive", destructive=True)}


def task(id, *deps, destructive=False, args=None):
    return TaskNode(id=id, worker="danger" if destructive else "w", instruction=id, depends_on=list(deps), args=args or {})


def llm_for(*tasks):
    return ScriptedLLM({"PlanDraft": [ExecutionPlan(tasks=list(tasks))]})


def config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}


class Recorder:
    """A worker recording its calls and the args it ran with."""

    def __init__(self):
        self.calls = {}

    async def __call__(self, task, upstream):
        self.calls.setdefault(task.id, []).append(task.args)
        return f"out-{task.id}"


@pytest.fixture
def ex(load):
    return load("ex3_hitl")


def tasks_map(*tasks):
    return {t.id: t for t in tasks}


def test_apply_review_approve_changes_nothing(ex):
    assert ex.apply_review(tasks_map(task("a")), {"action": "approve"}) == {}


def test_apply_review_edit_and_prune_with_dependents(ex):
    tasks = tasks_map(task("a"), task("b"), task("c", "b"))
    delta = ex.apply_review(tasks, {"action": "edit", "edits": {"a": {"args": {"n": 2}}}, "prune": ["b"]})
    assert delta["a"].args == {"n": 2}
    assert delta["b"].status == "cancelled"
    assert delta["c"].status == "cancelled", "a dependent left pending would deadlock"


def test_apply_review_rejects_typos(ex):
    tasks = tasks_map(task("a"))
    with pytest.raises(PlanValidationError):
        ex.apply_review(tasks, {"action": "edit", "prune": ["typo"]})
    with pytest.raises(PlanValidationError):
        ex.apply_review(tasks, {"action": "edit", "edits": {"a": {"status": "completed"}}})
    with pytest.raises(PlanValidationError):
        ex.apply_review(tasks, {"action": "maybe"})
    with pytest.raises(PlanValidationError):
        ex.apply_review(tasks, "approve")
    with pytest.raises(PlanValidationError):
        ex.apply_review(tasks, {"action": "edit", "edits": {"a": {"args": "not a dict"}}})


@pytest.mark.parametrize(
    "bad", [{"action": "approve"}, "yes", {"approved": "yes"}, {"approved": True, "args": "x"}]
)
def test_parse_gate_decision_rejects_malformed_answers(ex, bad):
    with pytest.raises(PlanValidationError):
        ex.parse_gate_decision(bad)


def test_parse_gate_decision_accepts_valid_answers(ex):
    assert ex.parse_gate_decision({"approved": False}) == {"approved": False}
    decision = {"approved": True, "args": {"n": 1}}
    assert ex.parse_gate_decision(decision) == decision


@pytest.mark.parametrize("bad", [{"approved": True}, {"action": "edit", "prune": ["typo"]}, "approve"])
async def test_bad_review_decision_asks_again_instead_of_locking_the_thread(ex, saver, bad):
    """Regression: the first bad decision replayed on every resume, so no later one could get through."""
    worker = Recorder()
    graph = ex.build_graph(llm_for(task("a")), worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)

    again = await graph.ainvoke(Command(resume=bad), cfg)
    payload = again["__interrupt__"][0].value
    assert payload["type"] == "plan_review"
    assert payload["error"], "the reviewer must be told what was wrong"
    assert worker.calls == {}

    result = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert result["tasks"]["a"].status == "completed"


async def test_bad_gate_decision_asks_again_and_never_counts_as_rejection(ex, saver):
    """Regression: {"action": "approve"} at the gate silently cancelled the task."""
    worker = Recorder()
    graph = ex.build_graph(llm_for(task("wipe", destructive=True)), worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    await graph.ainvoke(Command(resume={"action": "approve"}), cfg)

    again = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    gate = again["__interrupt__"][0].value
    assert gate["task_id"] == "wipe" and gate["error"]
    assert worker.calls == {}

    result = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert result["tasks"]["wipe"].status == "completed"


async def test_registry_decides_what_is_destructive_not_the_llm(ex, saver):
    """Regression: the gate read a flag the LLM filled in, so not flagging a task skipped it."""
    worker = Recorder()
    reply = {
        "tasks": [
            {"id": "wipe", "worker": "danger", "instruction": "x", "depends_on": [], "args_json": "{}",
             "destructive": False}
        ]
    }  # fmt: skip
    graph = ex.build_graph(ScriptedLLM({"PlanDraft": [reply]}), worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    paused = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert paused["__interrupt__"][0].value["task_id"] == "wipe"
    assert worker.calls == {}


async def test_reviewer_can_gate_a_task_the_registry_missed(ex, saver):
    worker = Recorder()
    graph = ex.build_graph(llm_for(task("risky")), worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    decision = {"action": "edit", "edits": {"risky": {"destructive": True}}}
    paused = await graph.ainvoke(Command(resume=decision), cfg)
    assert paused["__interrupt__"][0].value["task_id"] == "risky"
    assert worker.calls == {}


async def test_planner_is_told_the_workers_and_cannot_invent_one(ex, saver):
    invented = TaskNode(id="x", worker="invented", instruction="x")
    llm = ScriptedLLM({"PlanDraft": [ExecutionPlan(tasks=[invented]), ExecutionPlan(tasks=[task("a")])]})
    graph = ex.build_graph(llm, Recorder(), saver, workers=WORKERS)
    result = await graph.ainvoke({"objective": "goal"}, config())
    assert "Destructive" in llm.calls[0].prompt
    assert [t["id"] for t in result["__interrupt__"][0].value["tasks"]] == ["a"]


async def test_pauses_for_review_and_exposes_the_plan_as_json(ex, saver):
    worker = Recorder()
    graph = ex.build_graph(llm_for(task("a")), worker, saver, workers=WORKERS)
    result = await graph.ainvoke({"objective": "goal"}, config())
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "plan_review"
    assert payload["tasks"][0]["id"] == "a"
    assert worker.calls == {}, "nothing runs before approval"


async def test_edit_and_prune_then_resume_to_completion(ex, saver):
    worker = Recorder()
    llm = llm_for(task("a", args={"limit": 1}), task("b"), task("c", "b"))
    graph = ex.build_graph(llm, worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    decision = {"action": "edit", "edits": {"a": {"args": {"limit": 99}}}, "prune": ["b"]}
    result = await graph.ainvoke(Command(resume=decision), cfg)

    assert worker.calls == {"a": [{"limit": 99}]}
    assert result["tasks"]["a"].status == "completed"
    assert result["tasks"]["c"].status == "cancelled"
    assert "final_output" in result
    assert len(llm.calls) == 1, "resuming must not call the planner LLM again"


async def test_destructive_gate_lets_a_human_change_args_before_it_runs(ex, saver):
    worker = Recorder()
    llm = llm_for(task("backup"), task("wipe", "backup", destructive=True, args={"table": "prod"}))
    graph = ex.build_graph(llm, worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    paused = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)

    gate = paused["__interrupt__"][0].value
    assert (gate["type"], gate["task_id"]) == ("destructive_action", "wipe")
    assert worker.calls == {"backup": [{}]}, "the destructive task has not run yet"

    state = await graph.aget_state(cfg)
    assert state.values["tasks"]["wipe"].args == {"table": "prod"}, "state can be inspected mid-flight"

    result = await graph.ainvoke(Command(resume={"approved": True, "args": {"table": "staging"}}), cfg)
    assert worker.calls["wipe"] == [{"table": "staging"}]
    assert result["tasks"]["wipe"].args == {"table": "staging"}
    assert result["tasks"]["wipe"].status == "completed"


async def test_rejecting_a_destructive_action_cancels_it_and_its_dependents(ex, saver):
    worker = Recorder()
    llm = llm_for(task("wipe", destructive=True), task("report", "wipe"))
    graph = ex.build_graph(llm, worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    result = await graph.ainvoke(Command(resume={"approved": False}), cfg)
    assert worker.calls == {}
    assert result["tasks"]["wipe"].status == "cancelled"
    assert result["tasks"]["report"].status == "cancelled"
    assert "final_output" in result


async def test_parallel_gates_resume_together_and_siblings_do_not_rerun(ex, saver):
    worker = Recorder()
    llm = llm_for(task("drop_a", destructive=True), task("drop_b", destructive=True), task("safe"))
    graph = ex.build_graph(llm, worker, saver, workers=WORKERS)
    cfg = config()
    await graph.ainvoke({"objective": "goal"}, cfg)
    paused = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)

    interrupts = paused["__interrupt__"]
    assert {i.value["task_id"] for i in interrupts} == {"drop_a", "drop_b"}
    resume_map = {i.id: {"approved": i.value["task_id"] == "drop_a"} for i in interrupts}
    result = await graph.ainvoke(Command(resume=resume_map), cfg)

    assert result["tasks"]["drop_a"].status == "completed"
    assert result["tasks"]["drop_b"].status == "cancelled"
    assert len(worker.calls["safe"]) == 1, "a finished sibling must not re-run on resume"


class _Counter(TypedDict, total=False):
    answer: str


def _single_node_graph(node):
    builder = StateGraph(_Counter)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=InMemorySaver())


async def test_gotcha_raising_after_an_interrupt_locks_the_thread():
    """Target-independent. Why plan_review must ask again rather than raise."""
    seen = []

    def node(state):
        answer = interrupt("approve?")
        seen.append(answer)
        if answer != "good":
            raise ValueError(f"bad answer: {answer}")
        return {"answer": answer}

    graph = _single_node_graph(node)
    cfg = config()
    await graph.ainvoke({}, cfg)
    for _ in range(2):
        with pytest.raises(ValueError, match="bad answer: typo"):
            await graph.ainvoke(Command(resume="typo" if not seen else "good"), cfg)
    assert seen == ["typo", "typo"], "the corrected answer never reached the node"


async def test_gotcha_a_broad_except_around_interrupt_stops_the_pause():
    """Target-independent. interrupt() pauses by raising, so `except Exception` swallows the pause."""

    def node(state):
        try:
            interrupt("approve?")
        except Exception:
            return {"answer": "never asked"}
        return {"answer": "asked"}

    result = await _single_node_graph(node).ainvoke({}, config())
    assert result == {"answer": "never asked"}


async def test_gotcha_code_before_interrupt_runs_again_on_resume():
    """Target-independent. Why side effects belong after the interrupt or in another node."""
    runs = {"before": 0, "after": 0}

    def node(state):
        runs["before"] += 1
        answer = interrupt("approve?")
        runs["after"] += 1
        return {"answer": answer}

    builder = StateGraph(_Counter)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    cfg = config()
    await graph.ainvoke({}, cfg)
    await graph.ainvoke(Command(resume="yes"), cfg)
    assert runs == {"before": 2, "after": 1}
