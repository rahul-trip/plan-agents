import uuid
from typing import Annotated, Literal, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from planagents.schemas import ExecutionPlan, TaskNode

PLAN = ExecutionPlan(
    tasks=[
        TaskNode(id="a", worker="w", instruction="a"),
        TaskNode(id="b", worker="w", instruction="b", depends_on=["a"], args={"ok": False}),
        TaskNode(id="c", worker="w", instruction="c", depends_on=["b"]),
    ]
)


class Recorder:
    """Fails any task whose args say ok=False, and counts calls."""

    def __init__(self):
        self.calls = []

    async def __call__(self, task, upstream):
        self.calls.append(task.id)
        if task.args.get("ok") is False:
            raise ValueError("bad argument")
        return f"out-{task.id}"


def config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}


@pytest.fixture
def ex(load):
    return load("ex4_time_travel")


@pytest.fixture
async def failed_run(ex, saver):
    worker = Recorder()
    graph = ex.build_graph(PLAN, worker, saver)
    cfg = config()
    result = await graph.ainvoke({}, cfg)
    assert result["tasks"]["b"].status == "failed"
    assert worker.calls == ["a", "b"]
    return graph, cfg, worker


async def test_finds_the_checkpoint_right_before_the_failed_worker(ex, failed_run):
    graph, cfg, _ = failed_run
    snapshot = await ex.find_before_failure(graph, cfg, "b")
    assert snapshot.values["tasks"]["a"].status == "completed"
    assert snapshot.values["tasks"]["b"].status == "pending"
    assert snapshot.next == ("worker",)


async def test_fork_applies_the_fix_without_redoing_earlier_work(ex, failed_run):
    graph, cfg, worker = failed_run
    result = await ex.fix_and_resume(graph, cfg, "b", {"ok": True})
    assert all(t.status == "completed" for t in result["tasks"].values())
    assert result["tasks"]["b"].args == {"ok": True}
    assert worker.calls == ["a", "b", "b", "c"], "a must not re-run; b runs again with the fix"


async def test_original_failed_history_survives_the_fork(ex, failed_run):
    graph, cfg, _ = failed_run
    failed_head = await graph.aget_state(cfg)
    await ex.fix_and_resume(graph, cfg, "b", {"ok": True})

    assert (await graph.aget_state(failed_head.config)).values["tasks"]["b"].status == "failed"
    head = await graph.aget_state(cfg)
    assert head.values["tasks"]["b"].status == "completed", "the thread's head is now the fork"
    statuses = [s.values["tasks"]["b"].status async for s in graph.aget_state_history(cfg) if s.values.get("tasks")]
    assert "completed" in statuses


async def test_replay_reruns_later_nodes_without_any_patch(ex, failed_run):
    graph, cfg, worker = failed_run
    snapshot = await ex.find_before_failure(graph, cfg, "b")
    result = await graph.ainvoke(None, snapshot.config)
    assert result["tasks"]["b"].status == "failed", "same input, same failure"
    assert worker.calls == ["a", "b", "b"], "replay really does call the worker again"


async def test_fork_reruns_a_parallel_sibling_that_had_already_succeeded(ex, saver):
    """The checkpoint before a failed superstep predates its siblings too, so they run again.
    Another reason side effects need idempotency keys."""
    worker = Recorder()
    plan = ExecutionPlan(
        tasks=[
            TaskNode(id="sibling", worker="w", instruction="s"),
            TaskNode(id="b", worker="w", instruction="b", args={"ok": False}),
        ]
    )
    graph = ex.build_graph(plan, worker, saver)
    cfg = config()
    await graph.ainvoke({}, cfg)
    worker.calls.clear()
    await ex.fix_and_resume(graph, cfg, "b", {"ok": True})
    assert sorted(worker.calls) == ["b", "sibling"]


async def test_gotcha_a_default_subgraph_has_no_checkpoints_to_travel_to():
    """Target-independent. To the parent, a default subgraph is a single step."""

    class Inner(TypedDict, total=False):
        value: str

    async def first(state):
        return {"value": "first"}

    async def second(state):
        return {"value": "second"}

    inner = StateGraph(Inner)
    inner.add_node("inner_first", first)
    inner.add_node("inner_second", second)
    inner.add_edge(START, "inner_first")
    inner.add_edge("inner_first", "inner_second")
    inner.add_edge("inner_second", END)

    outer = StateGraph(Inner)
    outer.add_node("sub", inner.compile())
    outer.add_edge(START, "sub")
    outer.add_edge("sub", END)
    graph = outer.compile(checkpointer=InMemorySaver())
    cfg = config()
    await graph.ainvoke({"value": "start"}, cfg)

    steps = [s.next async for s in graph.aget_state_history(cfg)]
    assert ("sub",) in steps
    assert not any("inner_first" in s or "inner_second" in s for s in steps)


def _merge(old, new):
    return {**(old or {}), **new}


class _State(TypedDict, total=False):
    values: Annotated[dict, _merge]


async def test_gotcha_fork_without_as_node_can_silently_do_nothing():
    """Target-independent. In a Command-routed graph, LangGraph infers as_node as the node with no
    outgoing edges, so the fork has nothing to run, and the resume ends with no error."""
    ran = []

    async def router(state) -> Command[Literal["work", "__end__"]]:
        return Command(goto="work" if "done" not in state["values"] else END)

    async def work(state):
        ran.append(state["values"].get("arg"))
        return {"values": {"done": True}}

    builder = StateGraph(_State)
    builder.add_node("router", router)
    builder.add_node("work", work)
    builder.add_edge(START, "router")
    builder.add_edge("work", "router")
    graph = builder.compile(checkpointer=InMemorySaver())
    cfg = config()
    await graph.ainvoke({"values": {"arg": "original"}}, cfg)
    assert ran == ["original"]

    before_work = [s async for s in graph.aget_state_history(cfg) if s.next == ("work",)][0]

    silent = await graph.aupdate_state(before_work.config, {"values": {"arg": "patched"}})
    assert (await graph.aget_state(silent)).next == ()
    await graph.ainvoke(None, silent)
    assert ran == ["original"], "the patched fork never ran"

    explicit = await graph.aupdate_state(before_work.config, {"values": {"arg": "patched"}}, as_node="work")
    assert (await graph.aget_state(explicit)).next == ("router",)
    await graph.ainvoke(None, explicit)
    assert ran == ["original", "patched"]
