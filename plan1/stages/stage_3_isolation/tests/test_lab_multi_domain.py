import sqlite3
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from planagents.dag import checkpoint_serde
from planagents.fakes import ScriptedLLM
from planagents.schemas import ExecutionPlan, TaskNode

CORPUS = {
    "refunds": "Refund policy: orders can be refunded within 30 days.",
    "shipping": "Shipping takes five days.",
}
TEST_CODE = "from solution import double\n\ndef test_double():\n    assert double(21) == 42\n"
GOOD_CODE = "def double(x):\n    return x * 2\n"
BAD_CODE = "def double(x):\n    return x + 2  # CANARY-INTERNAL-CODE\n"


def plan():
    return ExecutionPlan(
        tasks=[
            TaskNode(id="revenue", worker="sql", instruction="What is total revenue?"),
            TaskNode(id="policy", worker="search", instruction="What is the refund policy?"),
            TaskNode(
                id="script",
                worker="python",
                instruction="Write double(x).",
                depends_on=["revenue", "policy"],
                args={"test_code": TEST_CODE},
            ),
        ]
    )


def handler(key, prompt):
    """Scripted by content, because parallel workers interleave their LLM calls.

    It only looks for data each prompt must contain, never at your wording: the fix prompt has the
    failing code, the write prompt has the instruction, and the summary prompt has the rows.
    """
    if key == "SqlQuery":
        # Wrong first, so the analyst's retry history contains a canary.
        if "CANARY_BAD_COLUMN" in prompt:
            return {"sql": "SELECT SUM(total) FROM orders"}
        return {"sql": "SELECT CANARY_BAD_COLUMN FROM orders"}
    if "CANARY-INTERNAL-CODE" in prompt:
        return GOOD_CODE
    if "Write double" in prompt:
        return BAD_CODE
    if "42.0" in prompt:
        return "Revenue is 42."
    return "Refunds within 30 days."


@pytest.fixture
def lab(load):
    return load("lab_multi_domain")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE orders (id INTEGER, total REAL)")
    connection.execute("INSERT INTO orders VALUES (1, 42.0)")
    connection.commit()
    connection.close()
    return str(path)


def test_search_ranks_by_matching_words(lab):
    assert lab.search_corpus(CORPUS, "refund policy for orders")[0] == "refunds"
    assert lab.search_corpus(CORPUS, "quantum") == []
    assert lab.search_corpus(CORPUS, "policy days", top_k=1) == ["refunds"]


async def test_search_subgraph_reports_no_results_without_an_llm_call(lab):
    llm = ScriptedLLM({})
    output = await lab.build_search_subgraph(llm, CORPUS).ainvoke({"question": "quantum"})
    assert output["verified"] is False and not llm.calls


async def test_all_three_domains_complete_and_downstream_gets_summaries(lab, db):
    llm = ScriptedLLM(handler)
    graph = lab.build_parent_graph(llm, plan(), db, CORPUS)
    result = await graph.ainvoke({"objective": "demo"}, {"recursion_limit": 40})
    tasks = result["tasks"]
    assert all(t.status == "completed" for t in tasks.values())
    assert tasks["revenue"].result == "Revenue is 42."
    assert tasks["script"].result == GOOD_CODE
    write_prompt = next(c.prompt for c in llm.calls if "Write double" in c.prompt)
    assert "Revenue is 42." in write_prompt and "Refunds within 30 days." in write_prompt
    assert "CANARY_BAD_COLUMN" not in write_prompt, "another worker's internal retry leaked into this prompt"


async def test_zero_context_leakage_in_every_parent_update_and_checkpoint(lab, db):
    """The capstone check: scan everything the parent graph ever held."""
    saver = InMemorySaver(serde=checkpoint_serde())
    graph = lab.build_parent_graph(ScriptedLLM(handler), plan(), db, CORPUS, checkpointer=saver)
    config = {"configurable": {"thread_id": "leak"}, "recursion_limit": 40}

    updates = []
    async for chunk in graph.astream({"objective": "demo"}, config, stream_mode="updates"):
        updates.append(chunk)
    assert len(updates) > 4

    history = [snapshot async for snapshot in graph.aget_state_history(config)]
    assert len(history) > 4
    everything = repr(updates) + repr([snapshot.values for snapshot in history])

    for forbidden in ("CANARY", "messages", "test_output", "db_schema", "CREATE TABLE"):
        assert forbidden not in everything, f"{forbidden!r} leaked into the parent graph"
    for snapshot in history:
        assert set(snapshot.values) <= {"objective", "tasks", "final_output"}


async def test_worker_error_is_contained_instead_of_aborting_the_run(lab, db):
    """Regression: the failure mode Stage 2 teaches you to fix."""
    bad = ExecutionPlan(
        tasks=[
            TaskNode(id="policy", worker="search", instruction="What is the refund policy?"),
            TaskNode(id="oops", worker="not_a_worker", instruction="x"),
        ]
    )
    result = await lab.build_parent_graph(ScriptedLLM(handler), bad, db, CORPUS).ainvoke({"objective": "demo"}, {"recursion_limit": 40})
    assert result["tasks"]["oops"].status == "failed" and "not_a_worker" in result["tasks"]["oops"].result
    assert result["tasks"]["policy"].status == "completed", "the sibling's work must survive"


async def test_gotcha_a_key_outside_the_schema_is_dropped_silently():
    """Target-independent. Why a stray `messages` key from a worker doesn't leak, or raise an error.
    To break isolation on purpose, put the data in a field the schema has, such as a task's result."""

    class State(TypedDict, total=False):
        tasks: dict

    async def node(state):
        return {"tasks": {"a": 1}, "messages": ["internal history"]}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    assert await builder.compile().ainvoke({}) == {"tasks": {"a": 1}}


async def test_unverified_subgraph_fails_the_task_and_stops_downstream(lab, db):
    def always_bad_sql(key, prompt):
        return {"sql": "BAD SQL"} if key == "SqlQuery" else handler(key, prompt)

    graph = lab.build_parent_graph(ScriptedLLM(always_bad_sql), plan(), db, CORPUS)
    result = await graph.ainvoke({"objective": "demo"}, {"recursion_limit": 40})
    assert result["tasks"]["revenue"].status == "failed"
    assert result["tasks"]["script"].status == "pending"
    assert "failed" in result["final_output"]
