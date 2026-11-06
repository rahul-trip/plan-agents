import sqlite3

import pytest

from planagents.fakes import ScriptedLLM

GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"
TEST_CODE = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
CODE_INPUT = {"instruction": "Write add(a, b)", "test_code": TEST_CODE}
CONFIG = {"recursion_limit": 25}


@pytest.fixture
def ex(load):
    return load("ex4_subgraphs")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE orders (id INTEGER, total REAL)")
    connection.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 10.0), (2, 32.0)])
    connection.commit()
    connection.close()
    return str(path)


async def test_code_generator_passes_first_time(ex):
    llm = ScriptedLLM({"text": [GOOD]})
    output = await ex.build_code_generator_subgraph(llm).ainvoke(CODE_INPUT, CONFIG)
    assert output["verified"] is True
    assert output["artifact"] == GOOD
    assert len(llm.calls) == 1


async def test_fix_loop_runs_real_pytest_and_feeds_failures_back(ex):
    llm = ScriptedLLM({"text": [BAD, f"```python\n{GOOD}```"]})
    output = await ex.build_code_generator_subgraph(llm).ainvoke(CODE_INPUT, CONFIG)
    assert output["verified"] is True
    assert output["artifact"] == GOOD
    assert "assert" in llm.calls[1].prompt, "the fix prompt must include pytest's output"


async def test_gives_up_unverified_after_max_attempts(ex):
    llm = ScriptedLLM({"text": [BAD, BAD]})
    output = await ex.build_code_generator_subgraph(llm, max_attempts=2).ainvoke(CODE_INPUT, CONFIG)
    assert output["verified"] is False
    assert len(llm.calls) == 2


async def test_output_schema_hides_the_internal_history(ex):
    llm = ScriptedLLM({"text": [BAD, GOOD]})
    output = await ex.build_code_generator_subgraph(llm).ainvoke(CODE_INPUT, CONFIG)
    assert set(output) == {"artifact", "verified"}


async def test_analyst_inspects_queries_and_summarizes(ex, db):
    llm = ScriptedLLM({"SqlQuery": [{"sql": "SELECT SUM(total) FROM orders"}], "text": ["Revenue is 42."]})
    graph = ex.build_data_analyst_subgraph(llm)
    output = await graph.ainvoke({"question": "Total revenue?", "db_path": db}, CONFIG)
    assert output == {"summary": "Revenue is 42.", "verified": True}
    assert "CREATE TABLE orders" in llm.calls_for("SqlQuery")[0].prompt
    assert "42.0" in llm.calls_for("text")[0].prompt


async def test_analyst_retries_bad_sql_with_the_error(ex, db):
    llm = ScriptedLLM(
        {
            "SqlQuery": [{"sql": "SELECT nope FROM orders"}, {"sql": "SELECT COUNT(*) FROM orders"}],
            "text": ["Two orders."],
        }
    )
    output = await ex.build_data_analyst_subgraph(llm).ainvoke({"question": "How many?", "db_path": db}, CONFIG)
    assert output["verified"] is True
    assert "nope" in llm.calls_for("SqlQuery")[1].prompt


async def test_analyst_gives_up_without_a_summary_call(ex, db):
    llm = ScriptedLLM({"SqlQuery": [{"sql": "BAD SQL"}, {"sql": "BAD SQL"}]})
    graph = ex.build_data_analyst_subgraph(llm, max_attempts=2)
    output = await graph.ainvoke({"question": "?", "db_path": db}, CONFIG)
    assert output["verified"] is False
    assert not llm.calls_for("text")


def test_writes_fail_at_the_database_level(ex, db):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ex.run_readonly_query(db, "DROP TABLE orders")
    assert ex.run_readonly_query(db, "SELECT COUNT(*) FROM orders") == [(2,)]


@pytest.mark.parametrize("name", ["with space.db", "data#1.db", "data?1.db", "100%.db"])
def test_read_only_holds_for_unusual_file_names(ex, tmp_path, name):
    """Regression: a # or ? in the path dropped mode=ro, and the write went to a different file."""
    path = tmp_path / name
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (x)")
    connection.commit()
    connection.close()
    assert ex.run_readonly_query(str(path), "SELECT COUNT(*) FROM t") == [(0,)]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ex.run_readonly_query(str(path), "CREATE TABLE evil (x)")
    assert [p.name for p in tmp_path.iterdir()] == [name], "no stray database file was created"
