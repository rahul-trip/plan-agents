"""Smoke test with the real LLM named by LLM_MODEL. Run with: pytest -m live --target=solutions"""

import os

import pytest
from dotenv import load_dotenv

from planagents.llm import get_llm


@pytest.mark.live
async def test_real_llm_plans_a_valid_dag_that_runs(load):
    load_dotenv()
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is not set")

    async def worker(task, upstream):
        return f"done: {task.instruction}"

    graph = load("ex3_blueprint_graph").build_graph(get_llm(), worker)
    result = await graph.ainvoke(
        {"objective": "Fetch two independent sources, then combine them into a summary. Use 3 tasks."},
        {"recursion_limit": 25},
    )
    assert result["tasks"] and all(t.status == "completed" for t in result["tasks"].values())
