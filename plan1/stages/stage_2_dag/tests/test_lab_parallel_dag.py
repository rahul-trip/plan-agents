import pytest

from planagents.dag import validate_plan
from planagents.fakes import ScriptedLLM


@pytest.fixture
def lab(load):
    return load("lab_parallel_dag")


def test_plan_has_six_valid_nodes_and_three_roots(lab):
    tasks = validate_plan(lab.six_node_plan())
    assert len(tasks) == 6
    assert {t.id for t in tasks.values() if not t.depends_on} == set(lab.ROOTS)


async def test_timestamps_prove_parallel_branches_and_waiting_synthesis(lab):
    plan = lab.six_node_plan()
    run = await lab.run_lab(ScriptedLLM({"PlanDraft": [plan]}))
    timeline = run.timeline
    assert timeline.overlap(*lab.ROOTS), "independent branches must run at the same time"
    assert timeline.overlap("extract_breaking_changes", "compare_schema")
    assert timeline.starts_after("synthesize_report", "extract_breaking_changes", "compare_schema")
    assert timeline.starts_after("synthesize_report", *lab.ROOTS)


async def test_roots_share_a_single_superstep(lab):
    run = await lab.run_lab(ScriptedLLM({"PlanDraft": [lab.six_node_plan()]}))
    assert [sorted(batch) for batch in run.batches] == [
        sorted(lab.ROOTS),
        ["compare_schema", "extract_breaking_changes"],
        ["synthesize_report"],
    ]
