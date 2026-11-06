import pytest

from planagents.schemas import Step


def step(id: str, status: str = "pending") -> Step:
    return Step(id=id, description="d", assigned_worker="w", status=status)


@pytest.fixture
def merge(load):
    return load("ex2_reducer").merge_steps


def ids(steps):
    return [s.id for s in steps]


def test_updates_in_place_instead_of_appending(merge):
    merged = merge([step("a"), step("b")], [step("a", "completed")])
    assert ids(merged) == ["a", "b"]
    assert merged[0].status == "completed"


def test_new_ids_are_appended_in_order(merge):
    assert ids(merge([step("a")], [step("x"), step("y")])) == ["a", "x", "y"]


def test_handles_empty_initial_state(merge):
    assert ids(merge(None, [step("a")])) == ["a"]


def test_idempotent(merge):
    update = [step("a", "completed"), step("c")]
    once = merge([step("a"), step("b")], update)
    assert merge(once, update) == once


def test_does_not_mutate_inputs(merge):
    existing, update = [step("a")], [step("a", "completed"), step("b")]
    merge(existing, update)
    assert existing == [step("a")]
    assert ids(update) == ["a", "b"]
