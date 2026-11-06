import pytest


@pytest.fixture
def budget_guard(load):
    """Study workflow #2: a pytest fixture that polices every payload inside worker_executor.

    Wrap the worker under test:

        def test_x(budget_guard):
            guard = budget_guard(max_tokens=100, canaries=["SECRET"])
            graph = build_graph(llm, guard.wrap(worker))
            ...

    At teardown the test fails if any violation was recorded, even if the graph caught the exception
    and carried on, or if the guard never saw a payload, because a guard that never ran proves nothing.
    Tests that provoke violations on purpose pass expect_violations=True.
    """
    guards = []

    def make(max_tokens: int = 2000, canaries: list[str] | None = None, expect_violations: bool = False):
        guard = load("ex3_budget_guard").PayloadGuard(max_tokens, canaries or [])
        guards.append((guard, expect_violations))
        return guard

    yield make
    for guard, expect_violations in guards:
        assert guard.checked > 0, "budget_guard was never exercised"
        if not expect_violations:
            guard.assert_clean()
