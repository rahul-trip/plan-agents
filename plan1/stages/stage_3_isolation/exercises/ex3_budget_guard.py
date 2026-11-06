"""Stage 3, ex3: enforce token budgets in tests (study workflow #2).

Wrap the worker so every payload is checked at the moment it arrives. tests/conftest.py turns this
into the `budget_guard` pytest fixture. Read that fixture too: it is half of this exercise.

The trap this design avoids: raising isn't enough. A real worker_executor catches exceptions and
marks the task failed, so a violation raised inside it never reaches your test. The guard therefore
also records every violation, and the fixture checks that record when the test ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from planagents.schemas import TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


class TokenBudgetExceeded(AssertionError):
    pass


class ContextLeak(AssertionError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough estimate of 4 characters per token. Fine for a guard; never use it for billing."""
    return len(text) // 4


@dataclass
class PayloadGuard:
    max_tokens: int
    canaries: list[str] = field(default_factory=list)
    largest_payload: int = 0
    checked: int = 0
    violations: list[str] = field(default_factory=list)

    def _reject(self, error: AssertionError) -> None:
        self.violations.append(str(error))
        raise error

    def check(self, task: TaskNode, upstream: dict[str, str]) -> None:
        """Reject this payload (through self._reject) if it breaks a rule, in this order:

        - ContextLeak if upstream has any key not in task.depends_on. Name that key in the message.
        - ContextLeak if any canary string appears in the payload. Name that canary in the message.
        - TokenBudgetExceeded if the payload's estimated tokens exceed max_tokens.

        The payload is task.instruction plus upstream's keys and values. If it passes, update checked
        and largest_payload. A rejected payload doesn't count as checked.
        """
        raise NotImplementedError("implement the three checks in order, using self._reject, then update the counters")

    def wrap(self, worker: Worker) -> Worker:
        """Return a worker that runs check() before calling the original."""
        raise NotImplementedError("return an async wrapper that calls self.check then awaits worker")

    def assert_clean(self) -> None:
        """Fail if any violation was recorded, even one a graph swallowed."""
        if self.violations:
            raise ContextLeak("; ".join(self.violations))
