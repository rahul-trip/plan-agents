"""Stage 4, ex1: retry, circuit breaker and Evaluator. Custom track.

Time is injected, so tests never sleep. The retry decorator handles a blip; the breaker stops you
hammering something that is down.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, ValidationError


class TransientError(Exception):
    """Worth retrying: timeouts, 429s, 5xx."""


class PermanentError(Exception):
    """Retrying can't help: 403, 404, a missing data source."""


class CircuitOpenError(Exception):
    """The breaker refused the call without attempting it."""


def retry(
    max_attempts: int = 3,
    retry_on: tuple[type[Exception], ...] = (TransientError,),
    base_delay: float = 0.1,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
):
    """Decorator for async functions: exponential backoff (base_delay * 2**attempt).

    Only exceptions in retry_on are retried; anything else propagates at once. After
    max_attempts, re-raise the last error. No sleep after the final attempt.
    """
    # >>> TODO: return a decorator using functools.wraps; try up to max_attempts, sleep between attempts
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retry_on:
                    if attempt == max_attempts - 1:
                        raise
                    await sleep(base_delay * 2**attempt)

        return wrapper

    return decorator
    # <<<


class CircuitBreaker:
    """Closed -> open after failure_threshold consecutive failures -> half-open after reset_timeout.

    closed: calls pass; a success resets the failure count.
    open: calls raise CircuitOpenError without running.
    half-open: one trial call passes. Success closes the circuit; failure reopens it and restarts
    the timeout. While that trial is in flight, other callers get CircuitOpenError.

    Calls can overlap. A call that started while closed may finish after the circuit opened, and its
    outcome is stale: only the half-open trial may close an open circuit, and a stale failure must
    not restart the timeout.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self.trial_in_flight = False

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        # >>> TODO: closed if opened_at is None; half_open once reset_timeout has elapsed; otherwise open
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_timeout else "open"
        # <<<

    def __call__(self, fn):
        """Use as a decorator on an async function."""
        # >>> TODO: wrap fn: refuse when open (or half_open with a trial in flight), run it, update failures and opened_at
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            state = self.state
            if state == "open" or (state == "half_open" and self.trial_in_flight):
                raise CircuitOpenError(getattr(fn, "__name__", repr(fn)))
            self.trial_in_flight = state == "half_open"
            try:
                result = await fn(*args, **kwargs)
            except Exception:
                if state == "half_open":
                    self.opened_at = self.clock()
                elif self.opened_at is None:
                    self.failures += 1
                    if self.failures >= self.failure_threshold:
                        self.opened_at = self.clock()
                raise
            else:
                if state == "half_open":
                    self.opened_at = None
                    self.failures = 0
                elif self.opened_at is None:
                    self.failures = 0
                return result
            finally:
                if state == "half_open":
                    self.trial_in_flight = False

        return wrapper
        # <<<


class Verdict(BaseModel):
    passed: bool
    kind: Literal["ok", "schema", "empty"] = "ok"
    reason: str = ""


class Evaluator:
    """Deterministic gate on a worker's output. Runs before anything expensive looks at it."""

    def __init__(self, schema: type[BaseModel] | None = None):
        self.schema = schema

    def evaluate(self, output: str | None) -> Verdict:
        """None or blank is kind='empty'. If schema is set, output must parse with
        schema.model_validate_json, else kind='schema' with the error as reason."""
        # >>> TODO: return Verdict(passed=False, kind='empty') for blank; validate against the schema; otherwise passed=True
        if output is None or not output.strip():
            return Verdict(passed=False, kind="empty", reason="Output is empty")
        if self.schema is not None:
            try:
                self.schema.model_validate_json(output)
            except ValidationError as exc:
                return Verdict(passed=False, kind="schema", reason=str(exc))
        return Verdict(passed=True)
        # <<<
