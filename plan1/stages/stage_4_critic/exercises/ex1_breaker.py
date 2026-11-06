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
    raise NotImplementedError("return a decorator using functools.wraps; try up to max_attempts, sleep between attempts")


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
        raise NotImplementedError("closed if opened_at is None; half_open once reset_timeout has elapsed; otherwise open")

    def __call__(self, fn):
        """Use as a decorator on an async function."""
        raise NotImplementedError("wrap fn: refuse when open (or half_open with a trial in flight), run it, update failures and opened_at")


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
        raise NotImplementedError("return Verdict(passed=False, kind='empty') for blank; validate against the schema; otherwise passed=True")
