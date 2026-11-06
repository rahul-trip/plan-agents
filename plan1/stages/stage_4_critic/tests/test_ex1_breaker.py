import asyncio

import pytest
from pydantic import BaseModel


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Flaky:
    """Raises the scripted exceptions in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture
def ex(load):
    return load("ex1_breaker")


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def fake_sleep(sleeps):
    async def sleep(seconds):
        sleeps.append(seconds)

    return sleep


async def test_retry_recovers_from_transient_errors_with_backoff(ex, fake_sleep, sleeps):
    fn = Flaky(ex.TransientError(), ex.TransientError())
    assert await ex.retry(3, base_delay=1, sleep=fake_sleep)(fn)() == "ok"
    assert fn.calls == 3
    assert sleeps == [1, 2]


async def test_retry_gives_up_after_max_attempts_without_a_final_sleep(ex, fake_sleep, sleeps):
    fn = Flaky(*[ex.TransientError("x") for _ in range(5)])
    with pytest.raises(ex.TransientError):
        await ex.retry(2, base_delay=1, sleep=fake_sleep)(fn)()
    assert fn.calls == 2
    assert sleeps == [1]


async def test_permanent_errors_are_not_retried(ex, fake_sleep):
    fn = Flaky(ex.PermanentError())
    with pytest.raises(ex.PermanentError):
        await ex.retry(3, sleep=fake_sleep)(fn)()
    assert fn.calls == 1


async def _fail_times(breaker_fn, exc_type, times):
    for _ in range(times):
        with pytest.raises(exc_type):
            await breaker_fn()


async def test_breaker_opens_after_threshold_and_refuses_calls(ex):
    breaker = ex.CircuitBreaker(failure_threshold=2, clock=Clock())
    fn = Flaky(*[ValueError() for _ in range(9)])
    guarded = breaker(fn)
    assert breaker.state == "closed"
    await _fail_times(guarded, ValueError, 2)
    assert breaker.state == "open"
    with pytest.raises(ex.CircuitOpenError):
        await guarded()
    assert fn.calls == 2, "an open breaker must not call through"


async def test_success_resets_the_failure_count(ex):
    ok_then_fail = ex.CircuitBreaker(failure_threshold=2, clock=Clock())
    outcomes = iter([ValueError(), "ok", ValueError(), "ok"])

    async def fn():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    wrapped = ok_then_fail(fn)
    with pytest.raises(ValueError):
        await wrapped()
    assert await wrapped() == "ok"
    with pytest.raises(ValueError):
        await wrapped()
    assert ok_then_fail.state == "closed", "failures must be consecutive to open the circuit"


async def test_half_open_trial_success_closes(ex):
    clock = Clock()
    breaker = ex.CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    guarded = breaker(Flaky(ValueError()))
    await _fail_times(guarded, ValueError, 1)
    clock.now = 10
    assert breaker.state == "half_open"
    assert await guarded() == "ok"
    assert breaker.state == "closed"


async def test_half_open_trial_failure_reopens_and_restarts_the_timeout(ex):
    clock = Clock()
    breaker = ex.CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    guarded = breaker(Flaky(ValueError(), ValueError()))
    await _fail_times(guarded, ValueError, 1)
    clock.now = 10
    await _fail_times(guarded, ValueError, 1)
    assert breaker.state == "open"
    clock.now = 19
    assert breaker.state == "open"
    clock.now = 20
    assert breaker.state == "half_open"


async def test_half_open_allows_only_one_trial_at_a_time(ex):
    clock = Clock()
    breaker = ex.CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    release = asyncio.Event()
    outcomes = [ValueError()]

    async def fn():
        if outcomes:
            raise outcomes.pop()
        await release.wait()
        return "ok"

    guarded = breaker(fn)
    await _fail_times(guarded, ValueError, 1)
    clock.now = 10
    trial = asyncio.create_task(guarded())
    await asyncio.sleep(0)
    with pytest.raises(ex.CircuitOpenError):
        await guarded()
    release.set()
    assert await trial == "ok"


async def test_stale_success_cannot_close_an_open_circuit(ex):
    """Regression: a slow call from before the outage closed the circuit while the service was down."""
    breaker = ex.CircuitBreaker(failure_threshold=1, reset_timeout=100, clock=Clock())
    release = asyncio.Event()

    async def slow():
        await release.wait()
        return "ok"

    in_flight = asyncio.create_task(breaker(slow)())
    await asyncio.sleep(0)
    await _fail_times(breaker(Flaky(ValueError())), ValueError, 1)
    assert breaker.state == "open"
    release.set()
    assert await in_flight == "ok"
    assert breaker.state == "open"


async def test_stale_failure_does_not_restart_the_timeout(ex):
    clock = Clock()
    breaker = ex.CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    release = asyncio.Event()

    async def slow():
        await release.wait()
        raise ValueError("late")

    in_flight = asyncio.create_task(breaker(slow)())
    await asyncio.sleep(0)
    await _fail_times(breaker(Flaky(ValueError())), ValueError, 1)
    clock.now = 9
    release.set()
    with pytest.raises(ValueError):
        await in_flight
    clock.now = 10
    assert breaker.state == "half_open"


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("output", [None, "", "   "])
def test_evaluator_rejects_empty(ex, output):
    assert ex.Evaluator().evaluate(output).kind == "empty"


def test_evaluator_checks_schema(ex):
    evaluator = ex.Evaluator(Answer)
    assert evaluator.evaluate('{"value": 3}').passed is True
    verdict = evaluator.evaluate('{"value": "many"}')
    assert (verdict.passed, verdict.kind) == (False, "schema")
    assert verdict.reason
