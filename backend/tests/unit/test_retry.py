"""Unit tests for app.core.resilience.retry."""

from __future__ import annotations

import asyncio
import random

import pytest

from app.core.errors import AuthError, PermanentError, RateLimitedError, TransientError
from app.core.resilience.retry import (
    NO_RETRY,
    RetryEvent,
    RetryPolicy,
    compute_backoff,
    retry_async,
)


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class Flaky:
    """Fails with the given errors in order, then returns 'ok'."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


POLICY = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=10.0)


def run(**kwargs: object) -> tuple[object, FakeSleep]:
    sleep = FakeSleep()
    result = asyncio.run(
        retry_async(operation="op", sleep=sleep, rng=random.Random(1), **kwargs)  # type: ignore[arg-type]
    )
    return result, sleep


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=5, max_delay=1)
    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(multiplier=0.5)


def test_backoff_is_bounded_by_exponential_ceiling() -> None:
    rng = random.Random(42)
    for attempt in range(1, 10):
        ceiling = min(POLICY.max_delay, POLICY.base_delay * 2 ** (attempt - 1))
        for _ in range(50):
            assert 0.0 <= compute_backoff(attempt, POLICY, rng) <= ceiling


def test_recovers_from_transient_errors() -> None:
    flaky = Flaky(TransientError("blip"), TransientError("blip"))
    result, sleep = run(operation_fn=flaky, policy=POLICY)
    assert result == "ok"
    assert flaky.calls == 3
    assert len(sleep.delays) == 2


def test_non_retryable_errors_raise_immediately() -> None:
    for error in (AuthError("revoked"), PermanentError("bad request")):
        flaky = Flaky(error)
        with pytest.raises(type(error)):
            run(operation_fn=flaky, policy=POLICY)
        assert flaky.calls == 1


def test_exhaustion_raises_last_error() -> None:
    flaky = Flaky(*(TransientError(f"fail {i}") for i in range(10)))
    with pytest.raises(TransientError, match="fail 3"):
        run(operation_fn=flaky, policy=POLICY)
    assert flaky.calls == POLICY.max_attempts


def test_ambiguous_failure_only_retried_when_allowed() -> None:
    flaky = Flaky(TimeoutError("read timeout"))
    with pytest.raises(TransientError):
        run(operation_fn=flaky, policy=POLICY)
    assert flaky.calls == 1

    flaky = Flaky(TimeoutError("read timeout"))
    result, _ = run(operation_fn=flaky, policy=POLICY, retry_ambiguous=True)
    assert result == "ok"
    assert flaky.calls == 2


def test_retry_after_is_honoured() -> None:
    flaky = Flaky(RateLimitedError("slow down", retry_after=8))
    result, sleep = run(operation_fn=flaky, policy=POLICY)
    assert result == "ok"
    assert sleep.delays[0] >= 8


def test_retry_after_beyond_max_delay_stops_retrying() -> None:
    flaky = Flaky(RateLimitedError("come back later", retry_after=600))
    with pytest.raises(RateLimitedError):
        run(operation_fn=flaky, policy=POLICY)
    assert flaky.calls == 1


def test_unknown_exception_is_classified_permanent_and_chained() -> None:
    flaky = Flaky(KeyError("bug"))
    with pytest.raises(PermanentError) as exc_info:
        run(operation_fn=flaky, policy=POLICY, provider="crm")
    assert isinstance(exc_info.value.__cause__, KeyError)
    assert exc_info.value.provider == "crm"


def test_on_retry_callback_sync_and_async() -> None:
    seen: list[RetryEvent] = []

    async def async_callback(event: RetryEvent) -> None:
        seen.append(event)

    flaky = Flaky(TransientError("a"), TransientError("b"))
    run(operation_fn=flaky, policy=POLICY, on_retry=async_callback)
    assert [event.attempt for event in seen] == [1, 2]
    assert seen[0].operation == "op"
    assert seen[0].max_attempts == POLICY.max_attempts

    seen.clear()
    run(operation_fn=Flaky(TransientError("c")), policy=POLICY, on_retry=seen.append)
    assert len(seen) == 1


def test_no_retry_policy() -> None:
    flaky = Flaky(TransientError("blip"))
    with pytest.raises(TransientError):
        run(operation_fn=flaky, policy=NO_RETRY)
    assert flaky.calls == 1


def test_cancellation_is_not_swallowed() -> None:
    flaky = Flaky(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(operation_fn=flaky, policy=POLICY)
    assert flaky.calls == 1
