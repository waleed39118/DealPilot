"""
Retry with exponential backoff and full jitter.

``delay = uniform(0, min(max_delay, base_delay * multiplier ** (attempt - 1)))``

Full jitter spreads retries from many workers so a recovering provider is not hit by a
synchronized wave. A provider's ``Retry-After`` hint is always honoured; if it exceeds
``max_delay`` the error is raised immediately so the job queue can reschedule the work
instead of holding a worker asleep.
"""

from __future__ import annotations

import asyncio
import inspect
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core.errors import DealPilotError, to_dealpilot_error


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")
        if self.base_delay > self.max_delay:
            raise ValueError("base_delay must be <= max_delay")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")


NO_RETRY = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0)


@dataclass(frozen=True, slots=True)
class RetryEvent:
    """Emitted before each retry sleep (feeds logs and the live activity feed)."""

    operation: str
    attempt: int
    max_attempts: int
    delay: float
    error: DealPilotError


RetryCallback = Callable[[RetryEvent], Awaitable[None] | None]
Sleeper = Callable[[float], Awaitable[None]]


def compute_backoff(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Full-jitter delay before retry number ``attempt`` (1-based)."""
    ceiling = min(policy.max_delay, policy.base_delay * policy.multiplier ** (attempt - 1))
    return rng.uniform(0.0, ceiling)


_DEFAULT_RNG = random.SystemRandom()


async def retry_async[T](
    operation_fn: Callable[[], Awaitable[T]],
    *,
    operation: str,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008  (immutable dataclass)
    provider: str | None = None,
    retry_ambiguous: bool = False,
    on_retry: RetryCallback | None = None,
    sleep: Sleeper = asyncio.sleep,
    rng: random.Random = _DEFAULT_RNG,
) -> T:
    """
    Run ``operation_fn`` until it succeeds, a non-retryable error occurs, or the
    attempts are exhausted. Always raises a ``DealPilotError`` on failure.

    ``retry_ambiguous`` must stay ``False`` for operations with external side effects
    (sending a message) unless the provider call carries an idempotency key.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation_fn()
        except Exception as exc:
            error = to_dealpilot_error(exc, provider=provider)
            if not _should_retry(error, attempt, policy, retry_ambiguous):
                if error is exc:
                    raise
                raise error from exc

            delay = compute_backoff(attempt, policy, rng)
            if error.retry_after is not None:
                delay = max(delay, error.retry_after)

            if on_retry is not None:
                result = on_retry(
                    RetryEvent(
                        operation=operation,
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay=delay,
                        error=error,
                    )
                )
                if inspect.isawaitable(result):
                    await result
            await sleep(delay)


def _should_retry(
    error: DealPilotError, attempt: int, policy: RetryPolicy, retry_ambiguous: bool
) -> bool:
    if not error.retryable or attempt >= policy.max_attempts:
        return False
    if error.may_have_succeeded and not retry_ambiguous:
        return False
    return error.retry_after is None or error.retry_after <= policy.max_delay
