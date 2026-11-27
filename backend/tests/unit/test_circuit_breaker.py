"""Unit tests for app.core.resilience.circuit_breaker."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from app.core.errors import (
    AuthError,
    PermanentError,
    ProviderUnavailableError,
    TransientError,
)
from app.core.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitStateChange,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(
    clock: FakeClock, changes: list[CircuitStateChange] | None = None, **kwargs: int
) -> CircuitBreaker:
    return CircuitBreaker(
        "mail",
        failure_threshold=kwargs.get("failure_threshold", 3),
        recovery_timeout=30,
        half_open_max_calls=kwargs.get("half_open_max_calls", 1),
        clock=clock,
        on_state_change=changes.append if changes is not None else None,
    )


def failing(error: Exception) -> Callable[[], Awaitable[str]]:
    async def call() -> str:
        raise error

    return call


async def succeed() -> str:
    return "ok"


async def trip(breaker: CircuitBreaker, times: int) -> None:
    for _ in range(times):
        with pytest.raises(TransientError):
            await breaker.call(failing(TransientError("down")))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker("x", failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker("x", recovery_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker("x", half_open_max_calls=0)


def test_opens_after_threshold_and_rejects_without_calling() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        changes: list[CircuitStateChange] = []
        breaker = make(clock, changes)
        await trip(breaker, 3)
        assert breaker.state is CircuitState.OPEN

        calls = 0

        async def counted() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        clock.now += 10
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await breaker.call(counted)
        assert calls == 0
        assert exc_info.value.retry_after == pytest.approx(20)
        assert [c.current for c in changes] == [CircuitState.OPEN]

    asyncio.run(scenario())


def test_half_open_success_closes() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        changes: list[CircuitStateChange] = []
        breaker = make(clock, changes)
        await trip(breaker, 3)
        clock.now += 31
        assert breaker.state is CircuitState.HALF_OPEN
        assert await breaker.call(succeed) == "ok"
        assert breaker.state is CircuitState.CLOSED
        assert breaker.consecutive_failures == 0
        assert [c.current for c in changes] == [
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
            CircuitState.CLOSED,
        ]

    asyncio.run(scenario())


def test_half_open_failure_reopens() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        breaker = make(clock)
        await trip(breaker, 3)
        clock.now += 31
        await trip(breaker, 1)
        assert breaker.state is CircuitState.OPEN
        with pytest.raises(ProviderUnavailableError):
            await breaker.call(succeed)

    asyncio.run(scenario())


def test_half_open_limits_concurrent_probes() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        breaker = make(clock)
        await trip(breaker, 3)
        clock.now += 31
        release = asyncio.Event()

        async def slow() -> str:
            await release.wait()
            return "ok"

        probe = asyncio.create_task(breaker.call(slow))
        await asyncio.sleep(0)
        with pytest.raises(ProviderUnavailableError, match="probing"):
            await breaker.call(succeed)
        release.set()
        assert await probe == "ok"
        assert breaker.state is CircuitState.CLOSED

    asyncio.run(scenario())


def test_cancelled_probe_releases_slot() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        breaker = make(clock)
        await trip(breaker, 3)
        clock.now += 31

        async def hang() -> str:
            await asyncio.Event().wait()
            return "never"

        probe = asyncio.create_task(breaker.call(hang))
        await asyncio.sleep(0)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        assert await breaker.call(succeed) == "ok"

    asyncio.run(scenario())


def test_auth_error_opens_immediately() -> None:
    async def scenario() -> None:
        breaker = make(FakeClock())
        with pytest.raises(AuthError):
            await breaker.call(failing(AuthError("revoked key")))
        assert breaker.state is CircuitState.OPEN

    asyncio.run(scenario())


def test_permanent_errors_do_not_trip() -> None:
    async def scenario() -> None:
        breaker = make(FakeClock())
        for _ in range(10):
            with pytest.raises(PermanentError):
                await breaker.call(failing(ValueError("our bug")))
        assert breaker.state is CircuitState.CLOSED
        assert breaker.consecutive_failures == 0

    asyncio.run(scenario())


def test_success_resets_failure_count() -> None:
    async def scenario() -> None:
        breaker = make(FakeClock())
        await trip(breaker, 2)
        await breaker.call(succeed)
        await trip(breaker, 2)
        assert breaker.state is CircuitState.CLOSED

    asyncio.run(scenario())


def test_manual_reset_and_async_callback() -> None:
    async def scenario() -> None:
        seen: list[CircuitStateChange] = []

        async def on_change(change: CircuitStateChange) -> None:
            seen.append(change)

        breaker = CircuitBreaker("mail", failure_threshold=1, on_state_change=on_change)
        with pytest.raises(AuthError):
            await breaker.call(failing(AuthError("revoked")))
        await breaker.reset("key replaced")
        assert breaker.state is CircuitState.CLOSED
        assert seen[-1].reason == "key replaced"
        assert "revoked" in seen[0].reason

    asyncio.run(scenario())
