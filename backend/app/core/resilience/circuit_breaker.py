"""
Per-provider circuit breaker.

CLOSED     → calls pass through; consecutive health failures are counted.
OPEN       → calls are rejected immediately with ``ProviderUnavailableError`` until
             ``recovery_timeout`` has elapsed (protects the provider and our quotas).
HALF_OPEN  → a limited number of probe calls are let through; a success closes the
             circuit, a failure re-opens it.

Only failures that say something about the provider's health trip the breaker:
TRANSIENT and RATE_LIMITED count towards ``failure_threshold``; AUTH opens the circuit
at once (a revoked key will not heal by retrying). PERMANENT errors (e.g. our own bad
request) do not affect the circuit.

State is per process. Each worker protects itself; the watchdog aggregates health.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from app.core.errors import ErrorCategory, ProviderUnavailableError, to_dealpilot_error


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitStateChange:
    name: str
    previous: CircuitState
    current: CircuitState
    reason: str


StateChangeCallback = Callable[[CircuitStateChange], Awaitable[None] | None]
Clock = Callable[[], float]


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Clock = time.monotonic,
        on_state_change: StateChangeCallback | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_in_flight = 0
        self._lock = asyncio.Lock()

    # ── Introspection ───────────────────────────────────────────────────────
    @property
    def state(self) -> CircuitState:
        """Current state, reporting OPEN→HALF_OPEN once the timeout has elapsed."""
        if self._state is CircuitState.OPEN and self._remaining_open_time() <= 0:
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def _remaining_open_time(self) -> float:
        return self._opened_at + self._recovery_timeout - self._clock()

    # ── Public API ──────────────────────────────────────────────────────────
    async def call[T](self, operation_fn: Callable[[], Awaitable[T]]) -> T:
        await self._before_call()
        try:
            result = await operation_fn()
        except Exception as exc:
            error = to_dealpilot_error(exc, provider=self.name)
            await self._on_failure(error.category, str(error))
            if error is exc:
                raise
            raise error from exc
        except BaseException:
            # Cancellation is not a provider failure, but a probe slot must be released.
            await self._release_probe()
            raise
        await self._on_success()
        return result

    async def reset(self, reason: str = "manual reset") -> None:
        """Force CLOSED, e.g. after the user replaced a revoked API key."""
        async with self._lock:
            await self._transition(CircuitState.CLOSED, reason)

    # ── Internals ───────────────────────────────────────────────────────────
    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                remaining = self._remaining_open_time()
                if remaining > 0:
                    raise ProviderUnavailableError(
                        f"circuit '{self.name}' is open",
                        provider=self.name,
                        retry_after=remaining,
                    )
                await self._transition(CircuitState.HALF_OPEN, "recovery timeout elapsed")
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self._half_open_max_calls:
                    raise ProviderUnavailableError(
                        f"circuit '{self.name}' is half-open and probing",
                        provider=self.name,
                        retry_after=self._recovery_timeout,
                    )
                self._half_open_in_flight += 1

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                await self._transition(CircuitState.CLOSED, "probe call succeeded")
            else:
                self._failures = 0

    async def _on_failure(self, category: ErrorCategory, description: str) -> None:
        async with self._lock:
            if category is ErrorCategory.PERMANENT:
                if self._state is CircuitState.HALF_OPEN:
                    self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                return
            if category is ErrorCategory.AUTH:
                await self._transition(CircuitState.OPEN, f"authentication failed: {description}")
                return
            if self._state is CircuitState.HALF_OPEN:
                await self._transition(CircuitState.OPEN, f"probe call failed: {description}")
                return
            self._failures += 1
            if self._failures >= self._failure_threshold:
                await self._transition(
                    CircuitState.OPEN,
                    f"{self._failures} consecutive failures; last: {description}",
                )

    async def _release_probe(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    async def _transition(self, new_state: CircuitState, reason: str) -> None:
        """Must be called with ``self._lock`` held."""
        previous = self._state
        if new_state is CircuitState.OPEN:
            self._opened_at = self._clock()
        if new_state is CircuitState.CLOSED:
            self._failures = 0
        self._half_open_in_flight = 0
        self._state = new_state
        if previous is new_state or self._on_state_change is None:
            return
        result = self._on_state_change(
            CircuitStateChange(name=self.name, previous=previous, current=new_state, reason=reason)
        )
        if inspect.isawaitable(result):
            await result
