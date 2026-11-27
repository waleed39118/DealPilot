"""
Ordered provider fallback chains.

A chain holds the same capability from several providers (primary LLM → fallback LLM,
SendGrid → Resend). Each provider is wrapped as::

    circuit breaker( retry with backoff( operation(client) ) )

so retries happen inside one logical call and the breaker counts that call once.

Failover rules:
* TRANSIENT / RATE_LIMITED / AUTH / open circuit → try the next provider.
* PERMANENT → stop: the request itself is bad and would fail everywhere.
* Ambiguous failures (``may_have_succeeded``) → stop unless the operation is declared
  ``idempotent``, because the first provider may already have delivered the side effect
  (e.g. the customer already received the email). ``idempotent`` also allows retrying
  ambiguous failures on the same provider.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

from app.core.errors import (
    AllProvidersFailedError,
    DealPilotError,
    ErrorCategory,
    to_dealpilot_error,
)
from app.core.resilience.circuit_breaker import CircuitBreaker
from app.core.resilience.retry import RetryCallback, RetryPolicy, Sleeper, retry_async


@dataclass(slots=True)
class ProviderSlot[C]:
    """One provider in a chain: a client plus its own breaker and retry policy."""

    name: str
    client: C
    breaker: CircuitBreaker
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    """Emitted when the chain abandons one provider for the next."""

    chain: str
    operation: str
    failed_provider: str
    next_provider: str
    error: DealPilotError


@dataclass(frozen=True, slots=True)
class FallbackResult[T]:
    value: T
    provider: str
    failed_attempts: tuple[DealPilotError, ...]

    @property
    def degraded(self) -> bool:
        """True when the primary provider was skipped or failed."""
        return bool(self.failed_attempts)


FallbackCallback = Callable[[FallbackEvent], Awaitable[None] | None]


class FallbackChain[C]:
    def __init__(
        self,
        name: str,
        slots: Sequence[ProviderSlot[C]],
        *,
        on_fallback: FallbackCallback | None = None,
        on_retry: RetryCallback | None = None,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        if not slots:
            raise ValueError("a fallback chain needs at least one provider")
        names = [slot.name for slot in slots]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate provider names in chain '{name}': {names}")
        self.name = name
        self._slots = tuple(slots)
        self._on_fallback = on_fallback
        self._on_retry = on_retry
        self._sleep = sleep

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self._slots)

    async def execute[T](
        self,
        operation_fn: Callable[[C], Awaitable[T]],
        *,
        operation: str,
        idempotent: bool = False,
    ) -> FallbackResult[T]:
        failures: list[DealPilotError] = []
        for index, slot in enumerate(self._slots):
            try:
                value = await slot.breaker.call(
                    partial(self._with_retry, slot, operation_fn, operation, idempotent)
                )
            except Exception as exc:
                error = to_dealpilot_error(exc, provider=slot.name)
                failures.append(error)
                if not _can_fail_over(error, idempotent):
                    if error is exc:
                        raise
                    raise error from exc
                next_slot = self._slots[index + 1] if index + 1 < len(self._slots) else None
                if next_slot is not None:
                    await self._notify(
                        FallbackEvent(
                            chain=self.name,
                            operation=operation,
                            failed_provider=slot.name,
                            next_provider=next_slot.name,
                            error=error,
                        )
                    )
                continue
            return FallbackResult(value=value, provider=slot.name, failed_attempts=tuple(failures))
        raise AllProvidersFailedError(operation, failures)

    async def _with_retry[T](
        self,
        slot: ProviderSlot[C],
        operation_fn: Callable[[C], Awaitable[T]],
        operation: str,
        idempotent: bool,
    ) -> T:
        return await retry_async(
            partial(operation_fn, slot.client),
            operation=operation,
            policy=slot.retry_policy,
            provider=slot.name,
            retry_ambiguous=idempotent,
            on_retry=self._on_retry,
            sleep=self._sleep,
        )

    async def _notify(self, event: FallbackEvent) -> None:
        if self._on_fallback is None:
            return
        result = self._on_fallback(event)
        if inspect.isawaitable(result):
            await result


def _can_fail_over(error: DealPilotError, idempotent: bool) -> bool:
    if error.category is ErrorCategory.PERMANENT:
        return False
    return idempotent or not error.may_have_succeeded
