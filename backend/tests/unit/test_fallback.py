"""Unit tests for app.core.resilience.fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from app.core.errors import (
    AllProvidersFailedError,
    AuthError,
    PermanentError,
    ProviderUnavailableError,
    TransientError,
)
from app.core.resilience.circuit_breaker import CircuitBreaker
from app.core.resilience.fallback import FallbackChain, FallbackEvent, ProviderSlot
from app.core.resilience.retry import RetryPolicy


@dataclass
class FakeMailer:
    name: str
    errors: list[Exception] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)

    async def send(self, body: str) -> str:
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append(body)
        return f"{self.name}-id"


async def no_sleep(_: float) -> None:
    return None


def chain(
    *mailers: FakeMailer, events: list[FallbackEvent] | None = None
) -> FallbackChain[FakeMailer]:
    slots = [
        ProviderSlot(
            name=mailer.name,
            client=mailer,
            breaker=CircuitBreaker(mailer.name, failure_threshold=3),
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=1.0),
        )
        for mailer in mailers
    ]
    return FallbackChain(
        "email",
        slots,
        sleep=no_sleep,
        on_fallback=events.append if events is not None else None,
    )


def send(fallback: FallbackChain[FakeMailer], idempotent: bool = False) -> object:
    return asyncio.run(
        fallback.execute(lambda m: m.send("hello"), operation="send_email", idempotent=idempotent)
    )


def test_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackChain("email", [])
    mailer = FakeMailer("a")
    with pytest.raises(ValueError, match="duplicate"):
        chain(mailer, FakeMailer("a"))


def test_primary_success_is_not_degraded() -> None:
    primary, backup = FakeMailer("primary"), FakeMailer("backup")
    result = send(chain(primary, backup))
    assert result.value == "primary-id"  # type: ignore[attr-defined]
    assert result.provider == "primary"  # type: ignore[attr-defined]
    assert not result.degraded  # type: ignore[attr-defined]
    assert backup.sent == []


def test_retries_within_provider_before_failing_over() -> None:
    primary = FakeMailer("primary", [TransientError("blip")])
    result = send(chain(primary, FakeMailer("backup")))
    assert result.provider == "primary"  # type: ignore[attr-defined]


def test_fails_over_on_auth_error_and_reports_event() -> None:
    events: list[FallbackEvent] = []
    primary = FakeMailer("primary", [AuthError("revoked key")])
    backup = FakeMailer("backup")
    fallback = chain(primary, backup, events=events)
    result = send(fallback)
    assert result.provider == "backup"  # type: ignore[attr-defined]
    assert result.degraded  # type: ignore[attr-defined]
    assert backup.sent == ["hello"]
    assert len(events) == 1
    assert events[0].failed_provider == "primary"
    assert events[0].next_provider == "backup"
    assert isinstance(events[0].error, AuthError)

    # The primary's circuit is now open: next call skips it without calling it.
    primary.errors.append(AssertionError("primary must not be called"))
    second = send(fallback)
    assert second.provider == "backup"  # type: ignore[attr-defined]
    assert isinstance(second.failed_attempts[0], ProviderUnavailableError)  # type: ignore[attr-defined]


def test_permanent_error_stops_the_chain() -> None:
    backup = FakeMailer("backup")
    with pytest.raises(PermanentError):
        send(chain(FakeMailer("primary", [PermanentError("invalid recipient")]), backup))
    assert backup.sent == []


def test_ambiguous_failure_never_duplicates_side_effects() -> None:
    backup = FakeMailer("backup")
    with pytest.raises(TransientError) as exc_info:
        send(chain(FakeMailer("primary", [TimeoutError("read timeout")]), backup))
    assert exc_info.value.may_have_succeeded
    assert backup.sent == []


def test_ambiguous_failure_fails_over_when_idempotent() -> None:
    primary = FakeMailer("primary", [TimeoutError("t1"), TimeoutError("t2")])
    result = send(chain(primary, FakeMailer("backup")), idempotent=True)
    assert result.provider == "backup"  # type: ignore[attr-defined]


def test_all_providers_failed() -> None:
    fallback = chain(
        FakeMailer("primary", [AuthError("revoked")]),
        FakeMailer("backup", [TransientError("down"), TransientError("down")]),
    )
    with pytest.raises(AllProvidersFailedError) as exc_info:
        send(fallback)
    error = exc_info.value
    assert [e.provider for e in error.errors] == ["primary", "backup"]
    assert error.retryable
    assert error.operation == "send_email"
