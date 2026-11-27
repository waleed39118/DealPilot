"""
Fault-injection tests: break things on purpose and verify the system
(1) catches the error, (2) applies a fallback, (3) records an incident and
(4) emits a phone notification explaining what happened and how it was handled.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from app.core.errors import AllProvidersFailedError, TransientError, error_from_http_status
from app.core.resilience.circuit_breaker import CircuitBreaker, CircuitState
from app.core.resilience.fallback import FallbackChain, ProviderSlot
from app.core.resilience.retry import RetryPolicy
from app.healing.incidents import (
    IncidentManager,
    IncidentStatus,
    InMemoryIncidentStore,
    circuit_handler,
    fallback_handler,
)
from app.healing.watchdog import Watchdog
from app.telemetry.events import EventBus, EventType, InMemorySink


@dataclass
class ChaosMailer:
    """Email provider whose failures are injected by the test."""

    name: str
    fault: BaseException | None = None
    sent: list[str] = field(default_factory=list)

    async def send(self, body: str) -> str:
        if self.fault is not None:
            raise self.fault
        self.sent.append(body)
        return f"{self.name}-msg"

    async def check(self) -> None:
        if self.fault is not None:
            raise self.fault


async def no_sleep(_: float) -> None:
    return None


class Rig:
    def __init__(self) -> None:
        self.store = InMemoryIncidentStore()
        self.feed = InMemorySink("feed")
        self.manager = IncidentManager(self.store, EventBus([self.feed]))
        self.primary = ChaosMailer("sendgrid")
        self.backup = ChaosMailer("resend")
        self.breakers = {
            mailer.name: CircuitBreaker(
                mailer.name,
                failure_threshold=2,
                recovery_timeout=0.05,
                on_state_change=circuit_handler(self.manager, operation="send_email"),
            )
            for mailer in (self.primary, self.backup)
        }
        self.chain = FallbackChain(
            "email",
            [
                ProviderSlot(
                    m.name,
                    m,
                    self.breakers[m.name],
                    RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=1.0),
                )
                for m in (self.primary, self.backup)
            ],
            on_fallback=fallback_handler(self.manager),
            sleep=no_sleep,
        )

    def send(self) -> str:
        result = asyncio.run(self.chain.execute(lambda m: m.send("offer"), operation="send_email"))
        return result.provider


def test_network_outage_fails_over_and_notifies() -> None:
    rig = Rig()
    rig.primary.fault = ConnectionRefusedError("network down")

    assert rig.send() == "resend"
    assert rig.backup.sent == ["offer"]

    (incident,) = rig.store.incidents.values()
    assert incident.cause == "network_unreachable"
    assert incident.status is IncidentStatus.MITIGATED
    assert "switched from sendgrid to resend" in incident.actions_taken
    opened = rig.feed.of_type(EventType.INCIDENT_OPENED)
    assert len(opened) == 1 and opened[0].should_push


def test_wrong_api_key_disables_provider_and_requires_user() -> None:
    rig = Rig()
    rig.primary.fault = error_from_http_status(401, provider="sendgrid")

    assert rig.send() == "resend"
    assert rig.breakers["sendgrid"].state is CircuitState.OPEN

    (incident,) = rig.store.incidents.values()
    assert incident.cause == "invalid_credentials"
    assert incident.requires_user_action
    assert incident.status is IncidentStatus.OPEN  # not "mitigated": the key must be fixed
    event = rig.feed.of_type(EventType.INCIDENT_OPENED)[0]
    assert event.params["requires_user_action"] is True


def test_recovery_closes_circuit_and_resolves_incident() -> None:
    rig = Rig()
    rig.primary.fault = ConnectionRefusedError("network down")
    rig.send()
    rig.send()
    assert rig.breakers["sendgrid"].state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    rig.primary.fault = None
    asyncio.run(asyncio.sleep(0.06))  # recovery timeout elapses
    assert rig.send() == "sendgrid"

    assert rig.breakers["sendgrid"].state is CircuitState.CLOSED
    assert all(i.status is IncidentStatus.RESOLVED for i in rig.store.incidents.values())
    assert rig.feed.of_type(EventType.INCIDENT_RESOLVED)


def test_ambiguous_timeout_never_double_sends() -> None:
    rig = Rig()
    rig.primary.fault = TimeoutError("read timed out after request was sent")

    with pytest.raises(TransientError) as exc_info:
        rig.send()
    assert exc_info.value.may_have_succeeded
    assert rig.backup.sent == []  # the customer can never receive the offer twice


def test_total_outage_reports_all_providers_failed() -> None:
    rig = Rig()
    rig.primary.fault = ConnectionRefusedError("down")
    rig.backup.fault = error_from_http_status(503, provider="resend")

    with pytest.raises(AllProvidersFailedError) as exc_info:
        rig.send()
    assert exc_info.value.retryable  # the job queue will try again later


def test_watchdog_detects_and_reports_recovery() -> None:
    rig = Rig()
    rig.primary.fault = error_from_http_status(503, provider="sendgrid")
    watchdog = Watchdog([rig.primary], rig.manager, failure_threshold=2)

    async def scenario() -> None:
        await watchdog.run_once()
        await watchdog.run_once()
        rig.primary.fault = None
        await watchdog.run_once()

    asyncio.run(scenario())
    assert len(rig.feed.of_type(EventType.INCIDENT_OPENED)) == 1
    resolved = rig.feed.of_type(EventType.INCIDENT_RESOLVED)
    assert len(resolved) == 1
    assert "healthy again" in resolved[0].summary
