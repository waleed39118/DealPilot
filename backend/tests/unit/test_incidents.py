"""Unit tests for app.healing.incidents."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.core.errors import AuthError, TransientError
from app.healing.diagnostics import diagnose
from app.healing.incidents import IncidentManager, IncidentStatus, InMemoryIncidentStore
from app.telemetry.events import EventBus, EventType, InMemorySink, Severity


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 27, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def setup() -> tuple[IncidentManager, InMemoryIncidentStore, InMemorySink]:
    store, sink = InMemoryIncidentStore(), InMemorySink()
    return IncidentManager(store, EventBus([sink]), clock=Clock()), store, sink


def test_repeated_failures_open_one_incident_and_one_notification() -> None:
    manager, store, sink = setup()

    async def scenario() -> None:
        for _ in range(5):
            await manager.report(
                diagnose(TransientError("503"), provider="mail"),
                operation="send_email",
                action_taken="retried",
            )

    asyncio.run(scenario())
    (incident,) = store.incidents.values()
    assert incident.occurrences == 5
    assert len(incident.actions_taken) == 5
    opened = sink.of_type(EventType.INCIDENT_OPENED)
    assert len(opened) == 1
    assert opened[0].should_push
    assert opened[0].code == "incident.provider_outage"


def test_escalation_notifies_again() -> None:
    manager, _store, sink = setup()

    async def scenario() -> None:
        diagnosis = diagnose(TransientError("503"), provider="mail")
        incident = await manager.report(diagnosis, operation="op", action_taken="retried")
        critical = diagnosis.__class__(
            **{
                **{f: getattr(diagnosis, f) for f in diagnosis.__slots__},
                "severity": Severity.CRITICAL,
            }
        )
        updated = await manager.report(critical, operation="op", action_taken="still failing")
        assert updated.id == incident.id
        assert updated.severity is Severity.CRITICAL

    asyncio.run(scenario())
    updates = sink.of_type(EventType.INCIDENT_UPDATED)
    assert len(updates) == 1 and updates[0].push


def test_mitigate_then_resolve() -> None:
    manager, _store, sink = setup()

    async def scenario() -> None:
        incident = await manager.report(
            diagnose(ConnectionRefusedError(), provider="sendgrid"),
            operation="send_email",
            action_taken="switched to resend",
        )
        mitigated = await manager.mitigate(incident.fingerprint, "traffic on resend")
        assert mitigated is not None and mitigated.status is IncidentStatus.MITIGATED
        again = await manager.mitigate(incident.fingerprint, "noop")
        assert again is not None and again.actions_taken == mitigated.actions_taken
        resolved = await manager.resolve(incident.fingerprint, "sendgrid recovered")
        assert resolved is not None and resolved.status is IncidentStatus.RESOLVED
        assert resolved.resolution == "sendgrid recovered"
        assert await manager.resolve(incident.fingerprint, "twice") is None

    asyncio.run(scenario())
    resolved_events = sink.of_type(EventType.INCIDENT_RESOLVED)
    assert len(resolved_events) == 1
    assert resolved_events[0].severity is Severity.SUCCESS


def test_new_failure_after_resolution_opens_new_incident() -> None:
    manager, store, _ = setup()

    async def scenario() -> None:
        diagnosis = diagnose(AuthError("bad key"), provider="mail")
        first = await manager.report(diagnosis, operation="op", action_taken="disabled")
        await manager.resolve(first.fingerprint, "key replaced")
        second = await manager.report(diagnosis, operation="op", action_taken="disabled")
        assert second.id != first.id

    asyncio.run(scenario())
    assert len(store.incidents) == 2


def test_resolve_provider_resolves_all_matching() -> None:
    manager, store, _ = setup()

    async def scenario() -> int:
        await manager.report(
            diagnose(TransientError("503"), provider="mail"), operation="send", action_taken="a"
        )
        await manager.report(
            diagnose(AuthError("401"), provider="mail"), operation="send", action_taken="b"
        )
        await manager.report(
            diagnose(AuthError("401"), provider="crm"), operation="send", action_taken="c"
        )
        return await manager.resolve_provider("mail", operation="send", note="recovered")

    assert asyncio.run(scenario()) == 2
    statuses = sorted(i.status.value for i in store.incidents.values())
    assert statuses == ["open", "resolved", "resolved"]
