"""
Incident tracking: what went wrong, what the system did about it, and whether it is fixed.

Incidents are de-duplicated by fingerprint (cause + provider + operation): a provider
failing a hundred times opens one incident with a hundred occurrences, and the phone
receives one notification when it opens and one when it is resolved, not a hundred.

The ``*_handler`` factories plug the resilience layer (retry, circuit breaker, fallback
chain) into incidents and the live feed, so recovery is visible to the user:
"SendGrid unreachable → switched to Resend" and later "SendGrid recovered".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from app.core.resilience.circuit_breaker import (
    CircuitState,
    CircuitStateChange,
    StateChangeCallback,
)
from app.core.resilience.fallback import FallbackCallback, FallbackEvent
from app.healing.diagnostics import Diagnosis, diagnose
from app.telemetry.events import ActivityEvent, EventBus, EventType, Severity

_MAX_ACTIONS = 20
_SEVERITY_RANK = {Severity.INFO: 0, Severity.SUCCESS: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


class IncidentStatus(StrEnum):
    OPEN = "open"
    MITIGATED = "mitigated"  # automatic workaround in place (e.g. failover)
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class Incident:
    fingerprint: str
    cause: str
    summary: str
    severity: Severity
    requires_user_action: bool
    provider: str | None
    operation: str
    tenant_id: UUID | None
    first_seen: datetime
    last_seen: datetime
    id: UUID = field(default_factory=uuid4)
    status: IncidentStatus = IncidentStatus.OPEN
    occurrences: int = 1
    actions_taken: tuple[str, ...] = ()
    resolution: str | None = None


class IncidentStore(Protocol):
    async def get_active(self, fingerprint: str) -> Incident | None: ...

    async def list_active(self, *, provider: str, operation: str) -> list[Incident]: ...

    async def save(self, incident: Incident) -> None: ...


class InMemoryIncidentStore:
    def __init__(self) -> None:
        self.incidents: dict[UUID, Incident] = {}

    async def get_active(self, fingerprint: str) -> Incident | None:
        for incident in self.incidents.values():
            if (
                incident.fingerprint == fingerprint
                and incident.status is not IncidentStatus.RESOLVED
            ):
                return incident
        return None

    async def list_active(self, *, provider: str, operation: str) -> list[Incident]:
        return [
            incident
            for incident in self.incidents.values()
            if incident.provider == provider
            and incident.operation == operation
            and incident.status is not IncidentStatus.RESOLVED
        ]

    async def save(self, incident: Incident) -> None:
        self.incidents[incident.id] = incident


def fingerprint_for(cause: str, provider: str | None, operation: str) -> str:
    return f"{cause}|{provider or '-'}|{operation}"


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class IncidentManager:
    def __init__(self, store: IncidentStore, bus: EventBus, *, clock: Clock = _utcnow) -> None:
        self._store = store
        self._bus = bus
        self._clock = clock

    async def report(
        self,
        diagnosis: Diagnosis,
        *,
        operation: str,
        action_taken: str,
        tenant_id: UUID | None = None,
    ) -> Incident:
        """Record a failure. Opens a new incident or updates the active one."""
        now = self._clock()
        fingerprint = fingerprint_for(diagnosis.cause, diagnosis.provider, operation)
        existing = await self._store.get_active(fingerprint)

        if existing is None:
            incident = Incident(
                fingerprint=fingerprint,
                cause=diagnosis.cause,
                summary=diagnosis.summary,
                severity=diagnosis.severity,
                requires_user_action=diagnosis.requires_user_action,
                provider=diagnosis.provider,
                operation=operation,
                tenant_id=tenant_id,
                first_seen=now,
                last_seen=now,
                actions_taken=(action_taken,),
            )
            await self._store.save(incident)
            await self._emit(incident, EventType.INCIDENT_OPENED, action_taken)
            return incident

        escalated = _SEVERITY_RANK[diagnosis.severity] > _SEVERITY_RANK[existing.severity]
        incident = replace(
            existing,
            last_seen=now,
            occurrences=existing.occurrences + 1,
            severity=diagnosis.severity if escalated else existing.severity,
            requires_user_action=existing.requires_user_action or diagnosis.requires_user_action,
            status=IncidentStatus.OPEN if escalated else existing.status,
            actions_taken=_append_action(existing.actions_taken, action_taken),
        )
        await self._store.save(incident)
        if escalated:
            await self._emit(incident, EventType.INCIDENT_UPDATED, action_taken, push=True)
        return incident

    async def mitigate(self, fingerprint: str, note: str) -> Incident | None:
        """An automatic workaround is in place; the root cause still exists."""
        existing = await self._store.get_active(fingerprint)
        if existing is None or existing.status is IncidentStatus.MITIGATED:
            return existing
        incident = replace(
            existing,
            status=IncidentStatus.MITIGATED,
            last_seen=self._clock(),
            actions_taken=_append_action(existing.actions_taken, note),
        )
        await self._store.save(incident)
        await self._emit(incident, EventType.INCIDENT_UPDATED, note)
        return incident

    async def resolve(self, fingerprint: str, note: str) -> Incident | None:
        existing = await self._store.get_active(fingerprint)
        if existing is None:
            return None
        incident = replace(
            existing,
            status=IncidentStatus.RESOLVED,
            last_seen=self._clock(),
            resolution=note,
            actions_taken=_append_action(existing.actions_taken, note),
        )
        await self._store.save(incident)
        await self._emit(incident, EventType.INCIDENT_RESOLVED, note, severity=Severity.SUCCESS)
        return incident

    async def resolve_provider(self, provider: str, *, operation: str, note: str) -> int:
        """Resolve every active incident of a provider/operation; returns how many."""
        active = await self._store.list_active(provider=provider, operation=operation)
        for incident in active:
            await self.resolve(incident.fingerprint, note)
        return len(active)

    async def _emit(
        self,
        incident: Incident,
        event_type: EventType,
        action: str,
        *,
        severity: Severity | None = None,
        push: bool = False,
    ) -> None:
        await self._bus.publish(
            ActivityEvent(
                tenant_id=incident.tenant_id,
                type=event_type,
                severity=severity or incident.severity,
                code=f"incident.{incident.cause}",
                summary=f"{incident.summary} — {action}"[:500],
                params={
                    "incident_id": str(incident.id),
                    "provider": incident.provider or "",
                    "operation": incident.operation,
                    "status": incident.status.value,
                    "occurrences": incident.occurrences,
                    "requires_user_action": incident.requires_user_action,
                    "action": action[:500],
                },
                push=push,
            )
        )


def _append_action(actions: tuple[str, ...], action: str) -> tuple[str, ...]:
    return (*actions, action)[-_MAX_ACTIONS:]


# ── Resilience-layer hooks ──────────────────────────────────────────────────


def fallback_handler(
    manager: IncidentManager, *, tenant_id: UUID | None = None
) -> FallbackCallback:
    """``on_fallback`` for ``FallbackChain``: record the failure and the switch."""

    async def handle(event: FallbackEvent) -> None:
        diagnosis = diagnose(event.error, provider=event.failed_provider)
        incident = await manager.report(
            diagnosis,
            operation=event.operation,
            action_taken=f"switched from {event.failed_provider} to {event.next_provider}",
            tenant_id=tenant_id,
        )
        if not diagnosis.requires_user_action:
            await manager.mitigate(
                incident.fingerprint, f"traffic now served by {event.next_provider}"
            )

    return handle


def circuit_handler(
    manager: IncidentManager, *, operation: str, tenant_id: UUID | None = None
) -> StateChangeCallback:
    """``on_state_change`` for ``CircuitBreaker``: resolve incidents when it closes."""

    async def handle(change: CircuitStateChange) -> None:
        if change.current is CircuitState.CLOSED:
            await manager.resolve_provider(
                change.name,
                operation=operation,
                note=f"{change.name} recovered ({change.reason})",
            )

    return handle
