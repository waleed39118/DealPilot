"""
Live activity feed.

Every meaningful step (project analysed, offer sent, customer asked for a discount,
provider failed over, deal won) becomes an ``ActivityEvent``. Events are persisted to
the ``activity_events`` table, which Supabase Realtime streams to the desktop and mobile
apps; events marked ``push`` also trigger a phone notification.

Events carry a stable ``code`` plus primitive ``params`` so the apps render them in
Arabic or English. ``params`` are validated: only short primitive values, and keys that
look like credentials are rejected, so a secret can never reach a phone screen.

The ``EventBus`` isolates sinks: one failing sink (e.g. push provider down) never stops
the others or the business operation that emitted the event.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Protocol
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from app.domain.models import AwareDatetime, DomainModel

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    PROJECT_ANALYZED = "project_analyzed"
    PROSPECTS_FOUND = "prospects_found"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    OFFER_SENT = "offer_sent"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    ACTION_DEFERRED = "action_deferred"
    ACTION_DENIED = "action_denied"
    STAGE_CHANGED = "stage_changed"
    HUMAN_TAKEOVER = "human_takeover"
    DEAL_WON = "deal_won"
    DEAL_LOST = "deal_lost"
    PROVIDER_FAILOVER = "provider_failover"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    INCIDENT_OPENED = "incident_opened"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_RESOLVED = "incident_resolved"


class Severity(StrEnum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CRITICAL = "critical"


# Event types that always reach the phone, whatever their severity.
_ALWAYS_PUSH = frozenset(
    {
        EventType.APPROVAL_REQUESTED,
        EventType.DEAL_WON,
        EventType.HUMAN_TAKEOVER,
        EventType.INCIDENT_OPENED,
        EventType.INCIDENT_RESOLVED,
    }
)
_FORBIDDEN_PARAM_FRAGMENTS = ("key", "token", "secret", "password", "credential", "auth")

ParamValue = Annotated[str, Field(max_length=500)] | int | float | bool


class ActivityEvent(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = Field(
        default=None, description="None = platform-level event (e.g. provider outage)."
    )
    type: EventType
    severity: Severity = Severity.INFO
    code: str = Field(pattern=r"^[a-z0-9_.]{1,80}$")
    summary: str = Field(min_length=1, max_length=500)
    params: dict[str, ParamValue] = Field(default_factory=dict, max_length=20)
    project_id: UUID | None = None
    deal_id: UUID | None = None
    push: bool = False
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("params")
    @classmethod
    def _safe_params(cls, value: dict[str, ParamValue]) -> dict[str, ParamValue]:
        for name in value:
            if len(name) > 40 or not name.replace("_", "").isalnum():
                raise ValueError(f"invalid parameter name '{name[:40]}'")
            if any(fragment in name.lower() for fragment in _FORBIDDEN_PARAM_FRAGMENTS):
                raise ValueError(f"parameter '{name}' looks like a credential")
        return value

    @property
    def should_push(self) -> bool:
        return self.push or self.type in _ALWAYS_PUSH or self.severity is Severity.CRITICAL


class EventSink(Protocol):
    name: str

    async def publish(self, event: ActivityEvent) -> None: ...


class EventBus:
    def __init__(self, sinks: Sequence[EventSink] = ()) -> None:
        self._sinks = list(sinks)

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    async def publish(self, event: ActivityEvent) -> list[str]:
        """Deliver to every sink; returns the names of sinks that failed."""
        failed: list[str] = []
        for sink in self._sinks:
            try:
                await sink.publish(event)
            except Exception:
                logger.exception("event sink '%s' failed for event %s", sink.name, event.id)
                failed.append(sink.name)
        return failed


class InMemorySink:
    """Keeps events in memory: tests, local development, and the watchdog's own view."""

    def __init__(self, name: str = "memory", capacity: int = 1000) -> None:
        self.name = name
        self._capacity = capacity
        self.events: list[ActivityEvent] = []

    async def publish(self, event: ActivityEvent) -> None:
        self.events.append(event)
        if len(self.events) > self._capacity:
            del self.events[: len(self.events) - self._capacity]

    def of_type(self, event_type: EventType) -> list[ActivityEvent]:
        return [event for event in self.events if event.type is event_type]
