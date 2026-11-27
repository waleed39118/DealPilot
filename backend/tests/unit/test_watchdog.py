"""Unit tests for app.healing.watchdog."""

from __future__ import annotations

import asyncio

import pytest

from app.core.errors import AuthError, TransientError
from app.healing.incidents import IncidentManager, IncidentStatus, InMemoryIncidentStore
from app.healing.watchdog import HealthState, Watchdog
from app.telemetry.events import EventBus, EventType, InMemorySink


class ScriptedCheck:
    """Fails with the queued exceptions, then passes."""

    def __init__(self, name: str, *failures: BaseException) -> None:
        self.name = name
        self.failures = list(failures)
        self.calls = 0

    async def check(self) -> None:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)


class RemediableQueueCheck(ScriptedCheck):
    def __init__(self, *failures: BaseException) -> None:
        super().__init__("job_queue", *failures)
        self.remediations = 0

    async def remediate(self) -> str:
        self.remediations += 1
        return "re-queued 3 stuck jobs"


class HangingCheck:
    name = "slow_provider"

    async def check(self) -> None:
        await asyncio.Event().wait()


def make(*checks: object, threshold: int = 2, timeout: float = 1.0):  # type: ignore[no-untyped-def]
    store, sink = InMemoryIncidentStore(), InMemorySink()
    watchdog = Watchdog(
        checks,  # type: ignore[arg-type]
        IncidentManager(store, EventBus([sink])),
        failure_threshold=threshold,
        check_timeout_seconds=timeout,
    )
    return watchdog, store, sink


def test_validation() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make(ScriptedCheck("db"), ScriptedCheck("db"))
    with pytest.raises(ValueError, match="failure_threshold"):
        make(ScriptedCheck("db"), threshold=0)


def test_healthy_checks() -> None:
    watchdog, store, _ = make(ScriptedCheck("db"))
    snapshot = asyncio.run(watchdog.run_once())
    assert snapshot["db"].state is HealthState.HEALTHY
    assert snapshot["db"].last_latency_ms is not None
    assert watchdog.overall is HealthState.HEALTHY
    assert store.incidents == {}


def test_single_blip_does_not_alert() -> None:
    watchdog, store, _ = make(ScriptedCheck("db", TransientError("blip")))

    async def scenario() -> None:
        await watchdog.run_once()
        assert watchdog.overall is HealthState.DEGRADED
        await watchdog.run_once()

    asyncio.run(scenario())
    assert store.incidents == {}
    assert watchdog.overall is HealthState.HEALTHY


def test_down_opens_incident_then_recovery_resolves_it() -> None:
    check = ScriptedCheck(
        "db", TransientError("down"), TransientError("down"), TransientError("down")
    )
    watchdog, store, sink = make(check)

    async def scenario() -> None:
        for _ in range(4):
            await watchdog.run_once()

    asyncio.run(scenario())
    (incident,) = store.incidents.values()
    assert incident.status is IncidentStatus.RESOLVED
    assert incident.occurrences == 2
    assert len(sink.of_type(EventType.INCIDENT_OPENED)) == 1
    assert len(sink.of_type(EventType.INCIDENT_RESOLVED)) == 1


def test_changing_causes_are_all_resolved() -> None:
    check = ScriptedCheck("mail", TransientError("down"), TransientError("down"), AuthError("401"))
    watchdog, store, _ = make(check)

    async def scenario() -> None:
        for _ in range(4):
            await watchdog.run_once()

    asyncio.run(scenario())
    assert len(store.incidents) == 2
    assert all(i.status is IncidentStatus.RESOLVED for i in store.incidents.values())


def test_remediation_runs_and_is_recorded() -> None:
    check = RemediableQueueCheck(TransientError("stuck"), TransientError("stuck"))
    watchdog, store, _ = make(check)

    async def scenario() -> None:
        await watchdog.run_once()
        await watchdog.run_once()

    asyncio.run(scenario())
    assert check.remediations == 1
    (incident,) = store.incidents.values()
    assert "re-queued 3 stuck jobs" in incident.actions_taken[0]


def test_hanging_check_times_out_without_blocking_others() -> None:
    fast = ScriptedCheck("db")
    watchdog, _store, _ = make(HangingCheck(), fast, threshold=1, timeout=0.05)
    snapshot = asyncio.run(watchdog.run_once())
    assert snapshot["slow_provider"].state is HealthState.DOWN
    assert "did not respond in time" in (snapshot["slow_provider"].last_error or "")
    assert snapshot["db"].state is HealthState.HEALTHY
    assert watchdog.overall is HealthState.DOWN


def test_buggy_check_is_reported_not_raised() -> None:
    watchdog, store, _ = make(ScriptedCheck("crm", ZeroDivisionError()), threshold=1)
    asyncio.run(watchdog.run_once())
    (incident,) = store.incidents.values()
    assert incident.cause == "unexpected_error"


def test_run_forever_stops_on_event() -> None:
    check = ScriptedCheck("db")
    watchdog, _, _ = make(check)

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(watchdog.run_forever(stop))
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())
    assert check.calls >= 1
