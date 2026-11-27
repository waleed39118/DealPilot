"""
Watchdog: periodic health probes with automatic remediation.

Each ``HealthCheck`` probes one dependency (database, job queue, a provider, credential
validity). Every cycle the watchdog runs all checks concurrently, each with its own
timeout so a hanging dependency cannot stall the loop.

* A check must fail ``failure_threshold`` cycles in a row before an incident is opened
  (avoids alert storms from one-off blips).
* A failing check may expose ``remediate()`` (reconnect a pool, re-queue stuck jobs);
  the watchdog runs it and records the outcome in the incident.
* When a failing check passes again, its incident is resolved and the user is told.

The watchdog never raises: a bug inside a check is itself reported as an incident.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.core.errors import TimeoutExceededError
from app.healing.diagnostics import diagnose
from app.healing.incidents import IncidentManager, fingerprint_for

logger = logging.getLogger(__name__)


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # failing, below the alert threshold
    DOWN = "down"  # failing at or above the alert threshold


@runtime_checkable
class HealthCheck(Protocol):
    name: str

    async def check(self) -> None:
        """Return normally when healthy; raise (ideally a DealPilotError) when not."""
        ...


@runtime_checkable
class RemediableCheck(HealthCheck, Protocol):
    async def remediate(self) -> str:
        """Try to fix the dependency; return a short description of what was done."""
        ...


@dataclass(frozen=True, slots=True)
class CheckStatus:
    name: str
    state: HealthState
    consecutive_failures: int
    last_checked: datetime | None
    last_latency_ms: float | None
    last_error: str | None


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


_OPERATION = "health_check"


class Watchdog:
    def __init__(
        self,
        checks: Sequence[HealthCheck],
        incidents: IncidentManager,
        *,
        interval_seconds: float = 60.0,
        check_timeout_seconds: float = 10.0,
        failure_threshold: int = 2,
        clock: Clock = _utcnow,
    ) -> None:
        names = [check.name for check in checks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate health check names: {names}")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._checks = list(checks)
        self._incidents = incidents
        self._interval = interval_seconds
        self._timeout = check_timeout_seconds
        self._threshold = failure_threshold
        self._clock = clock
        self._status: dict[str, CheckStatus] = {
            check.name: CheckStatus(check.name, HealthState.HEALTHY, 0, None, None, None)
            for check in self._checks
        }
        self._open_fingerprints: dict[str, set[str]] = {}

    # ── Public API ──────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, CheckStatus]:
        """Current health of every dependency (for the dashboard)."""
        return dict(self._status)

    @property
    def overall(self) -> HealthState:
        states = {status.state for status in self._status.values()}
        if HealthState.DOWN in states:
            return HealthState.DOWN
        if HealthState.DEGRADED in states:
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    async def run_once(self) -> dict[str, CheckStatus]:
        await asyncio.gather(*(self._run_check(check) for check in self._checks))
        return self.snapshot()

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.run_once()
            except Exception:  # pragma: no cover - defensive; run_once already isolates
                logger.exception("watchdog cycle failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    # ── Internals ───────────────────────────────────────────────────────────
    async def _run_check(self, check: HealthCheck) -> None:
        started = time.perf_counter()
        try:
            await asyncio.wait_for(check.check(), timeout=self._timeout)
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            if isinstance(exc, TimeoutError):
                # The probe has no side effect, so its timeout is not ambiguous.
                exc = TimeoutExceededError(
                    f"health check timed out after {self._timeout:g}s", provider=check.name
                )
            await self._on_failure(check, exc, latency)
        else:
            latency = (time.perf_counter() - started) * 1000
            await self._on_success(check, latency)

    async def _on_success(self, check: HealthCheck, latency_ms: float) -> None:
        previous = self._status[check.name]
        self._status[check.name] = CheckStatus(
            check.name, HealthState.HEALTHY, 0, self._clock(), latency_ms, None
        )
        fingerprints = self._open_fingerprints.pop(check.name, set())
        if previous.state is HealthState.HEALTHY:
            return
        note = f"{check.name} healthy again after {previous.consecutive_failures} failed checks"
        for fingerprint in sorted(fingerprints):
            await self._incidents.resolve(fingerprint, note)

    async def _on_failure(self, check: HealthCheck, exc: BaseException, latency_ms: float) -> None:
        diagnosis = diagnose(exc, provider=check.name)
        failures = self._status[check.name].consecutive_failures + 1
        state = HealthState.DOWN if failures >= self._threshold else HealthState.DEGRADED
        self._status[check.name] = CheckStatus(
            check.name, state, failures, self._clock(), latency_ms, diagnosis.summary
        )
        if state is not HealthState.DOWN:
            return

        action = await self._try_remediation(check)
        incident = await self._incidents.report(
            diagnosis, operation=_OPERATION, action_taken=action
        )
        self._open_fingerprints.setdefault(check.name, set()).add(
            fingerprint_for(diagnosis.cause, diagnosis.provider, _OPERATION)
        )
        logger.warning(
            "health check '%s' down (%s, %d failures, incident %s)",
            check.name,
            diagnosis.cause,
            failures,
            incident.id,
        )

    async def _try_remediation(self, check: HealthCheck) -> str:
        if not isinstance(check, RemediableCheck):
            return "monitoring; no automatic remediation available"
        try:
            result = await asyncio.wait_for(check.remediate(), timeout=self._timeout)
        except Exception as exc:
            logger.exception("remediation for '%s' failed", check.name)
            return f"automatic remediation failed ({type(exc).__name__})"
        return f"automatic remediation: {result}"
