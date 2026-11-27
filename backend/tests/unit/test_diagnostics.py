"""Unit tests for app.healing.diagnostics."""

from __future__ import annotations

import socket

import pytest

from app.core.errors import (
    AllProvidersFailedError,
    AuthError,
    PermanentError,
    ProviderUnavailableError,
    RateLimitedError,
    TimeoutExceededError,
    TransientError,
    error_from_http_status,
)
from app.core.security.crypto import CryptoError
from app.domain.actions import InvalidActionError
from app.domain.deal_state import InvalidTransitionError
from app.healing.diagnostics import Remediation, diagnose
from app.telemetry.events import Severity


@pytest.mark.parametrize(
    ("exc", "cause", "remediation", "user_action"),
    [
        (AuthError("401"), "invalid_credentials", Remediation.DISABLE_PROVIDER, True),
        (RateLimitedError("429", retry_after=5), "rate_limited", Remediation.DEFER_JOB, False),
        (ProviderUnavailableError("open"), "circuit_open", Remediation.FAILOVER, False),
        (socket.gaierror("dns"), "network_unreachable", Remediation.FAILOVER, False),
        (ConnectionRefusedError(), "network_unreachable", Remediation.FAILOVER, False),
        (TimeoutError("read"), "ambiguous_delivery", Remediation.ESCALATE, False),
        (ConnectionResetError(), "ambiguous_delivery", Remediation.ESCALATE, False),
        (TimeoutExceededError("probe"), "timeout", Remediation.RETRY, False),
        (error_from_http_status(503, provider="mail"), "provider_outage", Remediation.RETRY, False),
        (
            error_from_http_status(400, provider="mail"),
            "rejected_request",
            Remediation.ESCALATE,
            False,
        ),
        (CryptoError("tag"), "encryption_failure", Remediation.ESCALATE, True),
        (InvalidActionError("bad json"), "invalid_model_output", Remediation.REGENERATE, False),
        (InvalidTransitionError("no"), "invalid_transition", Remediation.ESCALATE, False),
        (KeyError("customer-email@x.com"), "unexpected_error", Remediation.ESCALATE, False),
    ],
)
def test_diagnosis_table(
    exc: BaseException, cause: str, remediation: Remediation, user_action: bool
) -> None:
    diagnosis = diagnose(exc, provider="mail")
    assert diagnosis.cause == cause
    assert diagnosis.remediation is remediation
    assert diagnosis.requires_user_action is user_action


def test_auth_is_critical_and_keeps_provider() -> None:
    diagnosis = diagnose(AuthError("revoked"), provider="sendgrid")
    assert diagnosis.severity is Severity.CRITICAL
    assert diagnosis.provider == "sendgrid"
    assert "sendgrid" in diagnosis.summary


def test_rate_limit_keeps_retry_after() -> None:
    assert diagnose(RateLimitedError("slow", retry_after=12)).retry_after == 12


def test_all_providers_failed_remediation_depends_on_retryability() -> None:
    retryable = AllProvidersFailedError("send", [TransientError("down", provider="a")])
    assert diagnose(retryable).remediation is Remediation.DEFER_JOB
    final = AllProvidersFailedError("send", [AuthError("bad", provider="a")])
    assert diagnose(final).remediation is Remediation.ESCALATE
    assert diagnose(final).requires_user_action


def test_unexpected_error_summary_never_leaks_exception_text() -> None:
    diagnosis = diagnose(KeyError("customer-email@x.com"))
    assert "customer-email" not in diagnosis.summary
    assert "KeyError" in diagnosis.summary


def test_permanent_without_status() -> None:
    assert diagnose(PermanentError("bad")).cause == "unexpected_error"
