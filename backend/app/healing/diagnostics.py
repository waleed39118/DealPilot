"""
Error diagnosis.

Turns any exception into a ``Diagnosis``: the probable cause, how serious it is, the
remediation the system applies automatically, and whether the user must act (e.g.
replace a revoked API key). Cause codes are stable so the apps can explain them in
Arabic or English.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.ai.llm.base import ModelRefusalError, StructuredOutputError
from app.core.errors import (
    AllProvidersFailedError,
    AuthError,
    DealPilotError,
    ErrorCategory,
    NetworkError,
    ProviderUnavailableError,
    RateLimitedError,
    TimeoutExceededError,
    to_dealpilot_error,
)
from app.core.security.crypto import CryptoError
from app.domain.actions import InvalidActionError
from app.domain.deal_state import InvalidTransitionError
from app.telemetry.events import Severity


class Remediation(StrEnum):
    RETRY = "retry"  # retry with backoff on the same provider
    FAILOVER = "failover"  # switch to the next provider in the chain
    DEFER_JOB = "defer_job"  # hand the job back to the queue for later
    REGENERATE = "regenerate"  # ask the model again (invalid structured output)
    DISABLE_PROVIDER = "disable_provider"  # stop using it until the user fixes it
    ESCALATE = "escalate"  # no safe automatic fix: a person must look


@dataclass(frozen=True, slots=True)
class Diagnosis:
    cause: str
    summary: str
    severity: Severity
    remediation: Remediation
    requires_user_action: bool
    provider: str | None
    retry_after: float | None
    error: DealPilotError


def diagnose(exc: BaseException, *, provider: str | None = None) -> Diagnosis:
    error = to_dealpilot_error(exc, provider=provider)
    cause, summary, severity, remediation, user_action = _classify(exc, error)
    return Diagnosis(
        cause=cause,
        summary=summary,
        severity=severity,
        remediation=remediation,
        requires_user_action=user_action,
        provider=error.provider,
        retry_after=error.retry_after,
        error=error,
    )


def _classify(
    original: BaseException, error: DealPilotError
) -> tuple[str, str, Severity, Remediation, bool]:
    name = error.provider or "provider"
    if isinstance(error, AuthError):
        return (
            "invalid_credentials",
            f"{name} rejected the credentials (invalid, expired or revoked key)",
            Severity.CRITICAL,
            Remediation.DISABLE_PROVIDER,
            True,
        )
    if isinstance(error, RateLimitedError):
        return (
            "rate_limited",
            f"{name} is throttling requests",
            Severity.WARNING,
            Remediation.DEFER_JOB,
            False,
        )
    if isinstance(error, ProviderUnavailableError):
        return (
            "circuit_open",
            f"{name} is temporarily taken out of rotation after repeated failures",
            Severity.WARNING,
            Remediation.FAILOVER,
            False,
        )
    if isinstance(error, AllProvidersFailedError):
        return (
            "all_providers_failed",
            str(error.message),
            Severity.CRITICAL,
            Remediation.DEFER_JOB if error.retryable else Remediation.ESCALATE,
            not error.retryable,
        )
    if isinstance(error, CryptoError):
        return (
            "encryption_failure",
            "stored data could not be decrypted or encrypted",
            Severity.CRITICAL,
            Remediation.ESCALATE,
            True,
        )
    if isinstance(error, ModelRefusalError):
        return (
            "model_refused",
            "the language model declined the request",
            Severity.WARNING,
            Remediation.ESCALATE,
            False,
        )
    if isinstance(error, (InvalidActionError, StructuredOutputError)):
        return (
            "invalid_model_output",
            "the language model returned an invalid action",
            Severity.WARNING,
            Remediation.REGENERATE,
            False,
        )
    if isinstance(error, InvalidTransitionError):
        return (
            "invalid_transition",
            error.message,
            Severity.WARNING,
            Remediation.ESCALATE,
            False,
        )
    if error.category is ErrorCategory.TRANSIENT:
        if error.may_have_succeeded:
            return (
                "ambiguous_delivery",
                f"{name} did not confirm the request; it may or may not have been executed",
                Severity.WARNING,
                Remediation.ESCALATE,
                False,
            )
        if isinstance(error, TimeoutExceededError):
            return (
                "timeout",
                f"{name} did not respond in time",
                Severity.WARNING,
                Remediation.RETRY,
                False,
            )
        if isinstance(error, NetworkError):
            return (
                "network_unreachable",
                f"{name} could not be reached (network or DNS failure)",
                Severity.WARNING,
                Remediation.FAILOVER,
                False,
            )
        return (
            "provider_outage",
            f"{name} is failing temporarily",
            Severity.WARNING,
            Remediation.RETRY,
            False,
        )
    if error.status_code is not None:
        return (
            "rejected_request",
            f"{name} rejected the request (HTTP {error.status_code})",
            Severity.WARNING,
            Remediation.ESCALATE,
            False,
        )
    return (
        "unexpected_error",
        # Only the type: exception text may contain customer data and reaches phones.
        f"unexpected internal error ({type(original).__name__})",
        Severity.CRITICAL,
        Remediation.ESCALATE,
        False,
    )
