"""
Unified error model.

Every failure that crosses a module boundary is expressed as a ``DealPilotError`` with a
``category`` that drives recovery decisions:

* ``TRANSIENT``     — network blips, timeouts, 5xx. Safe to retry with backoff.
* ``RATE_LIMITED``  — provider throttling (HTTP 429). Retry after the advertised delay.
* ``AUTH``          — invalid/expired/revoked credentials (401/403). Never retried;
                      the provider is taken out of rotation and the user is alerted.
* ``PERMANENT``     — bad request, validation, unknown bugs. Never retried.

``may_have_succeeded`` marks *ambiguous* failures (e.g. a read timeout after the request
was sent). Retrying or failing over on those can duplicate a side effect such as a
customer message, so recovery code only does it when the caller explicitly opts in.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum


class ErrorCategory(StrEnum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    PERMANENT = "permanent"


RETRYABLE_CATEGORIES = frozenset({ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMITED})


class DealPilotError(Exception):
    """Base class for every classified error in the system."""

    default_category: ErrorCategory = ErrorCategory.PERMANENT

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory | None = None,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        may_have_succeeded: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category or self.default_category
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        self.may_have_succeeded = may_have_succeeded

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        parts.append(f"category={self.category.value}")
        return " | ".join(parts)


class TransientError(DealPilotError):
    default_category = ErrorCategory.TRANSIENT


class RateLimitedError(DealPilotError):
    default_category = ErrorCategory.RATE_LIMITED


class AuthError(DealPilotError):
    default_category = ErrorCategory.AUTH


class PermanentError(DealPilotError):
    default_category = ErrorCategory.PERMANENT


class ProviderUnavailableError(TransientError):
    """Raised without calling the provider because its circuit breaker is open."""


class AllProvidersFailedError(DealPilotError):
    """Every provider in a fallback chain failed for the same operation."""

    def __init__(self, operation: str, errors: Sequence[DealPilotError]) -> None:
        self.operation = operation
        self.errors = tuple(errors)
        retryable = any(error.retryable for error in self.errors)
        retry_hints = [error.retry_after for error in self.errors if error.retry_after]
        summary = "; ".join(f"{error.provider or '?'}: {error.message}" for error in self.errors)
        super().__init__(
            f"all providers failed for '{operation}' ({summary})",
            category=ErrorCategory.TRANSIENT if retryable else ErrorCategory.PERMANENT,
            retry_after=min(retry_hints) if retry_hints else None,
            may_have_succeeded=any(error.may_have_succeeded for error in self.errors),
        )


# ── HTTP classification ─────────────────────────────────────────────────────

_TRANSIENT_STATUSES = frozenset({408, 425, 500, 502, 503, 504})
_AMBIGUOUS_STATUSES = frozenset({504})  # upstream may have completed the request
_AUTH_STATUSES = frozenset({401, 403})


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            moment = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        seconds = (moment - (now or datetime.now(UTC))).total_seconds()
    return max(seconds, 0.0)


def classify_http_status(status_code: int) -> ErrorCategory:
    if status_code == 429:
        return ErrorCategory.RATE_LIMITED
    if status_code in _AUTH_STATUSES:
        return ErrorCategory.AUTH
    if status_code in _TRANSIENT_STATUSES or status_code >= 500:
        return ErrorCategory.TRANSIENT
    return ErrorCategory.PERMANENT


_CATEGORY_TO_CLASS: dict[ErrorCategory, type[DealPilotError]] = {
    ErrorCategory.TRANSIENT: TransientError,
    ErrorCategory.RATE_LIMITED: RateLimitedError,
    ErrorCategory.AUTH: AuthError,
    ErrorCategory.PERMANENT: PermanentError,
}


def error_from_http_status(
    status_code: int,
    *,
    provider: str,
    message: str = "",
    retry_after_header: str | None = None,
) -> DealPilotError:
    """Build the correctly typed error for a non-2xx provider response."""
    category = classify_http_status(status_code)
    error_class = _CATEGORY_TO_CLASS[category]
    return error_class(
        message or f"{provider} returned HTTP {status_code}",
        provider=provider,
        status_code=status_code,
        retry_after=parse_retry_after(retry_after_header),
        may_have_succeeded=status_code in _AMBIGUOUS_STATUSES,
    )


# ── Generic exception classification ────────────────────────────────────────


def to_dealpilot_error(exc: BaseException, *, provider: str | None = None) -> DealPilotError:
    """
    Map any exception to a ``DealPilotError``.

    Unknown exceptions are ``PERMANENT``: retrying a programming error only hides it.
    Network errors where the request provably never left (DNS failure, connection
    refused) are not ambiguous; other network errors and timeouts are.
    """
    if isinstance(exc, DealPilotError):
        if provider and exc.provider is None:
            exc.provider = provider
        return exc
    description = f"{type(exc).__name__}: {exc}".rstrip(": ")
    if isinstance(exc, (socket.gaierror, ConnectionRefusedError)):
        return TransientError(description, provider=provider, may_have_succeeded=False)
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return TransientError(description, provider=provider, may_have_succeeded=True)
    return PermanentError(description, provider=provider)
