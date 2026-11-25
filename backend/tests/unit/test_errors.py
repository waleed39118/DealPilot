"""Unit tests for app.core.errors."""

from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest

from app.core.errors import (
    AllProvidersFailedError,
    AuthError,
    ErrorCategory,
    PermanentError,
    RateLimitedError,
    TransientError,
    classify_http_status,
    error_from_http_status,
    parse_retry_after,
    to_dealpilot_error,
)


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (429, ErrorCategory.RATE_LIMITED),
        (401, ErrorCategory.AUTH),
        (403, ErrorCategory.AUTH),
        (408, ErrorCategory.TRANSIENT),
        (500, ErrorCategory.TRANSIENT),
        (503, ErrorCategory.TRANSIENT),
        (599, ErrorCategory.TRANSIENT),
        (400, ErrorCategory.PERMANENT),
        (404, ErrorCategory.PERMANENT),
        (422, ErrorCategory.PERMANENT),
    ],
)
def test_classify_http_status(status: int, category: ErrorCategory) -> None:
    assert classify_http_status(status) is category


def test_parse_retry_after_variants() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after("  ") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("-5") == 0.0
    assert parse_retry_after("Thu, 01 Jan 2026 12:00:30 GMT", now=now) == 30.0


def test_error_from_http_status_types_and_metadata() -> None:
    limited = error_from_http_status(429, provider="mail", retry_after_header="7")
    assert isinstance(limited, RateLimitedError)
    assert limited.retry_after == 7.0
    assert limited.retryable

    auth = error_from_http_status(401, provider="mail")
    assert isinstance(auth, AuthError)
    assert not auth.retryable

    gateway = error_from_http_status(504, provider="mail")
    assert isinstance(gateway, TransientError)
    assert gateway.may_have_succeeded

    bad = error_from_http_status(400, provider="mail", message="invalid recipient")
    assert isinstance(bad, PermanentError)
    assert "invalid recipient" in str(bad)
    assert "status=400" in str(bad)


def test_to_dealpilot_error_mapping() -> None:
    dns = to_dealpilot_error(socket.gaierror("no host"), provider="x")
    assert isinstance(dns, TransientError)
    assert not dns.may_have_succeeded

    refused = to_dealpilot_error(ConnectionRefusedError())
    assert not refused.may_have_succeeded

    timeout = to_dealpilot_error(TimeoutError("read timed out"))
    assert isinstance(timeout, TransientError)
    assert timeout.may_have_succeeded

    reset = to_dealpilot_error(ConnectionResetError())
    assert reset.may_have_succeeded

    bug = to_dealpilot_error(KeyError("missing"))
    assert isinstance(bug, PermanentError)

    original = AuthError("revoked")
    same = to_dealpilot_error(original, provider="mail")
    assert same is original
    assert same.provider == "mail"


def test_all_providers_failed_category() -> None:
    transient = AllProvidersFailedError(
        "send",
        [TransientError("down", provider="a", retry_after=9), AuthError("bad", provider="b")],
    )
    assert transient.category is ErrorCategory.TRANSIENT
    assert transient.retry_after == 9
    assert "a: down" in str(transient)

    permanent = AllProvidersFailedError("send", [AuthError("bad", provider="a")])
    assert permanent.category is ErrorCategory.PERMANENT
