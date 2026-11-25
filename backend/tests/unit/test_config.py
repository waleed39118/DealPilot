"""Unit tests for app.core.config."""

from __future__ import annotations

import base64
import secrets
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import AppMode, Environment, Settings, get_settings

VALID_KEY = base64.b64encode(secrets.token_bytes(32)).decode()
ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

LIVE_READY: dict[str, Any] = {
    "master_encryption_key": VALID_KEY,
    "app_mode": "live",
    "public_base_url": "https://api.example.com/",
    "supabase_url": "https://project.supabase.co",
    "supabase_service_role_key": "service-role",
    "database_url": "postgresql://user:pass@db:5432/postgres",
    "llm_primary_provider": "primary",
    "llm_primary_api_key": "llm-key",
    "llm_primary_model": "model-x",
    "company_legal_name": "Example LLC",
    "company_postal_address": "1 Main St, Riyadh",
    "privacy_policy_url": "https://example.com/privacy",
    "email_from_address": "Sales@Example.com",
    "resend_api_key": "re_key",
    "resend_webhook_secret": "whsec",
}


def build(**overrides: Any) -> Settings:
    """Build settings from explicit values only (ignores the real environment/.env)."""
    values: dict[str, Any] = {"master_encryption_key": VALID_KEY, **overrides}
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()


def test_defaults_are_safe() -> None:
    settings = build()
    assert settings.app_mode is AppMode.SANDBOX
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.is_live is False
    assert settings.master_key_bytes() == base64.b64decode(VALID_KEY)


def test_master_key_is_mandatory() -> None:
    with pytest.raises(ValidationError, match="master_encryption_key"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("not-base64!!", "valid base64"),
        (base64.b64encode(b"short").decode(), "32 bytes"),
    ],
)
def test_master_key_rejects_invalid_values(key: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        build(master_encryption_key=key)


def test_csv_lists_and_normalization_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_ENCRYPTION_KEY", VALID_KEY)
    monkeypatch.setenv("SANDBOX_ALLOWED_EMAILS", " Me@Example.com , ops@example.com,")
    monkeypatch.setenv("SANDBOX_ALLOWED_PHONES", "+966500000000,+12025550123")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:1420, tauri://localhost")
    settings = get_settings()
    assert settings.sandbox_allowed_emails == ["me@example.com", "ops@example.com"]
    assert settings.sandbox_allowed_phones == ["+966500000000", "+12025550123"]
    assert settings.cors_allowed_origins == ["http://localhost:1420", "tauri://localhost"]
    assert get_settings() is settings


def test_invalid_phone_rejected() -> None:
    with pytest.raises(ValidationError, match=r"E\.164"):
        build(sandbox_allowed_phones="0500000000")


def test_invalid_email_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid email"):
        build(sandbox_allowed_emails="not-an-email")


def test_blank_secrets_are_treated_as_unset() -> None:
    settings = build(supabase_service_role_key="", sendgrid_api_key="   ")
    assert settings.supabase_service_role_key is None
    assert settings.sendgrid_api_key is None


def test_live_mode_reports_every_missing_requirement() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build(app_mode="live")
    text = str(exc_info.value)
    for expected in (
        "PUBLIC_BASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "DATABASE_URL",
        "LLM_PRIMARY_API_KEY",
        "COMPANY_POSTAL_ADDRESS",
        "PRIVACY_POLICY_URL",
        "EMAIL_FROM_ADDRESS",
        "email provider key",
    ):
        assert expected in text


def test_live_mode_rejects_plain_http() -> None:
    with pytest.raises(ValidationError, match="PUBLIC_BASE_URL must be an https"):
        build(**{**LIVE_READY, "public_base_url": "http://api.example.com"})


def test_live_mode_requires_webhook_secret_for_enabled_provider() -> None:
    with pytest.raises(ValidationError, match="SENDGRID_WEBHOOK_PUBLIC_KEY"):
        build(**{**LIVE_READY, "sendgrid_api_key": "sg_key"})


def test_live_mode_starts_when_fully_configured() -> None:
    settings = build(**LIVE_READY)
    assert settings.is_live is True
    assert settings.missing_for_live() == []
    assert settings.public_base_url == "https://api.example.com"
    assert settings.email_from_address == "sales@example.com"


def test_production_forbids_wildcard_cors_and_debug_logging() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build(environment="production", cors_allowed_origins="*", log_level="DEBUG")
    text = str(exc_info.value)
    assert "CORS_ALLOWED_ORIGINS" in text
    assert "LOG_LEVEL=DEBUG" in text


def test_secrets_never_leak_in_repr() -> None:
    settings = build(**LIVE_READY)
    rendered = repr(settings) + str(settings.model_dump())
    for secret in ("service-role", "llm-key", "re_key", "whsec", VALID_KEY):
        assert secret not in rendered


def test_settings_are_immutable() -> None:
    settings = build()
    with pytest.raises(ValidationError):
        settings.app_mode = AppMode.LIVE  # type: ignore[misc]


def test_env_example_loads_in_sandbox(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        ENV_EXAMPLE.read_text(encoding="utf-8").replace(
            "MASTER_ENCRYPTION_KEY=\n", f"MASTER_ENCRYPTION_KEY={VALID_KEY}\n"
        ),
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.app_mode is AppMode.SANDBOX
    assert settings.public_base_url == ""
    assert settings.supabase_service_role_key is None
    assert settings.watchdog_interval_seconds == 60
