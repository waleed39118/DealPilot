"""
Central application configuration.

Single source of truth for every runtime setting and secret. Values are read from
environment variables (or a local ``.env`` file in development) and validated once
at startup, so a misconfigured deployment fails fast instead of failing mid-deal.

Security rules enforced here:
* Every secret is a ``SecretStr`` and never appears in ``repr``/logs.
* The encryption master key is mandatory and must decode to exactly 32 bytes (AES-256).
* ``APP_MODE=sandbox`` is the default; ``live`` refuses to start unless every
  compliance- and security-critical setting is present.
"""

from __future__ import annotations

import base64
import binascii
import re
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_AES_256_KEY_BYTES = 32


class AppMode(StrEnum):
    """Outbound safety mode. ``SANDBOX`` only ever contacts allow-listed recipients."""

    SANDBOX = "sandbox"
    LIVE = "live"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


CommaList = Annotated[list[str], NoDecode]


def _split_csv(value: object) -> object:
    """Accept ``"a, b ,c"`` from the environment as well as a real list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ── Runtime ─────────────────────────────────────────────────────────────
    app_mode: AppMode = AppMode.SANDBOX
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    public_base_url: str = Field(
        default="",
        description="Public HTTPS URL of this backend; used to build webhook callback URLs.",
    )
    cors_allowed_origins: CommaList = Field(default_factory=list)

    # ── Database / Realtime ─────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_service_role_key: SecretStr | None = None
    database_url: SecretStr | None = Field(
        default=None, description="Direct Postgres connection string (job queue)."
    )

    # ── Encryption at rest (AES-256-GCM envelope encryption) ────────────────
    master_encryption_key: SecretStr = Field(
        description="Base64-encoded 32-byte key-encryption key (KEK)."
    )
    master_encryption_key_version: int = Field(default=1, ge=1)

    # ── Language model providers ────────────────────────────────────────────
    llm_primary_provider: str = ""
    llm_primary_api_key: SecretStr | None = None
    llm_primary_model: str = ""
    llm_primary_fast_model: str = ""
    llm_fallback_provider: str = ""
    llm_fallback_api_key: SecretStr | None = None
    llm_fallback_model: str = ""

    # ── Sender identity & legal footer (CAN-SPAM / GDPR) ────────────────────
    company_legal_name: str = ""
    company_postal_address: str = ""
    privacy_policy_url: str = ""

    # ── Email ───────────────────────────────────────────────────────────────
    email_from_address: str = ""
    email_from_name: str = ""
    sendgrid_api_key: SecretStr | None = None
    sendgrid_webhook_public_key: SecretStr | None = None
    resend_api_key: SecretStr | None = None
    resend_webhook_secret: SecretStr | None = None

    # ── Twilio (voice + WhatsApp) ───────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: SecretStr | None = None
    twilio_voice_number: str = ""
    twilio_whatsapp_sender: str = ""

    # ── Fax ─────────────────────────────────────────────────────────────────
    fax_provider: str = ""
    fax_api_key: SecretStr | None = None

    # ── Licensed prospect-data providers ────────────────────────────────────
    apollo_api_key: SecretStr | None = None
    pdl_api_key: SecretStr | None = None
    hunter_api_key: SecretStr | None = None

    # ── Push notifications ──────────────────────────────────────────────────
    expo_access_token: SecretStr | None = None

    # ── Sandbox allow-lists ─────────────────────────────────────────────────
    sandbox_allowed_emails: CommaList = Field(default_factory=list)
    sandbox_allowed_phones: CommaList = Field(default_factory=list)

    # ── Self-healing ────────────────────────────────────────────────────────
    watchdog_interval_seconds: int = Field(default=60, ge=10, le=3600)

    # ── Field validation ────────────────────────────────────────────────────
    @field_validator(
        "supabase_service_role_key",
        "database_url",
        "llm_primary_api_key",
        "llm_fallback_api_key",
        "sendgrid_api_key",
        "sendgrid_webhook_public_key",
        "resend_api_key",
        "resend_webhook_secret",
        "twilio_auth_token",
        "fax_api_key",
        "apollo_api_key",
        "pdl_api_key",
        "hunter_api_key",
        "expo_access_token",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_unset(cls, value: object) -> object:
        """``KEY=`` in a .env file means "not configured", not an empty secret."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "cors_allowed_origins",
        "sandbox_allowed_emails",
        "sandbox_allowed_phones",
        mode="before",
    )
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("sandbox_allowed_emails")
    @classmethod
    def _validate_emails(cls, value: list[str]) -> list[str]:
        normalized = [email.lower() for email in value]
        invalid = [email for email in normalized if not _EMAIL_RE.match(email)]
        if invalid:
            raise ValueError(f"invalid email address(es): {', '.join(invalid)}")
        return normalized

    @field_validator("sandbox_allowed_phones", "twilio_voice_number")
    @classmethod
    def _validate_e164(cls, value: list[str] | str) -> list[str] | str:
        numbers = value if isinstance(value, list) else [value] if value else []
        invalid = [number for number in numbers if not _E164_RE.match(number)]
        if invalid:
            raise ValueError(
                f"phone number(s) must be in E.164 format (e.g. +9665XXXXXXXX): "
                f"{', '.join(invalid)}"
            )
        return value

    @field_validator("email_from_address")
    @classmethod
    def _validate_from_address(cls, value: str) -> str:
        if value and not _EMAIL_RE.match(value):
            raise ValueError("email_from_address is not a valid email address")
        return value.lower()

    @field_validator("public_base_url", "supabase_url", "privacy_policy_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("master_encryption_key")
    @classmethod
    def _validate_master_key(cls, value: SecretStr) -> SecretStr:
        try:
            raw = base64.b64decode(value.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("master_encryption_key must be valid base64") from exc
        if len(raw) != _AES_256_KEY_BYTES:
            raise ValueError(
                f"master_encryption_key must decode to {_AES_256_KEY_BYTES} bytes (got {len(raw)})"
            )
        return value

    # ── Cross-field validation ──────────────────────────────────────────────
    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> Self:
        problems = self.missing_for_live() if self.app_mode is AppMode.LIVE else []
        if self.environment is Environment.PRODUCTION:
            problems.extend(self._production_problems())
        if problems:
            raise ValueError("configuration is not safe to start:\n- " + "\n- ".join(problems))
        return self

    def missing_for_live(self) -> list[str]:
        """Return every unmet requirement for ``APP_MODE=live`` (empty list = ready)."""
        problems: list[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        require(
            self.public_base_url.startswith("https://"), "PUBLIC_BASE_URL must be an https:// URL"
        )
        require(self.supabase_url.startswith("https://"), "SUPABASE_URL must be an https:// URL")
        require(self.supabase_service_role_key is not None, "SUPABASE_SERVICE_ROLE_KEY is required")
        require(self.database_url is not None, "DATABASE_URL is required")
        require(
            bool(self.llm_primary_provider and self.llm_primary_model),
            "LLM_PRIMARY_PROVIDER and LLM_PRIMARY_MODEL are required",
        )
        require(self.llm_primary_api_key is not None, "LLM_PRIMARY_API_KEY is required")
        require(bool(self.company_legal_name), "COMPANY_LEGAL_NAME is required")
        require(bool(self.company_postal_address), "COMPANY_POSTAL_ADDRESS is required (CAN-SPAM)")
        require(bool(self.privacy_policy_url), "PRIVACY_POLICY_URL is required (GDPR)")
        require(bool(self.email_from_address), "EMAIL_FROM_ADDRESS is required")
        require(
            self.sendgrid_api_key is not None or self.resend_api_key is not None,
            "at least one email provider key (SENDGRID_API_KEY or RESEND_API_KEY) is required",
        )
        require(
            self.sendgrid_api_key is None or self.sendgrid_webhook_public_key is not None,
            "SENDGRID_WEBHOOK_PUBLIC_KEY is required when SendGrid is enabled",
        )
        require(
            self.resend_api_key is None or self.resend_webhook_secret is not None,
            "RESEND_WEBHOOK_SECRET is required when Resend is enabled",
        )
        require(
            not self.twilio_account_sid or self.twilio_auth_token is not None,
            "TWILIO_AUTH_TOKEN is required when TWILIO_ACCOUNT_SID is set",
        )
        return problems

    def _production_problems(self) -> list[str]:
        problems: list[str] = []
        if "*" in self.cors_allowed_origins:
            problems.append("CORS_ALLOWED_ORIGINS must not contain '*' in production")
        if self.log_level is LogLevel.DEBUG:
            problems.append("LOG_LEVEL=DEBUG is not allowed in production")
        return problems

    # ── Convenience accessors ───────────────────────────────────────────────
    @property
    def is_live(self) -> bool:
        return self.app_mode is AppMode.LIVE

    def master_key_bytes(self) -> bytes:
        """Decoded 32-byte KEK. Call only where the key is actually used."""
        return base64.b64decode(self.master_encryption_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (validated once)."""
    return Settings()
