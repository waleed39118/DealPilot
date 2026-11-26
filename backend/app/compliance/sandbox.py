"""
Sandbox recipient gate.

In ``APP_MODE=sandbox`` every outbound delivery must target an allow-listed address
(your own inbox / phone). Anything else is denied and logged, never silently rerouted:
rerouting would hide which real customer the agent tried to reach.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.compliance.decision import Reason, deny
from app.core.config import AppMode, Settings
from app.domain.models import Channel


@dataclass(frozen=True, slots=True)
class SandboxGate:
    mode: AppMode
    allowed_emails: frozenset[str]
    allowed_phones: frozenset[str]

    @classmethod
    def from_settings(cls, settings: Settings) -> SandboxGate:
        return cls(
            mode=settings.app_mode,
            allowed_emails=frozenset(settings.sandbox_allowed_emails),
            allowed_phones=frozenset(settings.sandbox_allowed_phones),
        )

    @property
    def active(self) -> bool:
        return self.mode is AppMode.SANDBOX

    def check(self, channel: Channel, recipient: str | None) -> Reason | None:
        """``None`` when delivery is permitted; otherwise the denial reason."""
        if not self.active:
            return None
        if recipient is None:
            return deny("sandbox_recipient_missing", "no recipient address to verify")
        allowed = self.allowed_emails if channel is Channel.EMAIL else self.allowed_phones
        normalized = recipient.lower() if channel is Channel.EMAIL else recipient
        if normalized not in allowed:
            return deny(
                "sandbox_recipient_not_allowed",
                f"sandbox mode: {channel.value} recipient is not on the allow-list",
            )
        return None
