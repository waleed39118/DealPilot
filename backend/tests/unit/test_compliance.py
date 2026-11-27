"""Unit tests for decision types, channel policy, sandbox gate and disclosures."""

from __future__ import annotations

import base64
import secrets
from datetime import UTC, datetime, timedelta

import pytest

from app.compliance.channel_policy import check_channel, eligible_channels, recipient_for
from app.compliance.decision import Decision, Verdict, defer, deny, needs_approval
from app.compliance.disclosure import (
    SenderIdentity,
    apply_disclosure,
    build_footer,
    voice_opening_statement,
)
from app.compliance.sandbox import SandboxGate
from app.core.config import AppMode, Settings
from app.domain.deal_state import DealStage
from app.domain.models import Channel, Consent, Deal, Language, LawfulBasis, Prospect
from tests.unit.conftest import TENANT

NOW = datetime(2026, 11, 25, 9, 0, tzinfo=UTC)
IDENTITY = SenderIdentity("Example Trading LLC", "1 King Fahd Rd, Riyadh", "https://ex.com/privacy")


def prospect(**overrides: object) -> Prospect:
    data: dict[str, object] = {
        "tenant_id": TENANT,
        "full_name": "Sara Ali",
        "company_name": "Retail Co",
        "email": "sara@retail.example",
        "phone": "+966500000001",
        "country": "SA",
        "timezone": "Asia/Riyadh",
        "data_source": "Apollo.io",
    }
    data.update(overrides)
    return Prospect(**data)  # type: ignore[arg-type]


def deal_for(p: Prospect, stage: DealStage = DealStage.ENGAGED) -> Deal:
    from uuid import uuid4

    return Deal(tenant_id=TENANT, project_id=uuid4(), prospect_id=p.id, stage=stage)


def consent(p: Prospect, channel: Channel, **overrides: object) -> Consent:
    data: dict[str, object] = {
        "tenant_id": TENANT,
        "prospect_id": p.id,
        "channel": channel,
        "granted_at": NOW - timedelta(days=1),
        "evidence": "Website opt-in form",
    }
    data.update(overrides)
    return Consent(**data)  # type: ignore[arg-type]


def check(channel: Channel, p: Prospect, d: Deal, consents=(), suppressed=False):  # type: ignore[no-untyped-def]
    return [
        r.code
        for r in check_channel(
            channel, prospect=p, deal=d, consents=consents, is_suppressed=suppressed, at=NOW
        )
    ]


# ── Decision ────────────────────────────────────────────────────────────────


def test_decision_precedence_and_deferral_time() -> None:
    later, latest = NOW + timedelta(hours=1), NOW + timedelta(hours=5)
    assert Decision.combine([]).allowed
    deferred = Decision.combine([defer("a", "x", later), defer("b", "y", latest)])
    assert deferred.verdict is Verdict.DEFER
    assert deferred.not_before == latest
    mixed = Decision.combine([defer("a", "x", later), needs_approval("b", "y")])
    assert mixed.verdict is Verdict.REQUIRE_APPROVAL
    assert mixed.not_before is None
    worst = Decision.combine([needs_approval("b", "y"), deny("c", "z")])
    assert worst.verdict is Verdict.DENY
    assert worst.codes == ("b", "c")


# ── Channel policy ──────────────────────────────────────────────────────────


def test_email_b2b_allowed_with_legitimate_interest() -> None:
    p = prospect()
    assert check(Channel.EMAIL, p, deal_for(p)) == []


@pytest.mark.parametrize(
    "overrides",
    [{"country": "DE", "timezone": "Europe/Berlin"}, {"lawful_basis": LawfulBasis.CONSENT}],
)
def test_email_requires_consent_in_opt_in_regimes(overrides: dict[str, object]) -> None:
    p = prospect(**overrides)
    d = deal_for(p)
    assert check(Channel.EMAIL, p, d) == ["email_consent_required"]
    assert check(Channel.EMAIL, p, d, [consent(p, Channel.EMAIL)]) == []


@pytest.mark.parametrize("channel", [Channel.WHATSAPP, Channel.VOICE])
def test_whatsapp_and_voice_require_active_opt_in(channel: Channel) -> None:
    p = prospect()
    d = deal_for(p)
    assert check(channel, p, d) == [f"{channel.value}_consent_required"]
    assert check(channel, p, d, [consent(p, channel)]) == []
    revoked = consent(p, channel, revoked_at=NOW - timedelta(minutes=1))
    assert check(channel, p, d, [revoked]) == [f"{channel.value}_consent_required"]
    other_channel = consent(p, Channel.EMAIL)
    assert check(channel, p, d, [other_channel]) == [f"{channel.value}_consent_required"]


def test_consent_of_another_prospect_does_not_count() -> None:
    p, stranger = prospect(), prospect()
    assert check(Channel.WHATSAPP, p, deal_for(p), [consent(stranger, Channel.WHATSAPP)]) == [
        "whatsapp_consent_required"
    ]


def test_fax_only_within_existing_deal() -> None:
    p = prospect(fax="+966110000000")
    assert check(Channel.FAX, p, deal_for(p, DealStage.QUALIFIED)) == ["fax_requires_existing_deal"]
    assert check(Channel.FAX, p, deal_for(p, DealStage.NEGOTIATING)) == []
    assert check(Channel.FAX, prospect(), deal_for(p)) == ["fax_address_missing"]


def test_suppression_blocks_every_channel() -> None:
    p = prospect()
    for channel in Channel:
        assert "contact_suppressed" in check(channel, p, deal_for(p), suppressed=True)
    assert "contact_suppressed" in check(Channel.EMAIL, p, deal_for(p, DealStage.OPTED_OUT))


def test_eligible_channels_and_recipients() -> None:
    p = prospect()
    d = deal_for(p)
    assert eligible_channels(prospect=p, deal=d, consents=[], is_suppressed=False, at=NOW) == [
        Channel.EMAIL
    ]
    with_opt_in = eligible_channels(
        prospect=p, deal=d, consents=[consent(p, Channel.WHATSAPP)], is_suppressed=False, at=NOW
    )
    assert with_opt_in == [Channel.EMAIL, Channel.WHATSAPP]
    assert recipient_for(Channel.EMAIL, p) == "sara@retail.example"
    assert recipient_for(Channel.VOICE, p) == "+966500000001"
    assert recipient_for(Channel.FAX, p) is None


# ── Sandbox ─────────────────────────────────────────────────────────────────


def test_sandbox_gate() -> None:
    gate = SandboxGate(
        AppMode.SANDBOX, frozenset({"me@test.example"}), frozenset({"+966500000009"})
    )
    assert gate.check(Channel.EMAIL, "ME@test.example") is None
    assert gate.check(Channel.WHATSAPP, "+966500000009") is None
    denied = gate.check(Channel.EMAIL, "customer@real.example")
    assert denied is not None and denied.code == "sandbox_recipient_not_allowed"
    assert gate.check(Channel.VOICE, "+966500000001") is not None
    missing = gate.check(Channel.FAX, None)
    assert missing is not None and missing.code == "sandbox_recipient_missing"

    live = SandboxGate(AppMode.LIVE, frozenset(), frozenset())
    assert live.check(Channel.EMAIL, "customer@real.example") is None


def test_sandbox_gate_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        master_encryption_key=base64.b64encode(secrets.token_bytes(32)).decode(),
        sandbox_allowed_emails="me@test.example",
        sandbox_allowed_phones="+966500000009",
    )
    gate = SandboxGate.from_settings(settings)
    assert gate.active
    assert gate.allowed_emails == frozenset({"me@test.example"})


# ── Disclosure ──────────────────────────────────────────────────────────────


def test_identity_validation() -> None:
    with pytest.raises(ValueError):
        SenderIdentity("", "addr", "https://x")
    with pytest.raises(ValueError):
        SenderIdentity("Co", "addr", "http://insecure")


def test_first_email_in_arabic_contains_every_disclosure() -> None:
    body = apply_disclosure(
        "مرحباً سارة،",
        language=Language.AR,
        channel=Channel.EMAIL,
        identity=IDENTITY,
        is_first_contact=True,
        data_source="Apollo.io",
        unsubscribe_url="https://ex.com/u/abc",
    )
    assert "مساعد آلي يتواصل نيابة عن Example Trading LLC" in body
    assert "Apollo.io" in body
    assert "https://ex.com/u/abc" in body
    assert "1 King Fahd Rd, Riyadh" in body
    assert "https://ex.com/privacy" in body


def test_follow_up_email_keeps_legal_footer_only() -> None:
    footer = build_footer(
        language=Language.EN,
        channel=Channel.EMAIL,
        identity=IDENTITY,
        is_first_contact=False,
        unsubscribe_url="https://ex.com/u/abc",
    )
    assert "automated assistant" not in footer
    assert "unsubscribe" in footer
    assert "1 King Fahd Rd" in footer


def test_email_requires_https_unsubscribe() -> None:
    with pytest.raises(ValueError, match="unsubscribe"):
        build_footer(
            language=Language.EN, channel=Channel.EMAIL, identity=IDENTITY, is_first_contact=False
        )


def test_whatsapp_first_contact_and_follow_up() -> None:
    first = build_footer(
        language=Language.EN, channel=Channel.WHATSAPP, identity=IDENTITY, is_first_contact=True
    )
    assert "automated assistant" in first
    assert "STOP" in first
    later = build_footer(
        language=Language.EN, channel=Channel.WHATSAPP, identity=IDENTITY, is_first_contact=False
    )
    assert later == ""
    assert (
        apply_disclosure(
            "Hi",
            language=Language.EN,
            channel=Channel.WHATSAPP,
            identity=IDENTITY,
            is_first_contact=False,
        )
        == "Hi"
    )


def test_disclosure_is_idempotent() -> None:
    kwargs = {
        "language": Language.EN,
        "channel": Channel.EMAIL,
        "identity": IDENTITY,
        "is_first_contact": True,
        "data_source": "Apollo.io",
        "unsubscribe_url": "https://ex.com/u/abc",
    }
    once = apply_disclosure("Hello", **kwargs)  # type: ignore[arg-type]
    assert apply_disclosure(once, **kwargs) == once  # type: ignore[arg-type]


def test_voice_uses_spoken_statement() -> None:
    with pytest.raises(ValueError, match="voice"):
        apply_disclosure(
            "x",
            language=Language.EN,
            channel=Channel.VOICE,
            identity=IDENTITY,
            is_first_contact=True,
        )
    assert "automated assistant calling on behalf of Example Trading LLC" in (
        voice_opening_statement(Language.EN, IDENTITY)
    )
    assert "مساعد آلي" in voice_opening_statement(Language.AR, IDENTITY)
