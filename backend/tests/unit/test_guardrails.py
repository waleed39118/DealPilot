"""Unit tests for app.compliance.guardrails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from app.compliance.decision import Verdict
from app.compliance.guardrails import GuardrailContext, evaluate
from app.compliance.sandbox import SandboxGate
from app.core.config import AppMode
from app.domain.actions import (
    ChangeStage,
    EscalateToHuman,
    OfferPrice,
    ScheduleCall,
    SendContract,
    SendInvoice,
    SendMessage,
)
from app.domain.deal_state import Actor, DealStage
from app.domain.models import (
    Channel,
    Consent,
    Deal,
    DealPolicy,
    Language,
    PaymentTerm,
    Project,
    ProjectBrief,
    ProjectStatus,
    Prospect,
)
from tests.unit.conftest import TENANT, brief_data, policy_data

S = DealStage
# 09:00 UTC = 12:00 in Riyadh: outside the default 20:00-09:00 quiet hours.
NOW = datetime(2026, 11, 25, 9, 0, tzinfo=UTC)
LIVE = SandboxGate(AppMode.LIVE, frozenset(), frozenset())


def build_ctx(
    stage: DealStage = S.NEGOTIATING,
    *,
    now: datetime = NOW,
    deal_overrides: dict[str, Any] | None = None,
    policy_overrides: dict[str, Any] | None = None,
    brief_overrides: dict[str, Any] | None = None,
    project_status: ProjectStatus = ProjectStatus.ACTIVE,
    **ctx_overrides: Any,
) -> GuardrailContext:
    project = Project(
        tenant_id=TENANT,
        brief=ProjectBrief(**brief_data(**(brief_overrides or {}))),
        policy=DealPolicy(**policy_data(**(policy_overrides or {}))),
        status=project_status,
    )
    prospect = Prospect(
        tenant_id=TENANT,
        full_name="Sara Ali",
        company_name="Retail Co",
        email="sara@retail.example",
        phone="+966500000001",
        country="SA",
        timezone="Asia/Riyadh",
        data_source="Apollo.io",
    )
    deal = Deal(
        tenant_id=TENANT,
        project_id=project.id,
        prospect_id=prospect.id,
        stage=stage,
        **(deal_overrides or {}),
    )
    values: dict[str, Any] = {
        "project": project,
        "deal": deal,
        "prospect": prospect,
        "sandbox": LIVE,
        "now": now,
    }
    values.update(ctx_overrides)
    return GuardrailContext(**values)


def message(ctx: GuardrailContext, **overrides: Any) -> SendMessage:
    data: dict[str, Any] = {
        "deal_id": ctx.deal.id,
        "rationale": "reply",
        "channel": Channel.EMAIL,
        "language": Language.EN,
        "body": "Thanks for your interest.",
    }
    data.update(overrides)
    return SendMessage(**data)


def offer(ctx: GuardrailContext, price: str, **overrides: Any) -> OfferPrice:
    data: dict[str, Any] = {
        "deal_id": ctx.deal.id,
        "rationale": "counter-offer",
        "price": Decimal(price),
        "currency": "SAR",
        "payment_term": PaymentTerm.UPFRONT,
    }
    data.update(overrides)
    return OfferPrice(**data)


# ── Integrity ───────────────────────────────────────────────────────────────


def test_action_for_another_deal_is_denied() -> None:
    ctx = build_ctx()
    decision = evaluate(message(ctx, deal_id=uuid4()), ctx)
    assert decision.codes == ("deal_mismatch",)


def test_escalation_always_allowed() -> None:
    ctx = build_ctx(S.HUMAN_TAKEOVER, kill_switch_active=True)
    action = EscalateToHuman(deal_id=ctx.deal.id, rationale="unsure", question="Can we do 20%?")
    assert evaluate(action, ctx).allowed


# ── Messages ────────────────────────────────────────────────────────────────


def test_reply_in_negotiation_is_allowed() -> None:
    ctx = build_ctx()
    assert evaluate(message(ctx), ctx).allowed


def test_first_contact_requires_approval_even_if_model_says_otherwise() -> None:
    ctx = build_ctx(S.QUALIFIED)
    decision = evaluate(message(ctx, is_first_contact=False), ctx)
    assert decision.verdict is Verdict.REQUIRE_APPROVAL
    assert "first_message_requires_approval" in decision.codes


def test_first_contact_without_review_when_user_disabled_it() -> None:
    ctx = build_ctx(S.QUALIFIED, policy_overrides={"require_approval_first_message": False})
    assert evaluate(message(ctx), ctx).allowed


def test_message_before_qualification_denied() -> None:
    ctx = build_ctx(S.RESEARCHED)
    assert "prospect_not_qualified" in evaluate(message(ctx), ctx).codes


@pytest.mark.parametrize(
    ("ctx_kwargs", "code"),
    [
        ({"kill_switch_active": True}, "kill_switch_active"),
        ({"project_status": ProjectStatus.PAUSED}, "project_not_active"),
        ({"is_suppressed": True}, "contact_suppressed"),
        ({"stage": S.WON}, "deal_closed"),
        ({"stage": S.HUMAN_TAKEOVER}, "human_takeover"),
    ],
)
def test_global_denials(ctx_kwargs: dict[str, Any], code: str) -> None:
    ctx = build_ctx(**ctx_kwargs)
    decision = evaluate(message(ctx), ctx)
    assert decision.verdict is Verdict.DENY
    assert code in decision.codes


def test_human_may_write_during_takeover() -> None:
    ctx = build_ctx(S.HUMAN_TAKEOVER)
    assert evaluate(message(ctx, proposed_by=Actor.HUMAN), ctx).allowed


def test_sandbox_blocks_real_customers() -> None:
    sandbox = SandboxGate(AppMode.SANDBOX, frozenset({"me@test.example"}), frozenset())
    ctx = build_ctx(sandbox=sandbox)
    assert evaluate(message(ctx), ctx).codes == ("sandbox_recipient_not_allowed",)


def test_whatsapp_without_opt_in_denied_and_with_opt_in_allowed() -> None:
    ctx = build_ctx()
    assert (
        "whatsapp_consent_required" in evaluate(message(ctx, channel=Channel.WHATSAPP), ctx).codes
    )
    opt_in = Consent(
        tenant_id=TENANT,
        prospect_id=ctx.prospect.id,
        channel=Channel.WHATSAPP,
        granted_at=NOW - timedelta(days=2),
        evidence="Opt-in checkbox",
    )
    ctx = GuardrailContext(**{**_fields(ctx), "consents": (opt_in,)})
    assert evaluate(message(ctx, channel=Channel.WHATSAPP), ctx).allowed


def _fields(ctx: GuardrailContext) -> dict[str, Any]:
    return {name: getattr(ctx, name) for name in GuardrailContext.__slots__}


def test_language_and_forbidden_claims() -> None:
    ctx = build_ctx(
        brief_overrides={"languages": {Language.AR}, "forbidden_claims": ["Guaranteed ROI"]}
    )
    decision = evaluate(message(ctx, body="We offer guaranteed roi to everyone"), ctx)
    assert set(decision.codes) >= {"language_not_in_brief", "forbidden_claim"}


def test_quiet_hours_defer_until_morning_local_time() -> None:
    night = datetime(2026, 11, 25, 19, 30, tzinfo=UTC)  # 22:30 in Riyadh
    ctx = build_ctx(now=night)
    decision = evaluate(message(ctx), ctx)
    assert decision.verdict is Verdict.DEFER
    assert decision.not_before == datetime(2026, 11, 26, 6, 0, tzinfo=UTC)  # 09:00 Riyadh


def test_follow_up_frequency_limits() -> None:
    recent = {"touches_without_reply": 1, "last_touch_at": NOW - timedelta(hours=10)}
    ctx = build_ctx(S.FOLLOW_UP, deal_overrides=recent)
    decision = evaluate(message(ctx), ctx)
    assert decision.verdict is Verdict.DEFER
    assert decision.not_before == NOW - timedelta(hours=10) + timedelta(hours=72)

    old = {"touches_without_reply": 1, "last_touch_at": NOW - timedelta(hours=80)}
    ctx = build_ctx(S.FOLLOW_UP, deal_overrides=old)
    assert evaluate(message(ctx), ctx).allowed

    exhausted = {"touches_without_reply": 3, "last_touch_at": NOW - timedelta(days=10)}
    ctx = build_ctx(S.FOLLOW_UP, deal_overrides=exhausted)
    assert evaluate(message(ctx), ctx).codes == ("max_touches_reached",)


def test_human_cannot_bypass_legal_rules() -> None:
    ctx = build_ctx(is_suppressed=True)
    decision = evaluate(message(ctx, proposed_by=Actor.HUMAN), ctx)
    assert decision.verdict is Verdict.DENY


# ── Offers & pricing authority ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("price", "verdict", "code"),
    [
        ("10000", Verdict.ALLOW, None),
        ("9000", Verdict.ALLOW, None),  # exactly 10% discount
        ("8999.99", Verdict.REQUIRE_APPROVAL, "discount_above_cap"),
        ("8000", Verdict.REQUIRE_APPROVAL, "discount_above_cap"),
        ("7999.99", Verdict.DENY, "below_price_floor"),
        ("30000", Verdict.REQUIRE_APPROVAL, "deal_value_above_threshold"),
    ],
)
def test_price_authority(price: str, verdict: Verdict, code: str | None) -> None:
    ctx = build_ctx()
    decision = evaluate(offer(ctx, price), ctx)
    assert decision.verdict is verdict
    if code:
        assert code in decision.codes


def test_binding_offer_and_payment_terms_need_approval() -> None:
    ctx = build_ctx()
    decision = evaluate(offer(ctx, "9500", binding=True, payment_term=PaymentTerm.NET_60), ctx)
    assert decision.verdict is Verdict.REQUIRE_APPROVAL
    assert set(decision.codes) == {"binding_offer_requires_approval", "payment_term_not_allowed"}


def test_currency_mismatch_denied_even_for_humans() -> None:
    ctx = build_ctx()
    action = offer(ctx, "9500", currency="USD", proposed_by=Actor.HUMAN)
    assert evaluate(action, ctx).codes == ("currency_mismatch",)


def test_human_below_floor_needs_explicit_confirmation() -> None:
    ctx = build_ctx()
    below = evaluate(offer(ctx, "5000", proposed_by=Actor.HUMAN, binding=True), ctx)
    assert below.verdict is Verdict.REQUIRE_APPROVAL
    assert below.codes == ("below_price_floor_confirmation",)
    # Within the floor, a human decision needs no extra step (no cap, terms or threshold).
    deep = offer(ctx, "8000", proposed_by=Actor.HUMAN, payment_term=PaymentTerm.NET_60)
    assert evaluate(deep, ctx).allowed


def test_no_offers_before_engagement() -> None:
    ctx = build_ctx(S.CONTACTED)
    assert "offer_not_allowed_in_stage" in evaluate(offer(ctx, "9500"), ctx).codes


# ── Contracts & invoices ────────────────────────────────────────────────────


def test_contract_always_requires_approval_from_agent() -> None:
    ctx = build_ctx(S.AWAITING_APPROVAL)
    action = SendContract(
        deal_id=ctx.deal.id,
        rationale="agreed",
        price=Decimal("9500"),
        currency="SAR",
        payment_term=PaymentTerm.UPFRONT,
        document_ref="contracts/1.pdf",
    )
    decision = evaluate(action, ctx)
    assert decision.verdict is Verdict.REQUIRE_APPROVAL
    assert decision.codes == ("contract_requires_approval",)


def test_contract_in_wrong_stage_denied() -> None:
    ctx = build_ctx(S.ENGAGED)
    action = SendContract(
        deal_id=ctx.deal.id,
        rationale="rushing",
        price=Decimal("9500"),
        currency="SAR",
        payment_term=PaymentTerm.UPFRONT,
        document_ref="contracts/1.pdf",
    )
    assert "contract_not_allowed_in_stage" in evaluate(action, ctx).codes


def test_invoice_after_contract() -> None:
    ctx = build_ctx(S.CONTRACT_SENT)
    action = SendInvoice(
        deal_id=ctx.deal.id,
        rationale="signed",
        amount=Decimal("9500"),
        currency="SAR",
        payment_term=PaymentTerm.UPFRONT,
    )
    assert evaluate(action, ctx).codes == ("invoice_requires_approval",)
    human = action.model_copy(update={"proposed_by": Actor.HUMAN})
    assert evaluate(human, ctx).allowed


# ── Calls ───────────────────────────────────────────────────────────────────


def test_call_rules() -> None:
    ctx = build_ctx()
    tomorrow_noon = datetime(2026, 11, 26, 9, 0, tzinfo=UTC)
    call = ScheduleCall(deal_id=ctx.deal.id, rationale="demo", at=tomorrow_noon)
    assert "voice_consent_required" in evaluate(call, ctx).codes

    opt_in = Consent(
        tenant_id=TENANT,
        prospect_id=ctx.prospect.id,
        channel=Channel.VOICE,
        granted_at=NOW - timedelta(days=1),
        evidence="Asked for a call by email",
    )
    ctx = GuardrailContext(**{**_fields(ctx), "consents": (opt_in,)})
    assert evaluate(call, ctx).allowed

    at_night = call.model_copy(update={"at": datetime(2026, 11, 26, 20, 0, tzinfo=UTC)})
    assert "call_in_quiet_hours" in evaluate(at_night, ctx).codes
    in_past = call.model_copy(update={"at": NOW - timedelta(hours=1)})
    assert "call_time_in_past" in evaluate(in_past, ctx).codes


# ── Stage changes ───────────────────────────────────────────────────────────


def test_stage_changes() -> None:
    ctx = build_ctx(S.AWAITING_APPROVAL)
    to_contract = ChangeStage(deal_id=ctx.deal.id, rationale="agreed", to_stage=S.CONTRACT_SENT)
    assert evaluate(to_contract, ctx).codes == ("stage_change_requires_approval",)

    ctx = build_ctx(S.NEGOTIATING)
    to_won = ChangeStage(deal_id=ctx.deal.id, rationale="done", to_stage=S.WON)
    assert evaluate(to_won, ctx).codes == ("invalid_stage_change",)
    to_approval = ChangeStage(deal_id=ctx.deal.id, rationale="ask", to_stage=S.AWAITING_APPROVAL)
    assert evaluate(to_approval, ctx).allowed
