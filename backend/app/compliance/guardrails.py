"""
Guardrails engine: the single gate every proposed action passes before execution.

``evaluate(action, context)`` is deterministic and has no side effects. It returns
ALLOW, DEFER (try again at ``not_before``), REQUIRE_APPROVAL or DENY, with every reason.
Nothing the language model writes can change the outcome: the decision depends only on
the typed action, the stored policy and the recorded facts in ``GuardrailContext``.

Authority model
---------------
* Legal rules (suppression, consent, sandbox, kill switch, quiet hours, contact
  frequency) apply to **every** actor, humans included.
* Commercial authority (price floor, discount cap, payment terms, deal-value threshold,
  approvals for first contact, binding offers, contracts and invoices) restricts the
  **agent and system**. A human-proposed action is already the user's decision, with
  one safeguard: a human price below the floor needs an explicit confirmation step
  (``below_price_floor_confirmation``) to catch typos and misuse of a hijacked session.
* Actions are re-evaluated at execution time, after approval, so a deferral or a newly
  revoked consent is still caught.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.compliance.channel_policy import check_channel, recipient_for
from app.compliance.decision import Decision, Reason, defer, deny, needs_approval
from app.compliance.sandbox import SandboxGate
from app.domain.actions import (
    ChangeStage,
    EscalateToHuman,
    OfferPrice,
    ProposedAction,
    ScheduleCall,
    SendContract,
    SendInvoice,
    SendMessage,
)
from app.domain.deal_state import (
    TERMINAL_STAGES,
    Actor,
    DealStage,
    InvalidTransitionError,
    check_transition,
    requires_approval,
)
from app.domain.models import (
    Channel,
    Consent,
    Deal,
    DealPolicy,
    PaymentTerm,
    Project,
    ProjectStatus,
    Prospect,
    QuietHours,
)

S = DealStage

_PRE_CONTACT_STAGES = frozenset({S.NEW, S.RESEARCHED})
_FIRST_CONTACT_STAGES = frozenset({S.QUALIFIED, S.PENDING_FIRST_APPROVAL})
_OFFER_STAGES = frozenset({S.ENGAGED, S.NEGOTIATING, S.AWAITING_APPROVAL, S.CONTRACT_SENT})
_CONTRACT_STAGES = frozenset({S.AWAITING_APPROVAL, S.CONTRACT_SENT})
_INVOICE_STAGES = frozenset({S.CONTRACT_SENT, S.WON})


@dataclass(frozen=True, slots=True)
class GuardrailContext:
    """Recorded facts the decision is based on (loaded from the database)."""

    project: Project
    deal: Deal
    prospect: Prospect
    sandbox: SandboxGate
    now: datetime
    consents: Sequence[Consent] = field(default_factory=tuple)
    is_suppressed: bool = False
    kill_switch_active: bool = False


def evaluate(action: ProposedAction, ctx: GuardrailContext) -> Decision:
    reasons: list[Reason] = []
    deal = ctx.deal

    if action.deal_id != deal.id:
        return Decision.combine([deny("deal_mismatch", "action targets a different deal")])
    if deal.tenant_id != ctx.prospect.tenant_id or deal.tenant_id != ctx.project.tenant_id:
        return Decision.combine([deny("tenant_mismatch", "context spans different tenants")])
    if deal.prospect_id != ctx.prospect.id or deal.project_id != ctx.project.id:
        return Decision.combine([deny("context_mismatch", "deal, prospect and project differ")])

    if isinstance(action, EscalateToHuman):
        return Decision.combine(reasons)
    if isinstance(action, ChangeStage):
        return Decision.combine(_check_stage_change(action, deal))

    # ── Everything below reaches the customer ──────────────────────────────
    reasons.extend(_global_outbound_checks(action.proposed_by, ctx))
    if isinstance(action, SendMessage):
        reasons.extend(_check_message(action, ctx))
    elif isinstance(action, OfferPrice):
        reasons.extend(_check_offer(action, ctx))
    elif isinstance(action, SendContract):
        reasons.extend(_check_commitment(action, ctx, _CONTRACT_STAGES, "contract"))
    elif isinstance(action, SendInvoice):
        reasons.extend(_check_commitment(action, ctx, _INVOICE_STAGES, "invoice"))
    elif isinstance(action, ScheduleCall):
        reasons.extend(_check_call(action, ctx))
    return Decision.combine(reasons)


# ── Shared checks ───────────────────────────────────────────────────────────


def _global_outbound_checks(actor: Actor, ctx: GuardrailContext) -> list[Reason]:
    reasons: list[Reason] = []
    if ctx.kill_switch_active:
        reasons.append(deny("kill_switch_active", "all outbound activity is paused"))
    if ctx.project.status is not ProjectStatus.ACTIVE:
        reasons.append(deny("project_not_active", f"project is {ctx.project.status.value}"))
    if ctx.deal.stage in TERMINAL_STAGES:
        reasons.append(deny("deal_closed", f"deal is {ctx.deal.stage.value}"))
    if ctx.deal.stage is S.HUMAN_TAKEOVER and actor is not Actor.HUMAN:
        reasons.append(deny("human_takeover", "a person is handling this conversation"))
    return reasons


def _channel_checks(channel: Channel, ctx: GuardrailContext) -> list[Reason]:
    reasons = check_channel(
        channel,
        prospect=ctx.prospect,
        deal=ctx.deal,
        consents=ctx.consents,
        is_suppressed=ctx.is_suppressed,
        at=ctx.now,
    )
    sandbox_reason = ctx.sandbox.check(channel, recipient_for(channel, ctx.prospect))
    if sandbox_reason is not None:
        reasons.append(sandbox_reason)
    return reasons


def _quiet_hours_reason(ctx: GuardrailContext, moment: datetime) -> Reason | None:
    quiet = ctx.project.policy.quiet_hours
    zone = ZoneInfo(ctx.prospect.timezone)
    local = moment.astimezone(zone)
    if not quiet.contains(local.time()):
        return None
    return defer(
        "quiet_hours",
        f"recipient's local time {local:%H:%M} is within quiet hours",
        _quiet_hours_end(quiet, local),
    )


def _quiet_hours_end(quiet: QuietHours, local: datetime) -> datetime:
    end = local.replace(hour=quiet.end.hour, minute=quiet.end.minute, second=0, microsecond=0)
    if end <= local:
        end += timedelta(days=1)
    return end


def _deal_value_reasons(amount: Decimal, policy: DealPolicy) -> list[Reason]:
    threshold = policy.approval_deal_value_above
    if threshold is not None and amount > threshold:
        return [
            needs_approval(
                "deal_value_above_threshold",
                f"value {amount} exceeds the approval threshold {threshold}",
            )
        ]
    return []


def _pricing_reasons(
    price: Decimal, currency: str, term: PaymentTerm, actor: Actor, policy: DealPolicy
) -> list[Reason]:
    reasons: list[Reason] = []
    if currency != policy.currency:
        reasons.append(
            deny("currency_mismatch", f"price must be in {policy.currency}, got {currency}")
        )
    if actor is Actor.HUMAN:
        if price < policy.price_floor:
            reasons.append(
                needs_approval(
                    "below_price_floor_confirmation",
                    f"{price} is below the price floor {policy.price_floor}; confirm with a reason",
                )
            )
        return reasons
    if price < policy.price_floor:
        reasons.append(
            deny("below_price_floor", f"{price} is below the price floor {policy.price_floor}")
        )
    elif price < policy.lowest_auto_price:
        reasons.append(
            needs_approval(
                "discount_above_cap",
                f"{price} is below the lowest automatic price {policy.lowest_auto_price}",
            )
        )
    if term not in policy.allowed_payment_terms:
        reasons.append(
            needs_approval(
                "payment_term_not_allowed", f"payment term '{term.value}' needs approval"
            )
        )
    reasons.extend(_deal_value_reasons(price, policy))
    return reasons


# ── Per-action checks ───────────────────────────────────────────────────────


def _check_stage_change(action: ChangeStage, deal: Deal) -> list[Reason]:
    try:
        check_transition(deal.stage, action.to_stage, actor=action.proposed_by, approval_id=None)
    except InvalidTransitionError as exc:
        if requires_approval(deal.stage, action.to_stage):
            return [needs_approval("stage_change_requires_approval", exc.message)]
        return [deny("invalid_stage_change", exc.message)]
    return []


def _check_message(action: SendMessage, ctx: GuardrailContext) -> list[Reason]:
    deal, policy, brief = ctx.deal, ctx.project.policy, ctx.project.brief
    reasons = _channel_checks(action.channel, ctx)

    if action.language not in brief.languages:
        reasons.append(
            deny("language_not_in_brief", f"project does not support '{action.language.value}'")
        )
    lowered = action.body.lower()
    for claim in brief.forbidden_claims:
        if claim.lower() in lowered:
            reasons.append(deny("forbidden_claim", "message contains a forbidden claim"))
            break

    if deal.stage in _PRE_CONTACT_STAGES:
        reasons.append(deny("prospect_not_qualified", "prospect has not been qualified yet"))
    # The stage, not the model's flag, decides whether this is a first contact.
    first_contact = action.is_first_contact or deal.stage in _FIRST_CONTACT_STAGES
    if (
        first_contact
        and policy.require_approval_first_message
        and action.proposed_by is not Actor.HUMAN
    ):
        reasons.append(
            needs_approval("first_message_requires_approval", "first message needs review")
        )

    if deal.touches_without_reply > 0:
        if deal.touches_without_reply >= policy.max_touches_per_prospect:
            reasons.append(
                deny(
                    "max_touches_reached",
                    f"{deal.touches_without_reply} unanswered messages already sent",
                )
            )
        elif deal.last_touch_at is not None:
            earliest = deal.last_touch_at + timedelta(hours=policy.min_hours_between_touches)
            if ctx.now < earliest:
                reasons.append(
                    defer("min_interval_between_touches", "follow-up sent too soon", earliest)
                )

    quiet = _quiet_hours_reason(ctx, ctx.now)
    if quiet is not None:
        reasons.append(quiet)
    return reasons


def _check_offer(action: OfferPrice, ctx: GuardrailContext) -> list[Reason]:
    reasons: list[Reason] = []
    if ctx.deal.stage not in _OFFER_STAGES:
        reasons.append(
            deny("offer_not_allowed_in_stage", f"no offers in stage '{ctx.deal.stage.value}'")
        )
    reasons.extend(
        _pricing_reasons(
            action.price,
            action.currency,
            action.payment_term,
            action.proposed_by,
            ctx.project.policy,
        )
    )
    if action.binding and action.proposed_by is not Actor.HUMAN:
        reasons.append(needs_approval("binding_offer_requires_approval", "binding offer"))
    quiet = _quiet_hours_reason(ctx, ctx.now)
    if quiet is not None:
        reasons.append(quiet)
    return reasons


def _check_commitment(
    action: SendContract | SendInvoice,
    ctx: GuardrailContext,
    stages: frozenset[DealStage],
    label: str,
) -> list[Reason]:
    reasons = _channel_checks(Channel.EMAIL, ctx)
    if ctx.deal.stage not in stages:
        reasons.append(
            deny(f"{label}_not_allowed_in_stage", f"no {label} in stage '{ctx.deal.stage.value}'")
        )
    amount = action.price if isinstance(action, SendContract) else action.amount
    reasons.extend(
        _pricing_reasons(
            amount, action.currency, action.payment_term, action.proposed_by, ctx.project.policy
        )
    )
    if action.proposed_by is not Actor.HUMAN:
        reasons.append(needs_approval(f"{label}_requires_approval", f"every {label} is reviewed"))
    return reasons


def _check_call(action: ScheduleCall, ctx: GuardrailContext) -> list[Reason]:
    reasons = _channel_checks(Channel.VOICE, ctx)
    if action.at <= ctx.now:
        reasons.append(deny("call_time_in_past", "call must be scheduled in the future"))
    quiet = _quiet_hours_reason(ctx, action.at)
    if quiet is not None:
        reasons.append(
            deny("call_in_quiet_hours", "call would take place in the recipient's quiet hours")
        )
    if ctx.deal.stage in _PRE_CONTACT_STAGES | _FIRST_CONTACT_STAGES:
        reasons.append(deny("call_before_engagement", "calls require an engaged prospect"))
    return reasons
