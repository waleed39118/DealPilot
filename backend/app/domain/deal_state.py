"""
Deal lifecycle state machine.

The language model may *propose* a stage change; only this module decides whether it
is allowed. Rules are structural guarantees, independent of any prompt:

* Allowed edges are listed explicitly in ``_TRANSITIONS``; anything else is rejected.
* A contract can only be sent from ``AWAITING_APPROVAL`` and only with an approval id,
  so no path lets the agent send a binding contract without a human decision.
* ``HUMAN_TAKEOVER`` is entered and left only by a human: while a person holds the
  conversation the agent cannot act on the deal.
* ``WON`` is set only by a human or by the system (signature/payment webhooks).
* An opt-out is always accepted, from any non-terminal stage, by any actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from app.core.errors import PermanentError

if TYPE_CHECKING:
    from app.domain.models import Deal


class Actor(StrEnum):
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


class DealStage(StrEnum):
    NEW = "new"
    RESEARCHED = "researched"
    DISQUALIFIED = "disqualified"
    QUALIFIED = "qualified"
    PENDING_FIRST_APPROVAL = "pending_first_approval"
    CONTACTED = "contacted"
    FOLLOW_UP = "follow_up"
    DORMANT = "dormant"
    ENGAGED = "engaged"
    NEGOTIATING = "negotiating"
    AWAITING_APPROVAL = "awaiting_approval"
    HUMAN_TAKEOVER = "human_takeover"
    CONTRACT_SENT = "contract_sent"
    WON = "won"
    LOST = "lost"
    OPTED_OUT = "opted_out"


S = DealStage

TERMINAL_STAGES = frozenset({S.WON, S.LOST, S.OPTED_OUT, S.DISQUALIFIED})

# Stages in which a live conversation exists and a human may take over.
_TAKEOVER_SOURCES = frozenset(
    {
        S.PENDING_FIRST_APPROVAL,
        S.CONTACTED,
        S.FOLLOW_UP,
        S.DORMANT,
        S.ENGAGED,
        S.NEGOTIATING,
        S.AWAITING_APPROVAL,
        S.CONTRACT_SENT,
    }
)

_TRANSITIONS: dict[DealStage, frozenset[DealStage]] = {
    S.NEW: frozenset({S.RESEARCHED, S.DISQUALIFIED}),
    S.RESEARCHED: frozenset({S.QUALIFIED, S.DISQUALIFIED}),
    S.QUALIFIED: frozenset({S.PENDING_FIRST_APPROVAL, S.CONTACTED, S.DISQUALIFIED}),
    S.PENDING_FIRST_APPROVAL: frozenset({S.CONTACTED, S.QUALIFIED, S.DISQUALIFIED}),
    S.CONTACTED: frozenset({S.ENGAGED, S.FOLLOW_UP}),
    S.FOLLOW_UP: frozenset({S.ENGAGED, S.DORMANT}),
    S.DORMANT: frozenset({S.ENGAGED, S.LOST}),
    S.ENGAGED: frozenset({S.NEGOTIATING, S.AWAITING_APPROVAL, S.DORMANT, S.LOST}),
    S.NEGOTIATING: frozenset({S.AWAITING_APPROVAL, S.LOST}),
    S.AWAITING_APPROVAL: frozenset({S.NEGOTIATING, S.CONTRACT_SENT, S.LOST}),
    S.CONTRACT_SENT: frozenset({S.WON, S.LOST, S.NEGOTIATING}),
    S.HUMAN_TAKEOVER: frozenset(
        {S.ENGAGED, S.NEGOTIATING, S.FOLLOW_UP, S.DORMANT, S.CONTRACT_SENT, S.WON, S.LOST}
    ),
    S.WON: frozenset(),
    S.LOST: frozenset(),
    S.OPTED_OUT: frozenset(),
    S.DISQUALIFIED: frozenset(),
}

# Edges that execute an outward commitment and therefore need a recorded approval.
_APPROVAL_GATED: frozenset[tuple[DealStage, DealStage]] = frozenset(
    {
        (S.PENDING_FIRST_APPROVAL, S.CONTACTED),
        (S.AWAITING_APPROVAL, S.CONTRACT_SENT),
    }
)


class InvalidTransitionError(PermanentError):
    """A requested stage change violates the deal lifecycle rules."""


@dataclass(frozen=True, slots=True)
class StageTransition:
    """Audit record of one stage change (persisted and streamed to the activity feed)."""

    deal_id: UUID
    from_stage: DealStage
    to_stage: DealStage
    actor: Actor
    reason: str
    approval_id: UUID | None
    occurred_at: datetime


def allowed_targets(stage: DealStage) -> frozenset[DealStage]:
    """Every stage reachable from ``stage`` (ignoring actor and approval rules)."""
    targets = set(_TRANSITIONS[stage])
    if stage not in TERMINAL_STAGES:
        targets.add(S.OPTED_OUT)
    if stage in _TAKEOVER_SOURCES:
        targets.add(S.HUMAN_TAKEOVER)
    return frozenset(targets)


def requires_approval(from_stage: DealStage, to_stage: DealStage) -> bool:
    return (from_stage, to_stage) in _APPROVAL_GATED


def check_transition(
    from_stage: DealStage,
    to_stage: DealStage,
    *,
    actor: Actor,
    approval_id: UUID | None = None,
) -> None:
    """Raise ``InvalidTransitionError`` unless the transition is permitted."""
    if from_stage is to_stage:
        raise InvalidTransitionError(f"deal is already in stage '{to_stage}'")
    if from_stage in TERMINAL_STAGES:
        raise InvalidTransitionError(f"stage '{from_stage}' is final")
    if to_stage not in allowed_targets(from_stage):
        raise InvalidTransitionError(f"transition '{from_stage}' → '{to_stage}' is not allowed")
    if to_stage is S.OPTED_OUT:
        return
    if (
        from_stage is S.HUMAN_TAKEOVER or to_stage is S.HUMAN_TAKEOVER
    ) and actor is not Actor.HUMAN:
        raise InvalidTransitionError("only a human can take over or release a conversation")
    if to_stage is S.WON and actor is Actor.AGENT:
        raise InvalidTransitionError("the agent cannot mark a deal as won")
    if requires_approval(from_stage, to_stage) and approval_id is None:
        raise InvalidTransitionError(
            f"transition '{from_stage}' → '{to_stage}' requires a recorded approval"
        )


def transition(
    deal: Deal,
    to_stage: DealStage,
    *,
    actor: Actor,
    reason: str,
    approval_id: UUID | None = None,
    now: datetime | None = None,
) -> tuple[Deal, StageTransition]:
    """
    Apply a validated stage change. Returns the updated (new, immutable) deal and the
    audit record. ``version`` is incremented for optimistic locking at persistence time.
    """
    if not reason.strip():
        raise InvalidTransitionError("a reason is required for every stage change")
    check_transition(deal.stage, to_stage, actor=actor, approval_id=approval_id)
    occurred_at = now or datetime.now(UTC)
    updated = deal.model_copy(
        update={"stage": to_stage, "version": deal.version + 1, "updated_at": occurred_at}
    )
    record = StageTransition(
        deal_id=deal.id,
        from_stage=deal.stage,
        to_stage=to_stage,
        actor=actor,
        reason=reason.strip(),
        approval_id=approval_id,
        occurred_at=occurred_at,
    )
    return updated, record
