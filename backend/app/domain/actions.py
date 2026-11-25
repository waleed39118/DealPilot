"""
Proposed actions.

Agents never act directly. They emit a ``ProposedAction``; the guardrails engine
returns ALLOW / REQUIRE_APPROVAL / DENY, and only then is anything executed.

``parse_proposed_action`` is the single entry point for language-model output: it
accepts JSON, rejects unknown action kinds and unknown fields, and enforces bounds,
so malformed or manipulated model output can never become an executable action.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, TypeAdapter, ValidationError

from app.core.errors import PermanentError
from app.domain.deal_state import Actor, DealStage
from app.domain.models import (
    Amount,
    AwareDatetime,
    Channel,
    DomainModel,
    Language,
    LongText,
    MediumText,
    PaymentTerm,
    ShortText,
)

MessageBody = Annotated[str, Field(min_length=1, max_length=10_000)]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _ActionBase(DomainModel):
    action_id: UUID = Field(default_factory=uuid4)
    deal_id: UUID
    proposed_by: Actor = Actor.AGENT
    rationale: MediumText
    created_at: AwareDatetime = Field(default_factory=_utcnow)


class SendMessage(_ActionBase):
    kind: Literal["send_message"] = "send_message"
    channel: Channel
    language: Language
    subject: ShortText | None = None
    body: MessageBody
    is_first_contact: bool = False


class OfferPrice(_ActionBase):
    """A price proposal. ``binding=True`` means a firm offer the customer can accept."""

    kind: Literal["offer_price"] = "offer_price"
    price: Amount
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    payment_term: PaymentTerm
    binding: bool = False


class SendContract(_ActionBase):
    kind: Literal["send_contract"] = "send_contract"
    price: Amount
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    payment_term: PaymentTerm
    document_ref: ShortText


class SendInvoice(_ActionBase):
    kind: Literal["send_invoice"] = "send_invoice"
    amount: Amount
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    payment_term: PaymentTerm


class ScheduleCall(_ActionBase):
    kind: Literal["schedule_call"] = "schedule_call"
    at: AwareDatetime
    duration_minutes: int = Field(default=30, ge=10, le=120)


class EscalateToHuman(_ActionBase):
    """The agent asks the user for a decision or for information missing from the brief."""

    kind: Literal["escalate_to_human"] = "escalate_to_human"
    question: LongText


class ChangeStage(_ActionBase):
    kind: Literal["change_stage"] = "change_stage"
    to_stage: DealStage


ProposedAction = Annotated[
    SendMessage
    | OfferPrice
    | SendContract
    | SendInvoice
    | ScheduleCall
    | EscalateToHuman
    | ChangeStage,
    Field(discriminator="kind"),
]

_ADAPTER: TypeAdapter[ProposedAction] = TypeAdapter(ProposedAction)

# Actions that create a commitment towards the customer.
COMMITTING_KINDS = frozenset({"send_contract", "send_invoice"})


class InvalidActionError(PermanentError):
    """Model output could not be turned into a valid action."""


_SYSTEM_ASSIGNED_FIELDS = frozenset({"action_id", "proposed_by", "created_at"})


def parse_proposed_action(
    payload: str | bytes | dict[str, object],
    *,
    actor: Actor = Actor.AGENT,
) -> ProposedAction:
    """
    Validate untrusted input into an action. Identity, author and timestamp are always
    assigned here, never taken from the payload: model output claiming to be a human
    decision is rejected outright.
    """
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidActionError("proposed action is not valid JSON") from exc
    if not isinstance(data, dict):
        raise InvalidActionError("proposed action must be a JSON object")
    smuggled = sorted(_SYSTEM_ASSIGNED_FIELDS.intersection(data))
    if smuggled:
        raise InvalidActionError(f"system-assigned field(s) not allowed: {', '.join(smuggled)}")
    try:
        return _ADAPTER.validate_python({**data, "proposed_by": actor})
    except ValidationError as exc:
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        raise InvalidActionError(
            f"invalid proposed action ({exc.error_count()} error(s) in: {', '.join(fields)})"
        ) from exc


def offered_amount(action: ProposedAction) -> Decimal | None:
    """The monetary value an action commits to, if any."""
    if isinstance(action, (OfferPrice, SendContract)):
        return action.price
    if isinstance(action, SendInvoice):
        return action.amount
    return None
