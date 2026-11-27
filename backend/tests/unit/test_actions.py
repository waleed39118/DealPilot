"""Unit tests for app.domain.actions."""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.actions import (
    ChangeStage,
    InvalidActionError,
    OfferPrice,
    SendContract,
    SendMessage,
    offered_amount,
    parse_proposed_action,
)
from app.domain.deal_state import Actor, DealStage

DEAL = str(uuid4())


def raw(**fields: object) -> str:
    """JSON text as a model would emit it (numbers stay JSON numbers)."""
    return json.dumps({"deal_id": DEAL, "rationale": "x", **fields})


def test_parse_message_from_json() -> None:
    action = parse_proposed_action(
        json.dumps(
            {
                "kind": "send_message",
                "deal_id": DEAL,
                "rationale": "First outreach",
                "channel": "email",
                "language": "ar",
                "subject": "عرض خاص",
                "body": "مرحباً، ...",
                "is_first_contact": True,
            }
        )
    )
    assert isinstance(action, SendMessage)
    assert action.proposed_by is Actor.AGENT
    assert action.is_first_contact


def test_parse_offer_keeps_exact_decimal() -> None:
    action = parse_proposed_action(
        raw(kind="offer_price", price=9500.1, currency="SAR", payment_term="upfront", binding=True)
    )
    assert isinstance(action, OfferPrice)
    assert action.price == Decimal("9500.1")
    assert offered_amount(action) == Decimal("9500.1")


def test_parse_from_dict_with_explicit_human_actor() -> None:
    action = parse_proposed_action(
        {
            "kind": "send_contract",
            "deal_id": DEAL,
            "rationale": "Agreed terms",
            "price": "9000",
            "currency": "SAR",
            "payment_term": "split_50_50",
            "document_ref": "contracts/2026/001.pdf",
        },
        actor=Actor.HUMAN,
    )
    assert isinstance(action, SendContract)
    assert action.proposed_by is Actor.HUMAN


@pytest.mark.parametrize("field", ["proposed_by", "action_id", "created_at"])
def test_model_cannot_set_system_fields(field: str) -> None:
    payload = {
        "kind": "change_stage",
        "deal_id": DEAL,
        "rationale": "x",
        "to_stage": "won",
        field: "human",
    }
    with pytest.raises(InvalidActionError, match="system-assigned"):
        parse_proposed_action(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[1, 2]",
        raw(kind="wire_money"),
        raw(kind="change_stage", to_stage="won", approved=True),
        raw(kind="offer_price", price=-5, currency="SAR", payment_term="upfront"),
        raw(kind="offer_price", price=10.123, currency="SAR", payment_term="upfront"),
        raw(kind="offer_price", price=10, currency="sar", payment_term="upfront"),
        raw(kind="send_message", channel="sms", language="en", body="hi"),
    ],
)
def test_invalid_payloads_rejected(payload: str) -> None:
    with pytest.raises(InvalidActionError):
        parse_proposed_action(payload)


def test_error_message_does_not_echo_payload_content() -> None:
    with pytest.raises(InvalidActionError) as exc_info:
        parse_proposed_action(
            {"kind": "send_message", "deal_id": DEAL, "rationale": "x", "body": "SECRET-TEXT"}
        )
    assert "SECRET-TEXT" not in str(exc_info.value)


def test_change_stage_and_non_monetary_actions() -> None:
    action = parse_proposed_action(
        {"kind": "change_stage", "deal_id": DEAL, "rationale": "replied", "to_stage": "engaged"}
    )
    assert isinstance(action, ChangeStage)
    assert action.to_stage is DealStage.ENGAGED
    assert offered_amount(action) is None
