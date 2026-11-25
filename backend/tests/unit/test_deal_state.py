"""Unit tests for app.domain.deal_state."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.domain.deal_state import (
    TERMINAL_STAGES,
    Actor,
    DealStage,
    InvalidTransitionError,
    allowed_targets,
    check_transition,
    requires_approval,
    transition,
)
from app.domain.models import Deal
from tests.unit.conftest import TENANT

S = DealStage


def deal(stage: DealStage = S.NEW) -> Deal:
    return Deal(tenant_id=TENANT, project_id=uuid4(), prospect_id=uuid4(), stage=stage)


def test_happy_path_to_won() -> None:
    approval = uuid4()
    current = deal()
    path = [
        (S.RESEARCHED, Actor.AGENT, None),
        (S.QUALIFIED, Actor.AGENT, None),
        (S.PENDING_FIRST_APPROVAL, Actor.AGENT, None),
        (S.CONTACTED, Actor.SYSTEM, approval),
        (S.ENGAGED, Actor.SYSTEM, None),
        (S.NEGOTIATING, Actor.AGENT, None),
        (S.AWAITING_APPROVAL, Actor.AGENT, None),
        (S.CONTRACT_SENT, Actor.SYSTEM, approval),
        (S.WON, Actor.SYSTEM, None),
    ]
    for target, actor, approval_id in path:
        current, record = transition(
            current, target, actor=actor, reason="step", approval_id=approval_id
        )
        assert record.to_stage is target
    assert current.stage is S.WON
    assert current.version == len(path) + 1


def test_every_stage_has_rules() -> None:
    for stage in DealStage:
        allowed_targets(stage)


def test_agent_cannot_reach_contract_without_approval_path() -> None:
    for stage in (S.ENGAGED, S.NEGOTIATING):
        with pytest.raises(InvalidTransitionError, match="not allowed"):
            check_transition(stage, S.CONTRACT_SENT, actor=Actor.AGENT, approval_id=uuid4())
    with pytest.raises(InvalidTransitionError, match="requires a recorded approval"):
        check_transition(S.AWAITING_APPROVAL, S.CONTRACT_SENT, actor=Actor.SYSTEM)
    assert requires_approval(S.AWAITING_APPROVAL, S.CONTRACT_SENT)
    assert requires_approval(S.PENDING_FIRST_APPROVAL, S.CONTACTED)


def test_agent_cannot_mark_won() -> None:
    with pytest.raises(InvalidTransitionError, match="cannot mark a deal as won"):
        check_transition(S.CONTRACT_SENT, S.WON, actor=Actor.AGENT)
    check_transition(S.CONTRACT_SENT, S.WON, actor=Actor.HUMAN)


def test_only_humans_take_over_and_release() -> None:
    for actor in (Actor.AGENT, Actor.SYSTEM):
        with pytest.raises(InvalidTransitionError, match="only a human"):
            check_transition(S.NEGOTIATING, S.HUMAN_TAKEOVER, actor=actor)
        with pytest.raises(InvalidTransitionError, match="only a human"):
            check_transition(S.HUMAN_TAKEOVER, S.NEGOTIATING, actor=actor)
    check_transition(S.NEGOTIATING, S.HUMAN_TAKEOVER, actor=Actor.HUMAN)
    check_transition(S.HUMAN_TAKEOVER, S.WON, actor=Actor.HUMAN)


def test_takeover_only_from_conversation_stages() -> None:
    with pytest.raises(InvalidTransitionError, match="not allowed"):
        check_transition(S.NEW, S.HUMAN_TAKEOVER, actor=Actor.HUMAN)


def test_opt_out_always_allowed_from_non_terminal() -> None:
    for stage in DealStage:
        if stage in TERMINAL_STAGES:
            continue
        for actor in Actor:
            check_transition(stage, S.OPTED_OUT, actor=actor)


def test_terminal_stages_are_final() -> None:
    for stage in TERMINAL_STAGES:
        assert allowed_targets(stage) == frozenset()
        with pytest.raises(InvalidTransitionError, match="final"):
            check_transition(stage, S.NEGOTIATING, actor=Actor.HUMAN)


def test_same_stage_and_missing_reason_rejected() -> None:
    with pytest.raises(InvalidTransitionError, match="already"):
        check_transition(S.ENGAGED, S.ENGAGED, actor=Actor.AGENT)
    with pytest.raises(InvalidTransitionError, match="reason"):
        transition(deal(), S.RESEARCHED, actor=Actor.AGENT, reason="  ")


def test_transition_record_and_immutability() -> None:
    original = deal(S.ENGAGED)
    when = datetime(2026, 11, 25, tzinfo=UTC)
    updated, record = transition(
        original, S.NEGOTIATING, actor=Actor.AGENT, reason=" price discussion ", now=when
    )
    assert original.stage is S.ENGAGED
    assert updated.stage is S.NEGOTIATING
    assert updated.updated_at == when
    assert record.from_stage is S.ENGAGED
    assert record.reason == "price discussion"
    assert record.deal_id == original.id
