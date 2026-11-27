"""Unit tests for app.telemetry.events."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.telemetry.events import ActivityEvent, EventBus, EventType, InMemorySink, Severity


def event(**overrides: object) -> ActivityEvent:
    data: dict[str, object] = {
        "type": EventType.MESSAGE_SENT,
        "code": "message.sent",
        "summary": "Offer sent to Retail Co",
        "params": {"company": "Retail Co", "amount": 9500},
    }
    data.update(overrides)
    return ActivityEvent(**data)  # type: ignore[arg-type]


def test_push_rules() -> None:
    assert not event().should_push
    assert event(push=True).should_push
    assert event(type=EventType.APPROVAL_REQUESTED).should_push
    assert event(type=EventType.DEAL_WON, severity=Severity.SUCCESS).should_push
    assert event(severity=Severity.CRITICAL).should_push


@pytest.mark.parametrize(
    "params",
    [
        {"api_key": "x"},
        {"authToken": "x"},
        {"db_password": "x"},
        {"bad name!": "x"},
        {"note": "x" * 501},
    ],
)
def test_params_reject_secrets_and_bad_values(params: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        event(params=params)


def test_invalid_code_rejected() -> None:
    with pytest.raises(ValidationError):
        event(code="Message Sent!")


class BrokenSink:
    name = "push"

    async def publish(self, event: ActivityEvent) -> None:
        raise ConnectionError("push provider down")


def test_failing_sink_does_not_block_others() -> None:
    first, last = InMemorySink("db"), InMemorySink("feed")
    bus = EventBus([first, BrokenSink(), last])
    failed = asyncio.run(bus.publish(event()))
    assert failed == ["push"]
    assert len(first.events) == 1
    assert len(last.events) == 1


def test_memory_sink_capacity_and_filter() -> None:
    sink = InMemorySink(capacity=3)

    async def fill() -> None:
        for i in range(5):
            await sink.publish(event(summary=f"event {i}"))
        await sink.publish(event(type=EventType.DEAL_WON, summary="won"))

    asyncio.run(fill())
    assert len(sink.events) == 3
    assert sink.events[-1].summary == "won"
    assert len(sink.of_type(EventType.DEAL_WON)) == 1
