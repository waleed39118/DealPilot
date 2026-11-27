"""Shared builders for domain tests."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.domain.models import (
    FAQ,
    DealPolicy,
    Language,
    Objection,
    PaymentTerm,
    ProjectBrief,
)

TENANT: UUID = uuid4()


def brief_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "Smart Inventory Suite",
        "summary": "Cloud inventory management for mid-size retailers.",
        "unique_value_proposition": "Cuts stock-outs by 30% with demand forecasting.",
        "target_segments": ["Retail chains (10-200 stores)"],
        "faqs": [FAQ(question=f"Question {i}?", answer=f"Answer {i}.") for i in range(3)],
        "expected_objections": [
            Objection(objection=f"Objection {i}", approved_response=f"Response {i}.")
            for i in range(3)
        ],
        "languages": {Language.AR, Language.EN},
    }
    data.update(overrides)
    return data


def policy_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "currency": "sar",
        "list_price": Decimal("10000"),
        "price_floor": Decimal("8000"),
        "max_auto_discount_pct": Decimal("10"),
        "allowed_payment_terms": {PaymentTerm.UPFRONT, PaymentTerm.SPLIT_50_50},
        "approval_deal_value_above": Decimal("25000"),
    }
    data.update(overrides)
    return data


@pytest.fixture
def brief() -> ProjectBrief:
    return ProjectBrief(**brief_data())


@pytest.fixture
def policy() -> DealPolicy:
    return DealPolicy(**policy_data())
