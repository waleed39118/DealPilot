"""
Core domain models.

All models are immutable and reject unknown fields, so data coming from the API, the
database or a language model cannot smuggle extra attributes in. Text is trimmed and
length-bounded to keep prompts and storage predictable.

* ``ProjectBrief``  — everything the agent may say about a product. Mandatory sections
                     (value proposition, segments, FAQs, objections) must be complete.
* ``DealPolicy``    — the agent's negotiation authority. Binding offers and contracts
                     always require approval; that cannot be switched off.
* ``Prospect``      — a contact with recorded data source and lawful basis (GDPR).
* ``Consent``       — channel-specific permission with evidence.
* ``Deal``          — one prospect for one project, driven by ``deal_state``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.domain.deal_state import DealStage

_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(_require_aware)]
ShortText = Annotated[str, Field(min_length=1, max_length=200)]
MediumText = Annotated[str, Field(min_length=1, max_length=2000)]
LongText = Annotated[str, Field(min_length=1, max_length=5000)]
Amount = Annotated[Decimal, Field(ge=0, max_digits=14, decimal_places=2)]
Percent = Annotated[Decimal, Field(ge=0, le=100, max_digits=5, decimal_places=2)]


class DomainModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )


# ── Enumerations ────────────────────────────────────────────────────────────


class Language(StrEnum):
    AR = "ar"
    EN = "en"


class Channel(StrEnum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    FAX = "fax"


class PaymentTerm(StrEnum):
    UPFRONT = "upfront"
    NET_15 = "net_15"
    NET_30 = "net_30"
    NET_60 = "net_60"
    SPLIT_50_50 = "split_50_50"
    MILESTONES = "milestones"


class LawfulBasis(StrEnum):
    """GDPR Art. 6 basis recorded for every contact."""

    LEGITIMATE_INTEREST = "legitimate_interest"
    CONSENT = "consent"
    CONTRACT = "contract"


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


# ── Value objects ───────────────────────────────────────────────────────────


class Money(DomainModel):
    amount: Amount
    currency: str

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        value = value.upper()
        if not _CURRENCY_RE.match(value):
            raise ValueError("currency must be an ISO 4217 code, e.g. SAR or USD")
        return value


class QuietHours(DomainModel):
    """Local-time window in which no outreach is sent. May wrap past midnight."""

    start: time = time(20, 0)
    end: time = time(9, 0)

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if self.start == self.end:
            raise ValueError("quiet hours start and end must differ")
        return self

    def contains(self, moment: time) -> bool:
        if self.start < self.end:
            return self.start <= moment < self.end
        return moment >= self.start or moment < self.end


class FAQ(DomainModel):
    question: MediumText
    answer: LongText


class Objection(DomainModel):
    objection: MediumText
    approved_response: LongText


# ── Project brief & negotiation authority ───────────────────────────────────


class ProjectBrief(DomainModel):
    """The only source of truth the agent may use when talking about a product."""

    name: ShortText
    summary: LongText
    unique_value_proposition: LongText
    target_segments: list[ShortText] = Field(min_length=1, max_length=20)
    faqs: list[FAQ] = Field(min_length=3, max_length=50)
    expected_objections: list[Objection] = Field(min_length=3, max_length=50)
    proof_points: list[MediumText] = Field(default_factory=list, max_length=30)
    forbidden_claims: list[MediumText] = Field(default_factory=list, max_length=30)
    languages: frozenset[Language] = Field(min_length=1)


class DealPolicy(DomainModel):
    """Hard limits on what the agent may offer without asking the user."""

    currency: str
    list_price: Amount
    price_floor: Amount
    max_auto_discount_pct: Percent = Decimal("10")
    allowed_payment_terms: frozenset[PaymentTerm] = Field(min_length=1)
    approval_deal_value_above: Amount | None = None
    require_approval_first_message: bool = True
    # Structural guarantees: the type only admits ``True``.
    require_approval_binding_offer: Literal[True] = True
    require_approval_contract: Literal[True] = True
    max_touches_per_prospect: int = Field(default=3, ge=1, le=10)
    min_hours_between_touches: int = Field(default=72, ge=24, le=720)
    quiet_hours: QuietHours = Field(default_factory=QuietHours)

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        return Money(amount=Decimal(0), currency=value).currency

    @model_validator(mode="after")
    def _prices_consistent(self) -> Self:
        if self.list_price <= 0:
            raise ValueError("list_price must be greater than zero")
        if self.price_floor <= 0 or self.price_floor > self.list_price:
            raise ValueError("price_floor must be greater than zero and not above list_price")
        return self

    @property
    def lowest_auto_price(self) -> Decimal:
        """Lowest price the agent may offer on its own: discount cap, never below floor."""
        discounted = self.list_price * (Decimal(100) - self.max_auto_discount_pct) / 100
        return max(self.price_floor, discounted.quantize(Decimal("0.01")))


class Project(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    brief: ProjectBrief
    policy: DealPolicy
    status: ProjectStatus = ProjectStatus.DRAFT
    created_at: AwareDatetime = Field(default_factory=_utcnow)


# ── Prospects & consent ─────────────────────────────────────────────────────


class Prospect(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    full_name: ShortText
    job_title: ShortText | None = None
    company_name: ShortText
    email: str | None = None
    phone: str | None = None
    country: str
    timezone: str
    language: Language = Language.EN
    data_source: ShortText = Field(description="Licensed provider the record came from.")
    source_reference: ShortText | None = None
    lawful_basis: LawfulBasis = LawfulBasis.LEGITIMATE_INTEREST
    collected_at: AwareDatetime = Field(default_factory=_utcnow)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if not _EMAIL_RE.match(value) or len(value) > 254:
            raise ValueError("invalid email address")
        return value

    @field_validator("phone")
    @classmethod
    def _phone(cls, value: str | None) -> str | None:
        if value is not None and not _E164_RE.match(value):
            raise ValueError("phone must be in E.164 format, e.g. +9665XXXXXXXX")
        return value

    @field_validator("country")
    @classmethod
    def _country(cls, value: str) -> str:
        value = value.upper()
        if not _COUNTRY_RE.match(value):
            raise ValueError("country must be an ISO 3166-1 alpha-2 code, e.g. SA")
        return value

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be an IANA name, e.g. Asia/Riyadh") from exc
        return value

    @model_validator(mode="after")
    def _reachable(self) -> Self:
        if self.email is None and self.phone is None:
            raise ValueError("a prospect needs at least an email or a phone number")
        return self

    def local_time(self, at: datetime) -> time:
        return _require_aware(at).astimezone(ZoneInfo(self.timezone)).time()


class Consent(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    prospect_id: UUID
    channel: Channel
    granted_at: AwareDatetime
    evidence: MediumText = Field(description="How consent was captured (form, reply, call).")
    revoked_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _order(self) -> Self:
        if self.revoked_at is not None and self.revoked_at < self.granted_at:
            raise ValueError("revoked_at cannot be before granted_at")
        return self

    def is_active(self, at: datetime) -> bool:
        moment = _require_aware(at)
        return self.granted_at <= moment and (self.revoked_at is None or moment < self.revoked_at)


# ── Deal ────────────────────────────────────────────────────────────────────


class Deal(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    project_id: UUID
    prospect_id: UUID
    stage: DealStage = DealStage.NEW
    agreed_price: Money | None = None
    payment_term: PaymentTerm | None = None
    touches_without_reply: int = Field(default=0, ge=0)
    last_touch_at: AwareDatetime | None = None
    version: int = Field(default=1, ge=1)
    created_at: AwareDatetime = Field(default_factory=_utcnow)
    updated_at: AwareDatetime = Field(default_factory=_utcnow)
