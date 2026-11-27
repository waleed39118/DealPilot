"""
Shared verdict types for every compliance check.

Precedence when several checks apply: DENY > REQUIRE_APPROVAL > DEFER > ALLOW.
Reason ``code`` values are stable identifiers; the desktop and mobile apps translate
them (ar/en). ``message`` is an internal English explanation for logs and audits.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Verdict(StrEnum):
    ALLOW = "allow"
    DEFER = "defer"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


_RANK = {Verdict.ALLOW: 0, Verdict.DEFER: 1, Verdict.REQUIRE_APPROVAL: 2, Verdict.DENY: 3}


@dataclass(frozen=True, slots=True)
class Reason:
    verdict: Verdict
    code: str
    message: str
    not_before: datetime | None = None  # only meaningful for DEFER


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reasons: tuple[Reason, ...]
    not_before: datetime | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    @classmethod
    def combine(cls, reasons: Iterable[Reason]) -> Decision:
        collected = tuple(reasons)
        if not collected:
            return cls(Verdict.ALLOW, ())
        verdict = max((reason.verdict for reason in collected), key=_RANK.__getitem__)
        deferrals = [r.not_before for r in collected if r.not_before is not None]
        not_before = max(deferrals) if verdict is Verdict.DEFER and deferrals else None
        return cls(verdict, collected, not_before)


def deny(code: str, message: str) -> Reason:
    return Reason(Verdict.DENY, code, message)


def needs_approval(code: str, message: str) -> Reason:
    return Reason(Verdict.REQUIRE_APPROVAL, code, message)


def defer(code: str, message: str, not_before: datetime) -> Reason:
    return Reason(Verdict.DEFER, code, message, not_before)
