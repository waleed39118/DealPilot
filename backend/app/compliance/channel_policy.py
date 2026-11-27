"""
Legal pre-conditions per outreach channel.

| Channel  | Requirement before any contact                                              |
|----------|-----------------------------------------------------------------------------|
| email    | an address; recorded consent where the country or lawful basis demands it   |
| whatsapp | a phone number + active WhatsApp opt-in (Meta Business Messaging policy)    |
| voice    | a phone number + active voice consent (US TCPA rules for artificial voices) |
| fax      | a fax number + an existing deal conversation (no unsolicited fax adverts)   |

For every channel: suppressed or opted-out contacts are never contacted.

``EMAIL_CONSENT_COUNTRIES`` is deliberately conservative and must be reviewed by
counsel for each target market before going live.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from app.compliance.decision import Reason, deny
from app.domain.deal_state import DealStage
from app.domain.models import Channel, Consent, Deal, LawfulBasis, Prospect

# Markets where unsolicited B2B email requires prior consent (opt-in regimes).
EMAIL_CONSENT_COUNTRIES = frozenset({"DE", "AT", "PL"})

# Stages that prove an existing, two-way business conversation.
_ESTABLISHED_STAGES = frozenset(
    {
        DealStage.ENGAGED,
        DealStage.NEGOTIATING,
        DealStage.AWAITING_APPROVAL,
        DealStage.HUMAN_TAKEOVER,
        DealStage.CONTRACT_SENT,
        DealStage.WON,
    }
)


def recipient_for(channel: Channel, prospect: Prospect) -> str | None:
    """The address a channel would deliver to, if the prospect has one."""
    if channel is Channel.EMAIL:
        return prospect.email
    if channel is Channel.FAX:
        return prospect.fax
    return prospect.phone


def has_active_consent(
    consents: Iterable[Consent], prospect: Prospect, channel: Channel, at: datetime
) -> bool:
    return any(
        consent.prospect_id == prospect.id
        and consent.tenant_id == prospect.tenant_id
        and consent.channel is channel
        and consent.is_active(at)
        for consent in consents
    )


def check_channel(
    channel: Channel,
    *,
    prospect: Prospect,
    deal: Deal,
    consents: Iterable[Consent],
    is_suppressed: bool,
    at: datetime,
) -> list[Reason]:
    """Return every reason the channel may not be used now (empty list = permitted)."""
    reasons: list[Reason] = []
    consent_list = list(consents)

    if is_suppressed or deal.stage is DealStage.OPTED_OUT:
        reasons.append(deny("contact_suppressed", "contact has opted out or is suppressed"))

    if recipient_for(channel, prospect) is None:
        reasons.append(
            deny(f"{channel.value}_address_missing", f"prospect has no {channel.value} address")
        )

    if channel is Channel.EMAIL:
        needs_consent = (
            prospect.country in EMAIL_CONSENT_COUNTRIES
            or prospect.lawful_basis is LawfulBasis.CONSENT
        )
        if needs_consent and not has_active_consent(consent_list, prospect, channel, at):
            reasons.append(
                deny(
                    "email_consent_required",
                    f"email to {prospect.country} contacts requires recorded consent",
                )
            )
    elif channel in (Channel.WHATSAPP, Channel.VOICE):
        if not has_active_consent(consent_list, prospect, channel, at):
            reasons.append(
                deny(
                    f"{channel.value}_consent_required",
                    f"{channel.value} contact requires an active recorded opt-in",
                )
            )
    elif channel is Channel.FAX and deal.stage not in _ESTABLISHED_STAGES:
        reasons.append(
            deny("fax_requires_existing_deal", "fax is only used with parties already in a deal")
        )
    return reasons


def eligible_channels(
    *,
    prospect: Prospect,
    deal: Deal,
    consents: Iterable[Consent],
    is_suppressed: bool,
    at: datetime,
) -> list[Channel]:
    """Channels that pass every legal pre-condition, in declaration order."""
    consent_list = list(consents)
    return [
        channel
        for channel in Channel
        if not check_channel(
            channel,
            prospect=prospect,
            deal=deal,
            consents=consent_list,
            is_suppressed=is_suppressed,
            at=at,
        )
    ]
