"""
Mandatory disclosures appended to outbound communication (Arabic / English).

* First contact on any written channel states, in neutral wording, that the sender is
  an automated assistant acting on behalf of the company (EU AI Act Art. 50,
  California B.O.T. Act), and where the contact data came from (GDPR Art. 14).
* Every email carries the company's legal name, postal address, privacy policy and a
  working unsubscribe link (CAN-SPAM, GDPR).
* WhatsApp first contact explains how to stop messages.
* Voice calls open with a spoken identification statement.

Disclosures are appended by the delivery pipeline, never left to the language model,
so they cannot be omitted or reworded by a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.domain.models import Channel, Language

_SEPARATOR = "\n\n—\n"


@dataclass(frozen=True, slots=True)
class SenderIdentity:
    company_legal_name: str
    postal_address: str
    privacy_policy_url: str

    def __post_init__(self) -> None:
        if not self.company_legal_name.strip() or not self.postal_address.strip():
            raise ValueError("company legal name and postal address are required")
        if not self.privacy_policy_url.startswith("https://"):
            raise ValueError("privacy policy URL must be an https:// URL")

    @classmethod
    def from_settings(cls, settings: Settings) -> SenderIdentity:
        return cls(
            company_legal_name=settings.company_legal_name,
            postal_address=settings.company_postal_address,
            privacy_policy_url=settings.privacy_policy_url,
        )


_TEXT: dict[Language, dict[str, str]] = {
    Language.AR: {
        "assistant": "هذه رسالة من مساعد آلي يتواصل نيابة عن {company}.",
        "source": "حصلنا على بيانات التواصل المهنية الخاصة بك من {source}.",
        "unsubscribe": "لإلغاء الاشتراك وعدم استلام رسائل أخرى: {url}",
        "address": "{company} — {address}",
        "privacy": "سياسة الخصوصية: {url}",
        "whatsapp_stop": "للتوقف عن استلام الرسائل أرسل كلمة «إيقاف» أو STOP.",
        "voice": (
            "مرحباً، معك مساعد آلي يتصل نيابة عن {company}. "
            "يمكنك في أي وقت طلب التحدث مع شخص، أو طلب عدم الاتصال بك مجدداً."
        ),
    },
    Language.EN: {
        "assistant": "This message is from an automated assistant contacting you on behalf of "
        "{company}.",
        "source": "We obtained your professional contact details from {source}.",
        "unsubscribe": "To unsubscribe from further messages: {url}",
        "address": "{company} — {address}",
        "privacy": "Privacy policy: {url}",
        "whatsapp_stop": "Reply STOP at any time to stop receiving messages.",
        "voice": (
            "Hello, this is an automated assistant calling on behalf of {company}. "
            "At any time you can ask to speak with a person or ask not to be called again."
        ),
    },
}


def build_footer(
    *,
    language: Language,
    channel: Channel,
    identity: SenderIdentity,
    is_first_contact: bool,
    data_source: str | None = None,
    unsubscribe_url: str | None = None,
) -> str:
    """The disclosure block for one message (empty string when none is required)."""
    text = _TEXT[language]
    lines: list[str] = []

    if is_first_contact and channel is not Channel.VOICE:
        lines.append(text["assistant"].format(company=identity.company_legal_name))
        if data_source and data_source.strip():
            lines.append(text["source"].format(source=data_source.strip()))

    if channel is Channel.EMAIL:
        if not unsubscribe_url or not unsubscribe_url.startswith("https://"):
            raise ValueError("every email requires an https:// unsubscribe URL")
        lines.append(text["unsubscribe"].format(url=unsubscribe_url))
        lines.append(
            text["address"].format(
                company=identity.company_legal_name, address=identity.postal_address
            )
        )
        lines.append(text["privacy"].format(url=identity.privacy_policy_url))
    elif channel is Channel.WHATSAPP and is_first_contact:
        lines.append(text["whatsapp_stop"])
    elif channel is Channel.FAX:
        lines.append(
            text["address"].format(
                company=identity.company_legal_name, address=identity.postal_address
            )
        )
    return "\n".join(lines)


def apply_disclosure(
    body: str,
    *,
    language: Language,
    channel: Channel,
    identity: SenderIdentity,
    is_first_contact: bool,
    data_source: str | None = None,
    unsubscribe_url: str | None = None,
) -> str:
    """Append the required footer once; calling it again on the result is a no-op."""
    if channel is Channel.VOICE:
        raise ValueError("voice calls use voice_opening_statement(), not a written footer")
    footer = build_footer(
        language=language,
        channel=channel,
        identity=identity,
        is_first_contact=is_first_contact,
        data_source=data_source,
        unsubscribe_url=unsubscribe_url,
    )
    if not footer:
        return body
    block = f"{_SEPARATOR}{footer}"
    if body.endswith(block):
        return body
    return f"{body.rstrip()}{block}"


def voice_opening_statement(language: Language, identity: SenderIdentity) -> str:
    """Spoken at the very start of every outbound or inbound AI call."""
    return _TEXT[language]["voice"].format(company=identity.company_legal_name)
