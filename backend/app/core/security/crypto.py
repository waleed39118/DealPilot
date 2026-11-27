"""
Envelope encryption at rest (AES-256-GCM).

Hierarchy
---------
* **KEK** (key-encryption key): 32-byte master key from configuration, versioned.
  Never touches business data directly.
* **DEK** (data-encryption key): random 32-byte key per tenant, stored only in wrapped
  (KEK-encrypted) form. Encrypts the actual fields: provider API keys, contact data,
  contracts.

Integrity & binding
-------------------
AES-GCM authenticates every ciphertext. Additional authenticated data (AAD) binds:
* a wrapped DEK to its ``tenant_id`` + ``key_id`` (a DEK cannot be moved to another tenant);
* a field ciphertext to its ``EncryptionContext`` (tenant, purpose, record) so a value
  copied into another row or column fails to decrypt instead of silently leaking.

Rotation
--------
* KEK rotation: ``MasterKeyRing.rewrap`` re-encrypts only the small wrapped DEKs; data
  stays untouched. Old KEK versions stay in the ring until every DEK is rewrapped.
* DEK rotation: create a new data key and re-encrypt fields lazily; tokens carry their
  ``key_id`` so old and new data coexist.

Token formats (ASCII, URL-safe, storable in TEXT columns)::

    wrapped DEK : dk1.<kek_version>.<base64url(nonce || ciphertext+tag)>
    field value : ef1.<key_id>.<base64url(nonce || ciphertext+tag)>

Nonces are 96-bit random values; with one DEK per tenant the number of encryptions per
key stays far below the 2^32 safety bound for random GCM nonces.

Note: Python cannot reliably wipe memory; plaintext keys are kept out of ``repr`` and
logs, and are held only for the duration of an operation by callers.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import Settings
from app.core.errors import PermanentError

KEY_BYTES = 32
NONCE_BYTES = 12
_TAG_BYTES = 16

_WRAPPED_PREFIX = "dk1"
_FIELD_PREFIX = "ef1"
_DEK_AAD_LABEL = b"dealpilot/dek/v1"
_FIELD_AAD_LABEL = b"dealpilot/field/v1"
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CryptoError(PermanentError):
    """Encryption/decryption failure. Messages never contain key or plaintext material."""


# ── Value objects ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EncryptionContext:
    """What a ciphertext belongs to. Must be identical at encrypt and decrypt time."""

    tenant_id: str
    purpose: str
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.purpose:
            raise ValueError("tenant_id and purpose are required")

    def to_aad(self) -> bytes:
        payload = json.dumps(
            {"tenant": self.tenant_id, "purpose": self.purpose, "record": self.record_id},
            sort_keys=True,
            separators=(",", ":"),
        )
        return _FIELD_AAD_LABEL + b"|" + payload.encode("utf-8")


@dataclass(frozen=True, slots=True)
class DataKey:
    """An unwrapped tenant DEK. ``wrapped`` is the only form that may be persisted."""

    key_id: str
    tenant_id: str
    wrapped: str
    plaintext: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id)
        if len(self.plaintext) != KEY_BYTES:
            raise CryptoError("data key must be 32 bytes")


# ── Master key ring (KEKs) ──────────────────────────────────────────────────


class MasterKeyRing:
    def __init__(self, keys: Mapping[int, bytes], active_version: int) -> None:
        if active_version not in keys:
            raise CryptoError(f"active master key version {active_version} is not in the ring")
        for version, key in keys.items():
            if version < 1:
                raise CryptoError("master key versions must be >= 1")
            if len(key) != KEY_BYTES:
                raise CryptoError(f"master key version {version} must be 32 bytes")
        self._ciphers = {version: AESGCM(key) for version, key in keys.items()}
        self.active_version = active_version

    @classmethod
    def from_settings(cls, settings: Settings) -> MasterKeyRing:
        return cls(settings.master_keys(), settings.master_encryption_key_version)

    def __repr__(self) -> str:
        return f"MasterKeyRing(active_version={self.active_version}, versions={self.versions})"

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(sorted(self._ciphers))

    # DEK lifecycle --------------------------------------------------------------
    def generate_data_key(self, *, tenant_id: str, key_id: str) -> DataKey:
        """Create a fresh random DEK and its wrapped (persistable) form."""
        _validate_key_id(key_id)
        if not tenant_id:
            raise ValueError("tenant_id is required")
        plaintext = AESGCM.generate_key(bit_length=256)
        wrapped = self._wrap(plaintext, tenant_id=tenant_id, key_id=key_id)
        return DataKey(key_id=key_id, tenant_id=tenant_id, wrapped=wrapped, plaintext=plaintext)

    def unwrap(self, wrapped: str, *, tenant_id: str, key_id: str) -> DataKey:
        version, blob = _parse_token(wrapped, _WRAPPED_PREFIX, "wrapped data key")
        try:
            kek_version = int(version)
        except ValueError as exc:
            raise CryptoError("malformed wrapped data key") from exc
        cipher = self._ciphers.get(kek_version)
        if cipher is None:
            raise CryptoError(f"master key version {kek_version} is not available")
        plaintext = _open(cipher, blob, _dek_aad(tenant_id, key_id), "wrapped data key")
        return DataKey(key_id=key_id, tenant_id=tenant_id, wrapped=wrapped, plaintext=plaintext)

    def needs_rewrap(self, wrapped: str) -> bool:
        version, _ = _parse_token(wrapped, _WRAPPED_PREFIX, "wrapped data key")
        return version != str(self.active_version)

    def rewrap(self, wrapped: str, *, tenant_id: str, key_id: str) -> str:
        """Re-encrypt a wrapped DEK under the active KEK (KEK rotation)."""
        data_key = self.unwrap(wrapped, tenant_id=tenant_id, key_id=key_id)
        return self._wrap(data_key.plaintext, tenant_id=tenant_id, key_id=key_id)

    def _wrap(self, plaintext: bytes, *, tenant_id: str, key_id: str) -> str:
        cipher = self._ciphers[self.active_version]
        blob = _seal(cipher, plaintext, _dek_aad(tenant_id, key_id))
        return f"{_WRAPPED_PREFIX}.{self.active_version}.{blob}"


# ── Field encryption (DEK) ──────────────────────────────────────────────────


def encrypt_field(value: str | bytes, data_key: DataKey, context: EncryptionContext) -> str:
    """Encrypt one value. Each call uses a new random nonce."""
    if context.tenant_id != data_key.tenant_id:
        raise CryptoError("data key belongs to a different tenant")
    plaintext = value.encode("utf-8") if isinstance(value, str) else value
    blob = _seal(AESGCM(data_key.plaintext), plaintext, context.to_aad())
    return f"{_FIELD_PREFIX}.{data_key.key_id}.{blob}"


def decrypt_field(token: str, data_key: DataKey, context: EncryptionContext) -> bytes:
    key_id, blob = _parse_token(token, _FIELD_PREFIX, "encrypted value")
    if key_id != data_key.key_id:
        raise CryptoError("encrypted value was produced with a different data key")
    if context.tenant_id != data_key.tenant_id:
        raise CryptoError("data key belongs to a different tenant")
    return _open(AESGCM(data_key.plaintext), blob, context.to_aad(), "encrypted value")


def decrypt_field_text(token: str, data_key: DataKey, context: EncryptionContext) -> str:
    try:
        return decrypt_field(token, data_key, context).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CryptoError("decrypted value is not valid UTF-8") from exc


def field_key_id(token: str) -> str:
    """Which data key a stored value needs (lets callers fetch the right DEK)."""
    key_id, _ = _parse_token(token, _FIELD_PREFIX, "encrypted value")
    return key_id


def is_encrypted_field(value: str) -> bool:
    return value.startswith(f"{_FIELD_PREFIX}.")


# ── Internals ───────────────────────────────────────────────────────────────


def _dek_aad(tenant_id: str, key_id: str) -> bytes:
    if not tenant_id:
        raise ValueError("tenant_id is required")
    _validate_key_id(key_id)
    return _DEK_AAD_LABEL + b"|" + tenant_id.encode("utf-8") + b"|" + key_id.encode("ascii")


def _validate_key_id(key_id: str) -> None:
    if not _KEY_ID_RE.match(key_id):
        raise ValueError("key_id must be 1-64 characters of [A-Za-z0-9_-]")


def _seal(cipher: AESGCM, plaintext: bytes, aad: bytes) -> str:
    nonce = os.urandom(NONCE_BYTES)
    return _b64encode(nonce + cipher.encrypt(nonce, plaintext, aad))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _open(cipher: AESGCM, blob: str, aad: bytes, label: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4))
    except (binascii.Error, ValueError) as exc:
        raise CryptoError(f"malformed {label}") from exc
    if len(raw) < NONCE_BYTES + _TAG_BYTES or _b64encode(raw) != blob:
        # The canonical-encoding check rejects alternative spellings of the same bytes,
        # so every stored ciphertext has exactly one valid textual form.
        raise CryptoError(f"malformed {label}")
    try:
        return cipher.decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], aad)
    except InvalidTag as exc:
        raise CryptoError(
            f"{label} failed authentication (wrong key, context or tampered)"
        ) from exc


def _parse_token(token: str, prefix: str, label: str) -> tuple[str, str]:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != prefix or not parts[1] or not parts[2]:
        raise CryptoError(f"malformed {label}")
    return parts[1], parts[2]
