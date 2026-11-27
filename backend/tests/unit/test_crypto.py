"""Unit tests for app.core.security.crypto."""

from __future__ import annotations

import base64
import secrets

import pytest

from app.core.config import Settings
from app.core.security.crypto import (
    CryptoError,
    DataKey,
    EncryptionContext,
    MasterKeyRing,
    decrypt_field,
    decrypt_field_text,
    encrypt_field,
    field_key_id,
    is_encrypted_field,
)

KEK_V1 = secrets.token_bytes(32)
KEK_V2 = secrets.token_bytes(32)
CTX = EncryptionContext(
    tenant_id="tenant-a", purpose="provider_credentials.api_key", record_id="r1"
)


def ring(active: int = 1) -> MasterKeyRing:
    keys = {1: KEK_V1} if active == 1 else {1: KEK_V1, 2: KEK_V2}
    return MasterKeyRing(keys, active)


def dek(key_ring: MasterKeyRing | None = None, tenant: str = "tenant-a") -> DataKey:
    return (key_ring or ring()).generate_data_key(tenant_id=tenant, key_id="dek_001")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def tamper(token: str, index: int = -1) -> str:
    """Flip one bit of the decoded ciphertext (default: last byte of the GCM tag)."""
    prefix, key, blob = token.split(".")
    raw = bytearray(_unb64(blob))
    raw[index] ^= 0x01
    return f"{prefix}.{key}.{_b64(bytes(raw))}"


# ── Key ring ────────────────────────────────────────────────────────────────


def test_ring_validation() -> None:
    with pytest.raises(CryptoError, match="not in the ring"):
        MasterKeyRing({1: KEK_V1}, 2)
    with pytest.raises(CryptoError, match="32 bytes"):
        MasterKeyRing({1: b"short"}, 1)
    with pytest.raises(CryptoError, match=">= 1"):
        MasterKeyRing({0: KEK_V1}, 0)


def test_ring_repr_hides_keys() -> None:
    text = repr(ring(2))
    assert "versions=(1, 2)" in text
    assert KEK_V1.hex() not in text


def test_generate_and_unwrap_data_key() -> None:
    key_ring = ring()
    data_key = dek(key_ring)
    assert data_key.wrapped.startswith("dk1.1.")
    assert data_key.plaintext.hex() not in data_key.wrapped
    assert "plaintext" not in repr(data_key)

    restored = key_ring.unwrap(data_key.wrapped, tenant_id="tenant-a", key_id="dek_001")
    assert restored.plaintext == data_key.plaintext


def test_wrapped_key_is_bound_to_tenant_and_key_id() -> None:
    key_ring = ring()
    data_key = dek(key_ring)
    with pytest.raises(CryptoError, match="failed authentication"):
        key_ring.unwrap(data_key.wrapped, tenant_id="tenant-b", key_id="dek_001")
    with pytest.raises(CryptoError, match="failed authentication"):
        key_ring.unwrap(data_key.wrapped, tenant_id="tenant-a", key_id="dek_002")


def test_unwrap_with_wrong_master_key_fails() -> None:
    data_key = dek()
    other = MasterKeyRing({1: secrets.token_bytes(32)}, 1)
    with pytest.raises(CryptoError, match="failed authentication"):
        other.unwrap(data_key.wrapped, tenant_id="tenant-a", key_id="dek_001")


def test_master_key_rotation_rewraps_without_touching_data() -> None:
    old_ring = ring(1)
    data_key = dek(old_ring)
    token = encrypt_field("sk_live_secret", data_key, CTX)

    new_ring = ring(2)
    assert new_ring.needs_rewrap(data_key.wrapped)
    rewrapped = new_ring.rewrap(data_key.wrapped, tenant_id="tenant-a", key_id="dek_001")
    assert rewrapped.startswith("dk1.2.")
    assert not new_ring.needs_rewrap(rewrapped)

    after_rotation = MasterKeyRing({2: KEK_V2}, 2)
    restored = after_rotation.unwrap(rewrapped, tenant_id="tenant-a", key_id="dek_001")
    assert decrypt_field_text(token, restored, CTX) == "sk_live_secret"


def test_unknown_master_key_version() -> None:
    wrapped = ring(2).generate_data_key(tenant_id="t", key_id="k").wrapped
    with pytest.raises(CryptoError, match="version 2 is not available"):
        ring(1).unwrap(wrapped, tenant_id="t", key_id="k")


@pytest.mark.parametrize("bad", ["", "dk1.1", "xx.1.abc", "dk1.one.abc", "dk1.1.!!!!", "dk1.1.QQ"])
def test_malformed_wrapped_keys(bad: str) -> None:
    with pytest.raises(CryptoError, match="malformed"):
        ring().unwrap(bad, tenant_id="t", key_id="k")


def test_invalid_key_id_rejected() -> None:
    with pytest.raises(ValueError, match="key_id"):
        ring().generate_data_key(tenant_id="t", key_id="has.dot")


# ── Field encryption ────────────────────────────────────────────────────────


def test_field_round_trip_text_and_bytes() -> None:
    data_key = dek()
    token = encrypt_field("مرحبا — hello", data_key, CTX)
    assert is_encrypted_field(token)
    assert field_key_id(token) == "dek_001"
    assert decrypt_field_text(token, data_key, CTX) == "مرحبا — hello"

    binary = bytes(range(256))
    assert decrypt_field(encrypt_field(binary, data_key, CTX), data_key, CTX) == binary


def test_same_plaintext_gives_different_ciphertexts() -> None:
    data_key = dek()
    assert encrypt_field("same", data_key, CTX) != encrypt_field("same", data_key, CTX)


@pytest.mark.parametrize(
    "other_context",
    [
        EncryptionContext("tenant-a", "provider_credentials.api_key", "r2"),
        EncryptionContext("tenant-a", "prospects.email", "r1"),
    ],
)
def test_ciphertext_cannot_be_moved_to_another_record_or_column(
    other_context: EncryptionContext,
) -> None:
    data_key = dek()
    token = encrypt_field("secret", data_key, CTX)
    with pytest.raises(CryptoError, match="failed authentication"):
        decrypt_field(token, data_key, other_context)


@pytest.mark.parametrize("index", [0, 12, -1])  # nonce, ciphertext body, tag
def test_tampering_is_detected(index: int) -> None:
    data_key = dek()
    token = encrypt_field("secret", data_key, CTX)
    with pytest.raises(CryptoError, match="failed authentication"):
        decrypt_field(tamper(token, index), data_key, CTX)


def test_non_canonical_encoding_rejected() -> None:
    data_key = dek()
    prefix, key, blob = encrypt_field("secret", data_key, CTX).split(".")
    padded = f"{prefix}.{key}.{blob}=="
    with pytest.raises(CryptoError, match="malformed"):
        decrypt_field(padded, data_key, CTX)


def test_wrong_data_key_or_tenant_rejected() -> None:
    data_key = dek()
    token = encrypt_field("secret", data_key, CTX)
    other = ring().generate_data_key(tenant_id="tenant-a", key_id="dek_002")
    with pytest.raises(CryptoError, match="different data key"):
        decrypt_field(token, other, CTX)

    foreign = dek(tenant="tenant-b")
    with pytest.raises(CryptoError, match="different tenant"):
        encrypt_field("x", foreign, CTX)


def test_error_messages_never_contain_secrets() -> None:
    data_key = dek()
    token = encrypt_field("sk_live_secret", data_key, CTX)
    with pytest.raises(CryptoError) as exc_info:
        decrypt_field(tamper(token), data_key, CTX)
    message = str(exc_info.value)
    assert "sk_live_secret" not in message
    assert data_key.plaintext.hex() not in message


def test_invalid_utf8_reported() -> None:
    data_key = dek()
    token = encrypt_field(b"\xff\xfe", data_key, CTX)
    with pytest.raises(CryptoError, match="UTF-8"):
        decrypt_field_text(token, data_key, CTX)


def test_context_requires_tenant_and_purpose() -> None:
    with pytest.raises(ValueError):
        EncryptionContext(tenant_id="", purpose="x")


# ── Integration with settings ───────────────────────────────────────────────


def test_ring_from_settings_includes_previous_keys() -> None:
    settings = Settings(
        _env_file=None,
        master_encryption_key=base64.b64encode(KEK_V2).decode(),
        master_encryption_key_version=2,
        master_encryption_previous_keys=f"1:{base64.b64encode(KEK_V1).decode()}",
    )
    key_ring = MasterKeyRing.from_settings(settings)
    assert key_ring.active_version == 2
    assert key_ring.versions == (1, 2)
    old_wrapped = ring(1).generate_data_key(tenant_id="t", key_id="k").wrapped
    assert key_ring.unwrap(old_wrapped, tenant_id="t", key_id="k")
