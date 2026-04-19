from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings


class WorkspaceCredentialError(RuntimeError):
    pass


def _resolved_fernet_key() -> bytes:
    configured = (getattr(settings, "CREDENTIAL_ENCRYPTION_KEY", "") or "").strip()
    if configured:
        return configured.encode("utf-8")

    # Fallback keeps local development simple when no dedicated key is set.
    digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet() -> Fernet:
    try:
        return Fernet(_resolved_fernet_key())
    except Exception as exc:  # pragma: no cover - defensive configuration guard
        raise WorkspaceCredentialError("Invalid credential encryption key configuration.") from exc


def encrypt_secret(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceCredentialError("Cannot encrypt an empty secret.")
    token = _fernet().encrypt(value.strip().encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(token: str) -> str:
    if not isinstance(token, str) or not token.strip():
        raise WorkspaceCredentialError("Cannot decrypt an empty token.")
    try:
        decrypted = _fernet().decrypt(token.encode("utf-8"))
    except InvalidToken as exc:
        raise WorkspaceCredentialError("Credential decryption failed.") from exc
    return decrypted.decode("utf-8")
