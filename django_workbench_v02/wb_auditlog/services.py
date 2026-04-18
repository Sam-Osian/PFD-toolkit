from __future__ import annotations

import hashlib

from django.http import HttpRequest

from .models import AuditEvent


def _extract_ip(request: HttpRequest) -> str:
    forwarded_for = (request.META.get("HTTP_X_FORWARDED_FOR") or "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return (request.META.get("REMOTE_ADDR") or "").strip()


def _hash_ip(ip: str) -> str | None:
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()


def log_audit_event(
    *,
    action_type: str,
    target_type: str,
    target_id: str,
    workspace=None,
    user=None,
    payload: dict | None = None,
    request: HttpRequest | None = None,
) -> AuditEvent:
    ip_hash = None
    user_agent = None
    if request is not None:
        ip_hash = _hash_ip(_extract_ip(request))
        user_agent = request.META.get("HTTP_USER_AGENT")

    return AuditEvent.objects.create(
        workspace=workspace,
        user=user,
        action_type=action_type,
        target_type=target_type,
        target_id=str(target_id),
        ip_hash=ip_hash,
        user_agent=user_agent,
        payload_json=payload or {},
    )
