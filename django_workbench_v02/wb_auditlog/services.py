from __future__ import annotations

import hashlib

from django.http import HttpRequest

from .models import ActionCacheEvent, AuditEvent


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


def log_action_cache_event(
    *,
    workspace,
    action_key: str,
    entity_type: str,
    entity_id: str,
    user=None,
    query: dict | None = None,
    options: dict | None = None,
    state_before: dict | None = None,
    state_after: dict | None = None,
    context: dict | None = None,
) -> ActionCacheEvent:
    return ActionCacheEvent.objects.create(
        workspace=workspace,
        user=user if user and getattr(user, "is_authenticated", False) else None,
        action_key=action_key,
        entity_type=entity_type,
        entity_id=str(entity_id),
        query_json=query or {},
        options_json=options or {},
        state_before_json=state_before or {},
        state_after_json=state_after or {},
        context_json=context or {},
    )
