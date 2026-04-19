from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_audit_event
from wb_workspaces.activity import is_human_view_request, should_update_last_viewed
from wb_workspaces.permissions import can_edit_workspace

from .models import Investigation


class InvestigationServiceError(ValidationError):
    pass


@transaction.atomic
def create_investigation(
    *,
    actor,
    workspace,
    title: str,
    question_text: str,
    scope_json: dict | None,
    method_json: dict | None,
    status: str,
    request=None,
) -> Investigation:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to create investigations.")

    investigation = Investigation.objects.create(
        workspace=workspace,
        created_by=actor,
        title=title,
        question_text=question_text or "",
        scope_json=scope_json or {},
        method_json=method_json or {},
        status=status,
    )
    log_audit_event(
        action_type="investigation.created",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=workspace,
        user=actor,
        payload={"title": investigation.title, "status": investigation.status},
        request=request,
    )
    return investigation


@transaction.atomic
def update_investigation(
    *,
    actor,
    investigation: Investigation,
    title: str,
    question_text: str,
    scope_json: dict | None,
    method_json: dict | None,
    status: str,
    request=None,
) -> Investigation:
    if not can_edit_workspace(actor, investigation.workspace):
        raise PermissionDenied("You do not have permission to update this investigation.")

    before = {
        "title": investigation.title,
        "question_text": investigation.question_text,
        "scope_json": investigation.scope_json,
        "method_json": investigation.method_json,
        "status": investigation.status,
    }

    investigation.title = title
    investigation.question_text = question_text or ""
    investigation.scope_json = scope_json or {}
    investigation.method_json = method_json or {}
    investigation.status = status
    investigation.save(
        update_fields=[
            "title",
            "question_text",
            "scope_json",
            "method_json",
            "status",
            "updated_at",
        ]
    )

    log_audit_event(
        action_type="investigation.updated",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=investigation.workspace,
        user=actor,
        payload={
            "before": before,
            "after": {
                "title": investigation.title,
                "question_text": investigation.question_text,
                "scope_json": investigation.scope_json,
                "method_json": investigation.method_json,
                "status": investigation.status,
            },
        },
        request=request,
    )
    return investigation


def record_investigation_view(*, investigation: Investigation, user=None, request=None) -> None:
    now = timezone.now()
    is_human_view = is_human_view_request(request=request)
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=investigation.last_viewed_at,
        now=now,
    ):
        investigation.last_viewed_at = now
        investigation.save(update_fields=["last_viewed_at"])

    workspace = investigation.workspace
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=workspace.last_viewed_at,
        now=now,
    ):
        workspace.last_viewed_at = now
        workspace.save(update_fields=["last_viewed_at"])

    log_audit_event(
        action_type="investigation.viewed",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=investigation.workspace,
        user=user if user and user.is_authenticated else None,
        payload={
            "status": investigation.status,
            "is_human_view": is_human_view,
        },
        request=request,
    )
