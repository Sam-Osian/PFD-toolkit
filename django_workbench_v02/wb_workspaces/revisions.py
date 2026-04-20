from __future__ import annotations

from copy import deepcopy

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction

from wb_auditlog.services import log_action_cache_event, log_audit_event
from wb_investigations.models import Investigation

from .models import (
    RevisionChangeType,
    Workspace,
    WorkspaceReportExclusion,
    WorkspaceRevision,
)
from .permissions import can_edit_workspace


class WorkspaceRevisionError(ValidationError):
    pass


def _normalize_exclusion_payload(exclusion: WorkspaceReportExclusion) -> dict:
    return {
        "report_identity": exclusion.report_identity,
        "reason": exclusion.reason,
        "report_title": exclusion.report_title,
        "report_date": exclusion.report_date,
        "report_url": exclusion.report_url,
    }


def capture_workspace_state(*, workspace: Workspace) -> dict:
    investigation = (
        Investigation.objects.filter(workspace=workspace)
        .order_by("-created_at")
        .first()
    )
    exclusions = WorkspaceReportExclusion.objects.filter(workspace=workspace).order_by(
        "created_at",
        "report_identity",
    )
    return {
        "version": 1,
        "investigation": (
            {
                "title": investigation.title,
                "question_text": investigation.question_text,
                "scope_json": deepcopy(investigation.scope_json or {}),
                "method_json": deepcopy(investigation.method_json or {}),
                "status": investigation.status,
            }
            if investigation is not None
            else None
        ),
        "report_exclusions": [_normalize_exclusion_payload(row) for row in exclusions],
    }


def _next_revision_number(*, workspace: Workspace) -> int:
    latest = workspace.revisions.order_by("-revision_number").first()
    if latest is None:
        return 1
    return int(latest.revision_number) + 1


@transaction.atomic
def write_workspace_revision(
    *,
    workspace: Workspace,
    actor=None,
    change_type: str = RevisionChangeType.EDIT,
    state_json: dict | None = None,
    request=None,
    payload: dict | None = None,
) -> WorkspaceRevision:
    parent_revision = workspace.current_revision
    revision = WorkspaceRevision.objects.create(
        workspace=workspace,
        revision_number=_next_revision_number(workspace=workspace),
        state_json=state_json if state_json is not None else capture_workspace_state(workspace=workspace),
        created_by=actor if actor and getattr(actor, "is_authenticated", False) else None,
        change_type=change_type,
        parent_revision=parent_revision,
    )
    workspace.current_revision = revision
    workspace.save(update_fields=["current_revision", "updated_at"])
    log_audit_event(
        action_type="workspace.revision_written",
        target_type="workspace_revision",
        target_id=str(revision.id),
        workspace=workspace,
        user=actor if actor and getattr(actor, "is_authenticated", False) else None,
        payload={
            "revision_id": str(revision.id),
            "revision_number": revision.revision_number,
            "change_type": revision.change_type,
            "parent_revision_id": str(parent_revision.id) if parent_revision else None,
            **(payload or {}),
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor if actor and getattr(actor, "is_authenticated", False) else None,
        action_key="revision.write",
        entity_type="workspace_revision",
        entity_id=str(revision.id),
        options={"change_type": revision.change_type},
        state_before={},
        state_after={
            "revision_number": revision.revision_number,
            "parent_revision_id": str(parent_revision.id) if parent_revision else None,
        },
        context=payload or {},
    )
    return revision


def _ensure_current_revision(*, workspace: Workspace) -> WorkspaceRevision:
    if workspace.current_revision is not None:
        return workspace.current_revision
    latest = workspace.revisions.order_by("-revision_number").first()
    if latest is None:
        latest = write_workspace_revision(
            workspace=workspace,
            actor=workspace.created_by,
            change_type=RevisionChangeType.SYSTEM,
        )
    workspace.current_revision = latest
    workspace.save(update_fields=["current_revision", "updated_at"])
    return latest


def _apply_workspace_state(
    *,
    workspace: Workspace,
    state_json: dict,
    actor,
) -> None:
    state = state_json if isinstance(state_json, dict) else {}
    investigation_state = state.get("investigation")
    investigation = Investigation.objects.filter(workspace=workspace).order_by("-created_at").first()

    if isinstance(investigation_state, dict):
        payload = {
            "title": str(investigation_state.get("title") or "Investigation").strip()[:255]
            or "Investigation",
            "question_text": str(investigation_state.get("question_text") or ""),
            "scope_json": deepcopy(investigation_state.get("scope_json") or {}),
            "method_json": deepcopy(investigation_state.get("method_json") or {}),
            "status": str(investigation_state.get("status") or "draft"),
        }
        if investigation is None:
            Investigation.objects.create(
                workspace=workspace,
                created_by=actor if actor and getattr(actor, "is_authenticated", False) else workspace.created_by,
                **payload,
            )
        else:
            investigation.title = payload["title"]
            investigation.question_text = payload["question_text"]
            investigation.scope_json = payload["scope_json"]
            investigation.method_json = payload["method_json"]
            investigation.status = payload["status"]
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
    elif investigation is not None:
        investigation.delete()

    WorkspaceReportExclusion.objects.filter(workspace=workspace).delete()
    raw_exclusions = state.get("report_exclusions")
    if not isinstance(raw_exclusions, list):
        return
    for row in raw_exclusions:
        if not isinstance(row, dict):
            continue
        identity = str(row.get("report_identity") or "").strip()
        if not identity:
            continue
        WorkspaceReportExclusion.objects.create(
            workspace=workspace,
            report_identity=identity,
            reason=str(row.get("reason") or "").strip(),
            report_title=str(row.get("report_title") or "").strip(),
            report_date=str(row.get("report_date") or "").strip(),
            report_url=str(row.get("report_url") or "").strip(),
            excluded_by=actor if actor and getattr(actor, "is_authenticated", False) else None,
        )


def _move_cursor_to_revision(
    *,
    workspace: Workspace,
    actor,
    target_revision: WorkspaceRevision,
    action_type: str,
    request=None,
    payload: dict | None = None,
) -> WorkspaceRevision:
    _apply_workspace_state(workspace=workspace, state_json=target_revision.state_json, actor=actor)
    workspace.current_revision = target_revision
    workspace.save(update_fields=["current_revision", "updated_at"])
    log_audit_event(
        action_type=action_type,
        target_type="workspace_revision",
        target_id=str(target_revision.id),
        workspace=workspace,
        user=actor if actor and getattr(actor, "is_authenticated", False) else None,
        payload={
            "revision_id": str(target_revision.id),
            "revision_number": target_revision.revision_number,
            **(payload or {}),
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor if actor and getattr(actor, "is_authenticated", False) else None,
        action_key=action_type,
        entity_type="workspace_revision",
        entity_id=str(target_revision.id),
        options={},
        state_before={},
        state_after={
            "revision_number": target_revision.revision_number,
            "change_type": target_revision.change_type,
        },
        context=payload or {},
    )
    return target_revision


@transaction.atomic
def undo_workspace_revision(*, actor, workspace: Workspace, request=None) -> WorkspaceRevision:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to edit this workbook.")
    current = _ensure_current_revision(workspace=workspace)
    if current.parent_revision is None:
        raise WorkspaceRevisionError("No earlier revision available to undo.")
    return _move_cursor_to_revision(
        workspace=workspace,
        actor=actor,
        target_revision=current.parent_revision,
        action_type="workspace.revision_undo",
        request=request,
    )


@transaction.atomic
def redo_workspace_revision(*, actor, workspace: Workspace, request=None) -> WorkspaceRevision:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to edit this workbook.")
    current = _ensure_current_revision(workspace=workspace)
    redo_target = current.child_revisions.order_by("-revision_number", "-created_at").first()
    if redo_target is None:
        raise WorkspaceRevisionError("No later revision available to redo.")
    return _move_cursor_to_revision(
        workspace=workspace,
        actor=actor,
        target_revision=redo_target,
        action_type="workspace.revision_redo",
        request=request,
    )


@transaction.atomic
def start_over_workspace_state(*, actor, workspace: Workspace, request=None) -> WorkspaceRevision:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to edit this workbook.")
    baseline = workspace.revisions.order_by("revision_number", "created_at").first()
    if baseline is None:
        raise WorkspaceRevisionError("No baseline revision is available for this workbook.")
    _apply_workspace_state(workspace=workspace, state_json=baseline.state_json, actor=actor)
    return write_workspace_revision(
        workspace=workspace,
        actor=actor,
        change_type=RevisionChangeType.RESTORE,
        state_json=capture_workspace_state(workspace=workspace),
        request=request,
        payload={
            "source_revision_id": str(baseline.id),
            "source_revision_number": baseline.revision_number,
            "action": "start_over",
        },
    )


@transaction.atomic
def revert_workspace_reports(*, actor, workspace: Workspace, request=None) -> WorkspaceRevision:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to edit this workbook.")
    deleted_count, _ = WorkspaceReportExclusion.objects.filter(workspace=workspace).delete()
    return write_workspace_revision(
        workspace=workspace,
        actor=actor,
        change_type=RevisionChangeType.RESTORE,
        state_json=capture_workspace_state(workspace=workspace),
        request=request,
        payload={
            "action": "revert_reports",
            "removed_exclusions": deleted_count,
        },
    )
