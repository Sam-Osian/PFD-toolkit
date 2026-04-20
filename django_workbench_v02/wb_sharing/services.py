from __future__ import annotations

from copy import deepcopy

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone
from django.utils.text import slugify

from wb_auditlog.services import log_audit_event
from wb_investigations.models import Investigation
from wb_runs.models import InvestigationRun, RunArtifact, RunEvent
from wb_workspaces.activity import is_human_view_request, should_update_last_viewed
from wb_workspaces.models import RevisionChangeType, WorkspaceVisibility, WorkspaceRevision
from wb_workspaces.services import create_workspace_for_user
from wb_workspaces.permissions import can_manage_shares
from wb_workspaces.revisions import capture_workspace_state, write_workspace_revision

from .models import ShareMode, WorkspaceShareLink


class WorkspaceShareLinkError(ValidationError):
    pass


def _next_copy_slug(*, actor, source_workspace, target_title: str = "") -> str:
    base = slugify(target_title or source_workspace.slug or source_workspace.title or "workbook-copy")[:80]
    if not base:
        base = "workbook-copy"

    candidate = base
    counter = 2
    while source_workspace.__class__.objects.filter(created_by=actor, slug=candidate).exists():
        suffix = f"-copy-{counter}"
        candidate = f"{base[: max(1, 100 - len(suffix))]}{suffix}"
        counter += 1
    return candidate


@transaction.atomic
def copy_share_link_to_workbook(*, actor, share_link: WorkspaceShareLink, target_title: str = "", request=None):
    source_workspace = share_link.workspace
    resolved_title = str(target_title or "").strip()[:255] or f"{source_workspace.title} (Copy)"
    target_slug = _next_copy_slug(
        actor=actor,
        source_workspace=source_workspace,
        target_title=resolved_title,
    )

    copied_workspace = create_workspace_for_user(
        user=actor,
        title=resolved_title,
        slug=target_slug,
        description=source_workspace.description,
        seed_initial_revision=False,
        request=request,
    )
    copied_workspace.visibility = WorkspaceVisibility.PRIVATE
    copied_workspace.is_listed = False
    copied_workspace.last_viewed_at = None
    copied_workspace.archived_at = None
    copied_workspace.save(
        update_fields=["visibility", "is_listed", "last_viewed_at", "archived_at", "updated_at"]
    )

    revision_map: dict[str, WorkspaceRevision] = {}
    for revision in source_workspace.revisions.order_by("revision_number", "created_at"):
        copied_revision = WorkspaceRevision.objects.create(
            workspace=copied_workspace,
            revision_number=revision.revision_number,
            state_json=deepcopy(revision.state_json),
            created_by=revision.created_by,
            change_type=revision.change_type,
            parent_revision=revision_map.get(str(revision.parent_revision_id))
            if revision.parent_revision_id
            else None,
            created_at=revision.created_at,
        )
        revision_map[str(revision.id)] = copied_revision

    source_current_revision = source_workspace.current_revision
    if source_current_revision is None:
        source_current_revision = source_workspace.revisions.order_by(
            "-revision_number",
            "-created_at",
        ).first()
    copied_workspace.current_revision = (
        revision_map.get(str(source_current_revision.id))
        if source_current_revision is not None
        else None
    )
    copied_workspace.save(update_fields=["current_revision", "updated_at"])

    source_investigation = (
        Investigation.objects.filter(workspace=source_workspace).order_by("-created_at").first()
    )
    copied_investigation = None
    run_map: dict[str, InvestigationRun] = {}
    if source_investigation is not None:
        copied_investigation = Investigation.objects.create(
            workspace=copied_workspace,
            created_by=source_investigation.created_by,
            title=source_investigation.title,
            question_text=source_investigation.question_text,
            scope_json=deepcopy(source_investigation.scope_json),
            method_json=deepcopy(source_investigation.method_json),
            status=source_investigation.status,
            last_viewed_at=None,
            created_at=source_investigation.created_at,
            updated_at=source_investigation.updated_at,
        )

        for source_run in source_investigation.runs.order_by("created_at", "queued_at"):
            copied_run = InvestigationRun.objects.create(
                investigation=copied_investigation,
                workspace=copied_workspace,
                requested_by=source_run.requested_by,
                run_type=source_run.run_type,
                status=source_run.status,
                progress_percent=source_run.progress_percent,
                queued_at=source_run.queued_at,
                started_at=source_run.started_at,
                finished_at=source_run.finished_at,
                cancel_requested_at=source_run.cancel_requested_at,
                cancel_requested_by=source_run.cancel_requested_by,
                cancel_reason=source_run.cancel_reason,
                worker_id=source_run.worker_id,
                error_code=source_run.error_code,
                error_message=source_run.error_message,
                input_config_json=deepcopy(source_run.input_config_json),
                query_start_date=source_run.query_start_date,
                query_end_date=source_run.query_end_date,
                created_at=source_run.created_at,
                updated_at=source_run.updated_at,
            )
            run_map[str(source_run.id)] = copied_run

        for source_run in source_investigation.runs.order_by("created_at", "queued_at"):
            copied_run = run_map[str(source_run.id)]
            for event in source_run.events.order_by("created_at"):
                RunEvent.objects.create(
                    run=copied_run,
                    event_type=event.event_type,
                    message=event.message,
                    payload_json=deepcopy(event.payload_json),
                    created_at=event.created_at,
                )
            for artifact in source_run.artifacts.order_by("created_at"):
                RunArtifact.objects.create(
                    run=copied_run,
                    workspace=copied_workspace,
                    artifact_type=artifact.artifact_type,
                    status=artifact.status,
                    storage_backend=artifact.storage_backend,
                    storage_uri=artifact.storage_uri,
                    content_hash=artifact.content_hash,
                    size_bytes=artifact.size_bytes,
                    metadata_json=deepcopy(artifact.metadata_json),
                    expires_at=artifact.expires_at,
                    last_viewed_at=None,
                    created_at=artifact.created_at,
                    updated_at=artifact.updated_at,
                )

    log_audit_event(
        action_type="sharing.workbook_copied",
        target_type="workspace",
        target_id=str(copied_workspace.id),
        workspace=copied_workspace,
        user=actor,
        payload={
            "source_workspace_id": str(source_workspace.id),
            "share_link_id": str(share_link.id),
            "copied_investigation_id": str(copied_investigation.id) if copied_investigation else None,
            "copied_run_count": len(run_map),
            "copied_revision_count": len(revision_map),
        },
        request=request,
    )
    return copied_workspace


def _latest_workspace_revision(workspace):
    if workspace.current_revision_id:
        current = WorkspaceRevision.objects.filter(
            id=workspace.current_revision_id,
            workspace=workspace,
        ).first()
        if current is not None:
            return current
        workspace.current_revision = None
        workspace.save(update_fields=["current_revision", "updated_at"])
    latest = workspace.revisions.order_by("-revision_number").first()
    if latest is not None:
        workspace.current_revision = latest
        workspace.save(update_fields=["current_revision", "updated_at"])
        return latest
    return write_workspace_revision(
        workspace=workspace,
        actor=workspace.created_by,
        change_type=RevisionChangeType.SYSTEM,
        state_json=capture_workspace_state(workspace=workspace),
        payload={"action": "share_snapshot_seed"},
    )


def _validate_share_mode_requirements(*, mode: str, snapshot_revision):
    if mode == ShareMode.SNAPSHOT and snapshot_revision is None:
        raise WorkspaceShareLinkError(
            "Snapshot share links require a workspace revision. Create a revision first."
        )


@transaction.atomic
def create_share_link(
    *,
    actor,
    workspace,
    mode: str,
    is_public: bool,
    expires_at=None,
    request=None,
) -> WorkspaceShareLink:
    if not can_manage_shares(actor, workspace):
        raise PermissionDenied("Only owners with share-management access can create share links.")

    snapshot_revision = None
    if mode == ShareMode.SNAPSHOT:
        snapshot_revision = _latest_workspace_revision(workspace)
    _validate_share_mode_requirements(mode=mode, snapshot_revision=snapshot_revision)

    share_link = WorkspaceShareLink.objects.create(
        workspace=workspace,
        created_by=actor,
        mode=mode,
        snapshot_revision=snapshot_revision,
        is_public=is_public,
        is_active=True,
        expires_at=expires_at,
    )
    log_audit_event(
        action_type="sharing.link_created",
        target_type="workspace_share_link",
        target_id=str(share_link.id),
        workspace=workspace,
        user=actor,
        payload={
            "mode": share_link.mode,
            "is_public": share_link.is_public,
            "is_active": share_link.is_active,
            "expires_at": share_link.expires_at.isoformat() if share_link.expires_at else None,
            "snapshot_revision_id": str(share_link.snapshot_revision_id)
            if share_link.snapshot_revision_id
            else None,
        },
        request=request,
    )
    return share_link


@transaction.atomic
def update_share_link(
    *,
    actor,
    share_link: WorkspaceShareLink,
    mode: str,
    is_public: bool,
    is_active: bool,
    expires_at,
    request=None,
) -> WorkspaceShareLink:
    workspace = share_link.workspace
    if not can_manage_shares(actor, workspace):
        raise PermissionDenied("Only owners with share-management access can update share links.")

    before = {
        "mode": share_link.mode,
        "is_public": share_link.is_public,
        "is_active": share_link.is_active,
        "expires_at": share_link.expires_at.isoformat() if share_link.expires_at else None,
        "snapshot_revision_id": str(share_link.snapshot_revision_id)
        if share_link.snapshot_revision_id
        else None,
    }

    share_link.mode = mode
    share_link.is_public = is_public
    share_link.is_active = is_active
    share_link.expires_at = expires_at
    if mode == ShareMode.SNAPSHOT and share_link.snapshot_revision is None:
        share_link.snapshot_revision = _latest_workspace_revision(workspace)
    if mode == ShareMode.LIVE:
        share_link.snapshot_revision = None

    _validate_share_mode_requirements(mode=share_link.mode, snapshot_revision=share_link.snapshot_revision)
    share_link.save(
        update_fields=[
            "mode",
            "is_public",
            "is_active",
            "expires_at",
            "snapshot_revision",
            "updated_at",
        ]
    )

    log_audit_event(
        action_type="sharing.link_updated",
        target_type="workspace_share_link",
        target_id=str(share_link.id),
        workspace=workspace,
        user=actor,
        payload={
            "before": before,
            "after": {
                "mode": share_link.mode,
                "is_public": share_link.is_public,
                "is_active": share_link.is_active,
                "expires_at": share_link.expires_at.isoformat() if share_link.expires_at else None,
                "snapshot_revision_id": str(share_link.snapshot_revision_id)
                if share_link.snapshot_revision_id
                else None,
            },
        },
        request=request,
    )
    return share_link


@transaction.atomic
def revoke_share_link(*, actor, share_link: WorkspaceShareLink, request=None) -> WorkspaceShareLink:
    workspace = share_link.workspace
    if not can_manage_shares(actor, workspace):
        raise PermissionDenied("Only owners with share-management access can revoke share links.")
    if not share_link.is_active:
        return share_link

    share_link.is_active = False
    share_link.save(update_fields=["is_active", "updated_at"])
    log_audit_event(
        action_type="sharing.link_revoked",
        target_type="workspace_share_link",
        target_id=str(share_link.id),
        workspace=workspace,
        user=actor,
        payload={"is_public": share_link.is_public, "mode": share_link.mode},
        request=request,
    )
    return share_link


def record_share_link_view(*, share_link: WorkspaceShareLink, user=None, request=None) -> None:
    now = timezone.now()
    is_human_view = is_human_view_request(request=request)
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=share_link.last_viewed_at,
        now=now,
    ):
        share_link.last_viewed_at = now
        share_link.save(update_fields=["last_viewed_at"])

    workspace = share_link.workspace
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=workspace.last_viewed_at,
        now=now,
    ):
        workspace.last_viewed_at = now
        workspace.save(update_fields=["last_viewed_at"])

    log_audit_event(
        action_type="sharing.link_viewed",
        target_type="workspace_share_link",
        target_id=str(share_link.id),
        workspace=workspace,
        user=user if user and user.is_authenticated else None,
        payload={
            "is_public": share_link.is_public,
            "mode": share_link.mode,
            "is_human_view": is_human_view,
        },
        request=request,
    )
