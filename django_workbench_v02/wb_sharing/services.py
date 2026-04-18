from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_audit_event
from wb_workspaces.permissions import can_manage_shares

from .models import ShareMode, WorkspaceShareLink


class WorkspaceShareLinkError(ValidationError):
    pass


def _latest_workspace_revision(workspace):
    return workspace.revisions.order_by("-revision_number").first()


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
    share_link.last_viewed_at = now
    share_link.save(update_fields=["last_viewed_at"])
    workspace = share_link.workspace
    workspace.last_viewed_at = now
    workspace.save(update_fields=["last_viewed_at"])

    log_audit_event(
        action_type="sharing.link_viewed",
        target_type="workspace_share_link",
        target_id=str(share_link.id),
        workspace=workspace,
        user=user if user and user.is_authenticated else None,
        payload={"is_public": share_link.is_public, "mode": share_link.mode},
        request=request,
    )
