from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.utils import timezone

from wb_auditlog.services import log_audit_event
from wb_runs.models import ArtifactStatus, RunArtifact

from .models import Workspace


@dataclass
class LifecycleMaintenanceResult:
    inactivity_days: int
    archive_retention_days: int
    artifacts_scanned: int = 0
    artifacts_expired: int = 0
    artifacts_expiry_refreshed: int = 0
    workspaces_scanned: int = 0
    workspaces_archived: int = 0
    archived_workspaces_scanned: int = 0
    workspaces_purged: int = 0


def _last_activity_at(*, last_viewed_at, created_at):
    return last_viewed_at or created_at


def run_lifecycle_maintenance(
    *,
    now=None,
    inactivity_days: int = 365,
    archive_retention_days: int = 60,
    dry_run: bool = False,
    archive_workspaces: bool = True,
    purge_archived_workspaces: bool = True,
) -> LifecycleMaintenanceResult:
    current_time = now or timezone.now()
    inactivity_window = timedelta(days=inactivity_days)
    archive_retention_window = timedelta(days=archive_retention_days)
    result = LifecycleMaintenanceResult(
        inactivity_days=inactivity_days,
        archive_retention_days=archive_retention_days,
    )

    ready_artifacts = RunArtifact.objects.select_related("workspace").filter(status=ArtifactStatus.READY)
    for artifact in ready_artifacts.iterator():
        result.artifacts_scanned += 1
        last_activity_at = _last_activity_at(
            last_viewed_at=artifact.last_viewed_at,
            created_at=artifact.created_at,
        )
        effective_expires_at = last_activity_at + inactivity_window
        is_inactive = effective_expires_at <= current_time

        if is_inactive:
            result.artifacts_expired += 1
            if dry_run:
                continue

            artifact.status = ArtifactStatus.EXPIRED
            artifact.expires_at = effective_expires_at
            artifact.save(update_fields=["status", "expires_at", "updated_at"])
            log_audit_event(
                action_type="run.artifact_expired_inactive",
                target_type="run_artifact",
                target_id=str(artifact.id),
                workspace=artifact.workspace,
                user=None,
                payload={
                    "run_id": str(artifact.run_id),
                    "artifact_type": artifact.artifact_type,
                    "last_activity_at": last_activity_at.isoformat(),
                    "expired_at": effective_expires_at.isoformat(),
                    "inactivity_days": inactivity_days,
                },
            )
            continue

        if artifact.expires_at != effective_expires_at:
            result.artifacts_expiry_refreshed += 1
            if dry_run:
                continue
            artifact.expires_at = effective_expires_at
            artifact.save(update_fields=["expires_at", "updated_at"])

    if not archive_workspaces and not purge_archived_workspaces:
        return result

    if archive_workspaces:
        active_workspaces = Workspace.objects.filter(archived_at__isnull=True)
        for workspace in active_workspaces.iterator():
            result.workspaces_scanned += 1
            last_activity_at = _last_activity_at(
                last_viewed_at=workspace.last_viewed_at,
                created_at=workspace.created_at,
            )
            if (last_activity_at + inactivity_window) > current_time:
                continue

            result.workspaces_archived += 1
            if dry_run:
                continue

            workspace.archived_at = current_time
            workspace.save(update_fields=["archived_at", "updated_at"])
            log_audit_event(
                action_type="workspace.auto_archived_inactive",
                target_type="workspace",
                target_id=str(workspace.id),
                workspace=workspace,
                user=None,
                payload={
                    "last_activity_at": last_activity_at.isoformat(),
                    "archived_at": current_time.isoformat(),
                    "inactivity_days": inactivity_days,
                },
            )

    if purge_archived_workspaces:
        from .models import WorkspaceUserState

        archived_workspaces = Workspace.objects.filter(archived_at__isnull=False)
        for workspace in archived_workspaces.iterator():
            result.archived_workspaces_scanned += 1
            if workspace.archived_at is None:
                continue
            if (workspace.archived_at + archive_retention_window) > current_time:
                continue
            result.workspaces_purged += 1
            if dry_run:
                continue

            workspace_id = str(workspace.id)
            workspace_title = workspace.title
            archived_at = workspace.archived_at
            WorkspaceUserState.objects.filter(active_workspace=workspace).update(active_workspace=None)
            workspace.delete()
            log_audit_event(
                action_type="workspace.auto_purged_archived",
                target_type="workspace",
                target_id=workspace_id,
                workspace=None,
                user=None,
                payload={
                    "workspace_title": workspace_title,
                    "archived_at": archived_at.isoformat() if archived_at else None,
                    "purged_at": current_time.isoformat(),
                    "archive_retention_days": archive_retention_days,
                },
            )

    return result
