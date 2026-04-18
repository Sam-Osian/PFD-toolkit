import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class ShareMode(models.TextChoices):
    SNAPSHOT = "snapshot", "Snapshot"
    LIVE = "live", "Live"


class WorkspaceShareLink(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="share_links",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_workspace_share_links",
    )
    mode = models.CharField(
        max_length=16,
        choices=ShareMode.choices,
        default=ShareMode.SNAPSHOT,
    )
    snapshot_revision = models.ForeignKey(
        "wb_workspaces.WorkspaceRevision",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="share_links",
    )
    is_public = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["workspace", "is_active"], name="idx_share_workspace_active"),
            models.Index(fields=["is_public", "is_active"], name="idx_share_public_active"),
            models.Index(fields=["last_viewed_at"], name="idx_share_last_viewed"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.workspace} ({self.mode})"
