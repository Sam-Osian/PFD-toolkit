import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class InvestigationStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    ACTIVE = "active", "Active"
    ARCHIVED = "archived", "Archived"


class Investigation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="investigations",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_investigations",
    )
    title = models.CharField(max_length=255)
    question_text = models.TextField(blank=True)
    scope_json = models.JSONField(default=dict)
    method_json = models.JSONField(default=dict)
    status = models.CharField(
        max_length=12,
        choices=InvestigationStatus.choices,
        default=InvestigationStatus.DRAFT,
    )
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["workspace", "status"], name="idx_inv_workspace_status"),
            models.Index(fields=["updated_at"], name="idx_inv_updated_at"),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title
