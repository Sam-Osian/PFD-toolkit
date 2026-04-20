import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class AuditEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_events",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_events",
    )
    action_type = models.CharField(max_length=100)
    target_type = models.CharField(max_length=100)
    target_id = models.CharField(max_length=100)
    ip_hash = models.CharField(max_length=128, null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    payload_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        indexes = [
            models.Index(fields=["workspace", "-created_at"], name="idx_audit_workspace_created"),
            models.Index(fields=["user", "-created_at"], name="idx_audit_user_created"),
            models.Index(fields=["action_type", "-created_at"], name="idx_audit_action_created"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.action_type} {self.target_type}:{self.target_id}"


class ActionCacheEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="action_cache_events",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="action_cache_events",
    )
    action_key = models.CharField(max_length=100)
    entity_type = models.CharField(max_length=100)
    entity_id = models.CharField(max_length=100)
    query_json = models.JSONField(default=dict)
    options_json = models.JSONField(default=dict)
    state_before_json = models.JSONField(default=dict)
    state_after_json = models.JSONField(default=dict)
    context_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        indexes = [
            models.Index(fields=["workspace", "-created_at"], name="idx_ac_workspace_created"),
            models.Index(fields=["action_key", "-created_at"], name="idx_ac_action_created"),
            models.Index(fields=["entity_type", "entity_id"], name="idx_ac_entity"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.action_key} {self.entity_type}:{self.entity_id}"
