import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class NotificationChannel(models.TextChoices):
    EMAIL = "email", "Email"


class NotificationTrigger(models.TextChoices):
    SUCCESS = "success", "Success"
    FAILURE = "failure", "Failure"
    ANY = "any", "Any"


class NotificationStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    SENT = "sent", "Sent"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


class NotificationRequest(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        "wb_runs.InvestigationRun",
        on_delete=models.CASCADE,
        related_name="notification_requests",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notification_requests",
    )
    channel = models.CharField(
        max_length=16,
        choices=NotificationChannel.choices,
        default=NotificationChannel.EMAIL,
    )
    notify_on = models.CharField(
        max_length=16,
        choices=NotificationTrigger.choices,
        default=NotificationTrigger.ANY,
    )
    status = models.CharField(
        max_length=16,
        choices=NotificationStatus.choices,
        default=NotificationStatus.PENDING,
    )
    sent_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["status", "created_at"], name="idx_notif_status_created"),
            models.Index(fields=["user", "status"], name="idx_notif_user_status"),
            models.Index(fields=["run"], name="idx_notif_run"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.run_id} -> {self.user} ({self.status})"
