import uuid

from django.conf import settings
from django.db import models
from django.db.models import F, Q
from django.utils import timezone


class RunType(models.TextChoices):
    FILTER = "filter", "Filter"
    THEMES = "themes", "Themes"
    EXTRACT = "extract", "Extract"
    EXPORT = "export", "Export"


class RunStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    STARTING = "starting", "Starting"
    RUNNING = "running", "Running"
    CANCELLING = "cancelling", "Cancelling"
    CANCELLED = "cancelled", "Cancelled"
    SUCCEEDED = "succeeded", "Succeeded"
    FAILED = "failed", "Failed"
    TIMED_OUT = "timed_out", "Timed out"


class RunEventType(models.TextChoices):
    STAGE = "stage", "Stage"
    PROGRESS = "progress", "Progress"
    WARNING = "warning", "Warning"
    ERROR = "error", "Error"
    INFO = "info", "Info"
    CANCEL_CHECK = "cancel_check", "Cancel check"


class ArtifactType(models.TextChoices):
    FILTERED_DATASET = "filtered_dataset", "Filtered dataset"
    THEME_SUMMARY = "theme_summary", "Theme summary"
    THEME_ASSIGNMENTS = "theme_assignments", "Theme assignments"
    EXTRACTION_TABLE = "extraction_table", "Extraction table"
    BUNDLE_EXPORT = "bundle_export", "Bundle export"
    PREVIEW = "preview", "Preview"


class ArtifactStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    BUILDING = "building", "Building"
    READY = "ready", "Ready"
    FAILED = "failed", "Failed"
    EXPIRED = "expired", "Expired"


class ArtifactStorageBackend(models.TextChoices):
    DB = "db", "DB"
    OBJECT_STORAGE = "object_storage", "Object storage"
    FILE = "file", "File"


class InvestigationRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    investigation = models.ForeignKey(
        "wb_investigations.Investigation",
        on_delete=models.CASCADE,
        related_name="runs",
    )
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="runs",
    )
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="requested_runs",
    )
    run_type = models.CharField(max_length=16, choices=RunType.choices)
    status = models.CharField(
        max_length=16,
        choices=RunStatus.choices,
        default=RunStatus.QUEUED,
    )
    progress_percent = models.PositiveSmallIntegerField(null=True, blank=True)
    queued_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    cancel_requested_at = models.DateTimeField(null=True, blank=True)
    cancel_requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cancel_requested_runs",
    )
    cancel_reason = models.TextField(blank=True)
    worker_id = models.CharField(max_length=255, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.TextField(blank=True)
    input_config_json = models.JSONField(default=dict)
    query_start_date = models.DateField(null=True, blank=True)
    query_end_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=Q(progress_percent__isnull=True)
                | (Q(progress_percent__gte=0) & Q(progress_percent__lte=100)),
                name="chk_run_progress_percent_range",
            ),
            models.CheckConstraint(
                check=Q(query_start_date__isnull=True)
                | Q(query_end_date__isnull=True)
                | Q(query_end_date__gte=F("query_start_date")),
                name="chk_run_query_date_order",
            ),
        ]
        indexes = [
            models.Index(
                fields=["workspace", "status", "queued_at"],
                name="idx_run_ws_stat_qd",
            ),
            models.Index(
                fields=["status", "queued_at", "created_at"],
                name="idx_run_stat_queue_cr",
            ),
            models.Index(
                fields=["investigation", "-created_at"],
                name="idx_run_inv_cr_desc",
            ),
            models.Index(fields=["status", "updated_at"], name="idx_run_status_updated"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.investigation} [{self.run_type}] - {self.status}"


class RunEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        "wb_runs.InvestigationRun",
        on_delete=models.CASCADE,
        related_name="events",
    )
    event_type = models.CharField(max_length=16, choices=RunEventType.choices)
    message = models.TextField()
    payload_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        indexes = [
            models.Index(fields=["run", "created_at"], name="idx_run_event_run_created"),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} {self.event_type}"


class RunArtifact(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        "wb_runs.InvestigationRun",
        on_delete=models.CASCADE,
        related_name="artifacts",
    )
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="artifacts",
    )
    artifact_type = models.CharField(max_length=32, choices=ArtifactType.choices)
    status = models.CharField(max_length=16, choices=ArtifactStatus.choices)
    storage_backend = models.CharField(max_length=24, choices=ArtifactStorageBackend.choices)
    storage_uri = models.CharField(max_length=1024, null=True, blank=True)
    content_hash = models.CharField(max_length=128, null=True, blank=True)
    size_bytes = models.BigIntegerField(null=True, blank=True)
    metadata_json = models.JSONField(default=dict)
    expires_at = models.DateTimeField(null=True, blank=True)
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=Q(size_bytes__isnull=True) | Q(size_bytes__gte=0),
                name="chk_artifact_size_non_negative",
            ),
        ]
        indexes = [
            models.Index(
                fields=["workspace", "status", "expires_at"],
                name="idx_art_ws_stat_exp",
            ),
            models.Index(
                fields=["status", "expires_at"],
                name="idx_art_stat_exp",
            ),
            models.Index(fields=["run", "artifact_type"], name="idx_artifact_run_type"),
            models.Index(fields=["last_viewed_at"], name="idx_artifact_last_viewed"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.run_id} {self.artifact_type} ({self.status})"


class RunWorkerHeartbeat(models.Model):
    worker_id = models.CharField(max_length=255, unique=True)
    state = models.CharField(max_length=32, blank=True)
    last_run = models.ForeignKey(
        "wb_runs.InvestigationRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    last_run_status = models.CharField(max_length=16, blank=True)
    last_error = models.TextField(blank=True)
    last_seen_at = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["last_seen_at"], name="idx_worker_hb_last_seen"),
        ]
        ordering = ["-last_seen_at"]

    def __str__(self) -> str:
        return f"{self.worker_id} ({self.state or 'unknown'})"
