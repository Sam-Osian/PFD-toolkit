from django.contrib import admin

from .models import InvestigationRun, RunArtifact, RunEvent


class RunEventInline(admin.TabularInline):
    model = RunEvent
    extra = 0
    fields = ["event_type", "message", "created_at"]
    readonly_fields = ["event_type", "message", "created_at"]
    can_delete = False


class RunArtifactInline(admin.TabularInline):
    model = RunArtifact
    extra = 0
    fields = ["artifact_type", "status", "storage_backend", "created_at"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(InvestigationRun)
class InvestigationRunAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "investigation",
        "workspace",
        "run_type",
        "status",
        "progress_percent",
        "requested_by",
        "queued_at",
        "updated_at",
    ]
    list_filter = ["run_type", "status", "queued_at", "updated_at"]
    search_fields = [
        "id",
        "investigation__title",
        "workspace__title",
        "requested_by__email",
        "requested_by__first_name",
        "requested_by__last_name",
    ]
    autocomplete_fields = [
        "investigation",
        "workspace",
        "requested_by",
        "cancel_requested_by",
    ]
    readonly_fields = ["created_at", "updated_at", "queued_at", "started_at", "finished_at"]
    inlines = [RunEventInline, RunArtifactInline]


@admin.register(RunEvent)
class RunEventAdmin(admin.ModelAdmin):
    list_display = ["run", "event_type", "created_at"]
    list_filter = ["event_type", "created_at"]
    search_fields = ["run__id", "message"]
    autocomplete_fields = ["run"]
    readonly_fields = ["created_at"]


@admin.register(RunArtifact)
class RunArtifactAdmin(admin.ModelAdmin):
    list_display = [
        "run",
        "workspace",
        "artifact_type",
        "status",
        "storage_backend",
        "expires_at",
        "last_viewed_at",
    ]
    list_filter = ["artifact_type", "status", "storage_backend"]
    search_fields = ["run__id", "workspace__title", "storage_uri", "content_hash"]
    autocomplete_fields = ["run", "workspace"]
    readonly_fields = ["created_at", "updated_at", "last_viewed_at"]
