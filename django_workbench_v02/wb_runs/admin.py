from django.contrib import admin
from django.contrib import messages
from django.utils import timezone

from .models import InvestigationRun, RunArtifact, RunEvent
from .services import approve_run_for_execution, reject_run_for_execution


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
    exclude = ["ops_override_encrypted_api_key"]
    list_display = [
        "id",
        "investigation",
        "workspace",
        "run_type",
        "status",
        "requires_approval",
        "approval_status",
        "approved_by",
        "approved_at",
        "progress_percent",
        "requested_by",
        "queued_at",
        "updated_at",
    ]
    list_filter = ["run_type", "status", "requires_approval", "approval_status", "queued_at", "updated_at"]
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
        "approved_by",
        "rejected_by",
    ]
    readonly_fields = [
        "created_at",
        "updated_at",
        "queued_at",
        "started_at",
        "finished_at",
        "approval_requested_at",
        "approved_at",
        "rejected_at",
        "ops_override_provider",
        "ops_override_key_last4",
        "ops_override_base_url",
    ]
    actions = ["approve_selected_runs_now", "reject_selected_runs"]
    inlines = [RunEventInline, RunArtifactInline]

    @admin.action(description="Approve selected runs now (superuser only)")
    def approve_selected_runs_now(self, request, queryset):
        if not request.user.is_superuser:
            self.message_user(
                request,
                "Only superusers can approve runs.",
                level=messages.ERROR,
            )
            return
        approved = 0
        skipped = 0
        for run in queryset.select_related("workspace", "investigation"):
            try:
                approve_run_for_execution(
                    actor=request.user,
                    run=run,
                    scheduled_for=run.queued_at or timezone.now(),
                    request=request,
                )
            except Exception:
                skipped += 1
            else:
                approved += 1
        self.message_user(
            request,
            f"Approved {approved} run(s). Skipped {skipped}.",
            level=messages.SUCCESS if approved else messages.WARNING,
        )

    @admin.action(description="Reject selected runs (superuser only)")
    def reject_selected_runs(self, request, queryset):
        if not request.user.is_superuser:
            self.message_user(
                request,
                "Only superusers can reject runs.",
                level=messages.ERROR,
            )
            return
        rejected = 0
        skipped = 0
        for run in queryset.select_related("workspace", "investigation"):
            try:
                reject_run_for_execution(
                    actor=request.user,
                    run=run,
                    reason="Rejected in admin.",
                    request=request,
                )
            except Exception:
                skipped += 1
            else:
                rejected += 1
        self.message_user(
            request,
            f"Rejected {rejected} run(s). Skipped {skipped}.",
            level=messages.SUCCESS if rejected else messages.WARNING,
        )

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.is_superuser:
            actions.pop("approve_selected_runs_now", None)
            actions.pop("reject_selected_runs", None)
        return actions


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
