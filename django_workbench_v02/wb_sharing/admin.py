from django.contrib import admin

from .models import WorkspaceShareLink


@admin.register(WorkspaceShareLink)
class WorkspaceShareLinkAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "workspace",
        "created_by",
        "mode",
        "is_public",
        "is_active",
        "expires_at",
        "last_viewed_at",
    ]
    list_filter = ["mode", "is_public", "is_active", "created_at"]
    search_fields = [
        "id",
        "workspace__title",
        "created_by__email",
        "created_by__first_name",
        "created_by__last_name",
    ]
    autocomplete_fields = ["workspace", "created_by", "snapshot_revision"]
    readonly_fields = ["created_at", "updated_at", "last_viewed_at"]
