from django.contrib import admin

from .models import AuditEvent


@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    list_display = [
        "created_at",
        "action_type",
        "target_type",
        "target_id",
        "workspace",
        "user",
    ]
    list_filter = ["action_type", "target_type", "created_at"]
    search_fields = [
        "action_type",
        "target_type",
        "target_id",
        "workspace__title",
        "user__email",
        "user__first_name",
        "user__last_name",
    ]
    autocomplete_fields = ["workspace", "user"]
    readonly_fields = ["created_at"]
