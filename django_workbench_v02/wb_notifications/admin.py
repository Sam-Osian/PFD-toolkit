from django.contrib import admin

from .models import NotificationRequest


@admin.register(NotificationRequest)
class NotificationRequestAdmin(admin.ModelAdmin):
    list_display = [
        "run",
        "user",
        "channel",
        "notify_on",
        "status",
        "sent_at",
        "created_at",
    ]
    list_filter = ["channel", "notify_on", "status", "created_at"]
    search_fields = [
        "run__id",
        "user__email",
        "user__first_name",
        "user__last_name",
        "error_message",
    ]
    autocomplete_fields = ["run", "user"]
    readonly_fields = ["created_at", "updated_at", "sent_at"]
