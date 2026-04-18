from django.contrib import admin

from .models import Investigation


@admin.register(Investigation)
class InvestigationAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "workspace",
        "created_by",
        "status",
        "last_viewed_at",
        "updated_at",
    ]
    list_filter = ["status", "created_at", "updated_at"]
    search_fields = [
        "title",
        "workspace__title",
        "created_by__email",
        "created_by__first_name",
        "created_by__last_name",
    ]
    autocomplete_fields = ["workspace", "created_by"]
    readonly_fields = ["created_at", "updated_at", "last_viewed_at"]
