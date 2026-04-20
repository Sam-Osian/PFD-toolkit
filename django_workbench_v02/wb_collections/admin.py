from django.contrib import admin

from .models import CollectionCardSnapshot


@admin.register(CollectionCardSnapshot)
class CollectionCardSnapshotAdmin(admin.ModelAdmin):
    list_display = ("key", "generated_at", "source_row_count", "updated_at")
    search_fields = ("key",)
    readonly_fields = ("created_at", "updated_at")
