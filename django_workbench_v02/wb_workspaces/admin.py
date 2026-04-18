from django.contrib import admin

from .models import Workspace, WorkspaceMembership, WorkspaceRevision


class WorkspaceMembershipInline(admin.TabularInline):
    model = WorkspaceMembership
    extra = 0
    autocomplete_fields = ["user"]
    fields = [
        "user",
        "role",
        "access_mode",
        "can_manage_members",
        "can_manage_shares",
        "can_run_workflows",
        "created_at",
        "updated_at",
    ]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "slug",
        "created_by",
        "visibility",
        "is_listed",
        "last_viewed_at",
        "updated_at",
    ]
    list_filter = ["visibility", "is_listed", "archived_at"]
    search_fields = [
        "title",
        "slug",
        "created_by__email",
        "created_by__first_name",
        "created_by__last_name",
    ]
    autocomplete_fields = ["created_by"]
    readonly_fields = ["created_at", "updated_at", "last_viewed_at", "archived_at"]
    inlines = [WorkspaceMembershipInline]


@admin.register(WorkspaceMembership)
class WorkspaceMembershipAdmin(admin.ModelAdmin):
    list_display = [
        "workspace",
        "user",
        "role",
        "access_mode",
        "can_manage_members",
        "can_manage_shares",
        "can_run_workflows",
    ]
    list_filter = ["role", "access_mode", "can_manage_members", "can_manage_shares"]
    search_fields = [
        "workspace__title",
        "user__email",
        "user__first_name",
        "user__last_name",
    ]
    autocomplete_fields = ["workspace", "user"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(WorkspaceRevision)
class WorkspaceRevisionAdmin(admin.ModelAdmin):
    list_display = [
        "workspace",
        "revision_number",
        "change_type",
        "created_by",
        "created_at",
    ]
    list_filter = ["change_type", "created_at"]
    search_fields = [
        "workspace__title",
        "created_by__email",
        "created_by__first_name",
        "created_by__last_name",
    ]
    autocomplete_fields = ["workspace", "created_by", "parent_revision"]
    readonly_fields = ["created_at"]
