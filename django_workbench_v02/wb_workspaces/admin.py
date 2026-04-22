from django.contrib import admin

from .models import (
    UserLLMCredential,
    UserLLMSetting,
    Workspace,
    WorkspaceCredential,
    WorkspaceLLMSetting,
    WorkspaceMembership,
    WorkspaceRevision,
    WorkspaceUserState,
)


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


@admin.register(WorkspaceCredential)
class WorkspaceCredentialAdmin(admin.ModelAdmin):
    list_display = [
        "workspace",
        "user",
        "provider",
        "key_last4",
        "last_used_at",
        "updated_at",
    ]
    list_filter = ["provider", "updated_at"]
    search_fields = [
        "workspace__title",
        "user__email",
        "user__first_name",
        "user__last_name",
        "key_last4",
    ]
    autocomplete_fields = ["workspace", "user"]
    readonly_fields = ["key_last4", "last_used_at", "created_at", "updated_at"]


@admin.register(WorkspaceUserState)
class WorkspaceUserStateAdmin(admin.ModelAdmin):
    list_display = ["user", "active_workspace", "updated_at"]
    search_fields = [
        "user__email",
        "user__first_name",
        "user__last_name",
        "active_workspace__title",
    ]
    autocomplete_fields = ["user", "active_workspace"]
    readonly_fields = ["updated_at"]


@admin.register(WorkspaceLLMSetting)
class WorkspaceLLMSettingAdmin(admin.ModelAdmin):
    list_display = [
        "workspace",
        "user",
        "provider",
        "model_name",
        "max_parallel_workers",
        "updated_at",
    ]
    list_filter = ["provider", "updated_at"]
    search_fields = [
        "workspace__title",
        "user__email",
        "user__first_name",
        "user__last_name",
        "model_name",
    ]
    autocomplete_fields = ["workspace", "user"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(UserLLMCredential)
class UserLLMCredentialAdmin(admin.ModelAdmin):
    list_display = ["user", "provider", "key_last4", "last_used_at", "updated_at"]
    list_filter = ["provider", "updated_at"]
    search_fields = ["user__email", "user__first_name", "user__last_name", "key_last4"]
    autocomplete_fields = ["user"]
    readonly_fields = ["key_last4", "last_used_at", "created_at", "updated_at"]


@admin.register(UserLLMSetting)
class UserLLMSettingAdmin(admin.ModelAdmin):
    list_display = ["user", "provider", "model_name", "max_parallel_workers", "updated_at"]
    list_filter = ["provider", "updated_at"]
    search_fields = ["user__email", "user__first_name", "user__last_name", "model_name"]
    autocomplete_fields = ["user"]
    readonly_fields = ["created_at", "updated_at"]
