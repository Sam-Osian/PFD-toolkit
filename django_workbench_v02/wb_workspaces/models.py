import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class WorkspaceVisibility(models.TextChoices):
    PRIVATE = "private", "Private"
    PUBLIC = "public", "Public"


class MembershipRole(models.TextChoices):
    OWNER = "owner", "Owner"
    EDITOR = "editor", "Editor"
    VIEWER = "viewer", "Viewer"


class MembershipAccessMode(models.TextChoices):
    EDIT = "edit", "Edit"
    READ_ONLY = "read_only", "Read-only"


class RevisionChangeType(models.TextChoices):
    EDIT = "edit", "Edit"
    UNDO = "undo", "Undo"
    REDO = "redo", "Redo"
    RESTORE = "restore", "Restore"
    SYSTEM = "system", "System"


class WorkspaceLLMProvider(models.TextChoices):
    OPENAI = "openai", "OpenAI"
    OPENROUTER = "openrouter", "OpenRouter"


class Workspace(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_workspaces",
    )
    title = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100)
    description = models.TextField(blank=True)
    visibility = models.CharField(
        max_length=16,
        choices=WorkspaceVisibility.choices,
        default=WorkspaceVisibility.PRIVATE,
    )
    is_listed = models.BooleanField(default=False)
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    archived_at = models.DateTimeField(null=True, blank=True)
    current_revision = models.ForeignKey(
        "wb_workspaces.WorkspaceRevision",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cursor_workspaces",
    )
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["created_by", "slug"],
                name="uniq_workspace_created_by_slug",
            ),
        ]
        indexes = [
            models.Index(fields=["visibility", "is_listed"], name="idx_ws_vis_listed"),
            models.Index(fields=["archived_at", "-updated_at"], name="idx_ws_arch_upd_desc"),
            models.Index(fields=["updated_at"], name="idx_ws_updated_at"),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title


class WorkspaceUserState(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workspace_state",
    )
    active_workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="active_for_users",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["active_workspace"], name="idx_wus_active_workspace"),
        ]

    def __str__(self) -> str:
        return f"{self.user_id}:{self.active_workspace_id}"


class WorkspaceMembership(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workspace_memberships",
    )
    role = models.CharField(
        max_length=12,
        choices=MembershipRole.choices,
        default=MembershipRole.VIEWER,
    )
    access_mode = models.CharField(
        max_length=16,
        choices=MembershipAccessMode.choices,
        default=MembershipAccessMode.READ_ONLY,
    )
    can_manage_members = models.BooleanField(default=False)
    can_manage_shares = models.BooleanField(default=False)
    can_run_workflows = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "user"],
                name="uniq_workspace_membership_user",
            ),
        ]
        indexes = [
            models.Index(fields=["workspace", "role"], name="idx_wsm_workspace_role"),
            models.Index(fields=["user"], name="idx_wsm_user"),
        ]

    def __str__(self) -> str:
        return f"{self.user} in {self.workspace} ({self.role}, {self.access_mode})"


class WorkspaceRevision(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="revisions",
    )
    revision_number = models.PositiveIntegerField()
    state_json = models.JSONField(default=dict)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="workspace_revisions",
    )
    change_type = models.CharField(
        max_length=12,
        choices=RevisionChangeType.choices,
        default=RevisionChangeType.EDIT,
    )
    parent_revision = models.ForeignKey(
        "wb_workspaces.WorkspaceRevision",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="child_revisions",
    )
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "revision_number"],
                name="uniq_workspace_revision_number",
            ),
            models.CheckConstraint(
                check=models.Q(revision_number__gt=0),
                name="chk_workspace_revision_number_gt_zero",
            ),
        ]
        indexes = [
            models.Index(
                fields=["workspace", "-revision_number"],
                name="idx_wsr_workspace_revdesc",
            ),
            models.Index(fields=["created_at"], name="idx_wsr_created_at"),
        ]

    def __str__(self) -> str:
        return f"{self.workspace} r{self.revision_number}"


class WorkspaceCredential(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="credentials",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workspace_credentials",
    )
    provider = models.CharField(max_length=16, choices=WorkspaceLLMProvider.choices)
    encrypted_api_key = models.TextField()
    base_url = models.URLField(blank=True)
    key_last4 = models.CharField(max_length=4)
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "user", "provider"],
                name="uniq_workspace_credential_scope",
            ),
        ]
        indexes = [
            models.Index(fields=["workspace", "user"], name="idx_ws_cred_scope"),
            models.Index(fields=["provider"], name="idx_ws_cred_provider"),
        ]

    def __str__(self) -> str:
        return f"{self.workspace} {self.user} {self.provider}"


class UserLLMCredential(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="llm_credentials",
    )
    provider = models.CharField(max_length=16, choices=WorkspaceLLMProvider.choices)
    encrypted_api_key = models.TextField()
    base_url = models.URLField(blank=True)
    key_last4 = models.CharField(max_length=4)
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "provider"],
                name="uniq_user_llm_credential_scope",
            ),
        ]
        indexes = [
            models.Index(fields=["user"], name="idx_user_llm_cred_user"),
            models.Index(fields=["provider"], name="idx_user_llm_cred_provider"),
        ]

    def __str__(self) -> str:
        return f"{self.user} {self.provider}"


class UserLLMSetting(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="llm_setting",
    )
    provider = models.CharField(
        max_length=16,
        choices=WorkspaceLLMProvider.choices,
        default=WorkspaceLLMProvider.OPENAI,
    )
    model_name = models.CharField(max_length=255, default="gpt-4.1-mini")
    max_parallel_workers = models.PositiveSmallIntegerField(default=1)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=models.Q(max_parallel_workers__gte=1)
                & models.Q(max_parallel_workers__lte=32),
                name="chk_user_llm_setting_workers_bounds",
            ),
        ]
        indexes = [
            models.Index(fields=["provider"], name="idx_user_llm_provider"),
        ]

    def __str__(self) -> str:
        return f"{self.user} {self.provider} {self.model_name}"


class WorkspaceLLMSetting(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="llm_settings",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workspace_llm_settings",
    )
    provider = models.CharField(
        max_length=16,
        choices=WorkspaceLLMProvider.choices,
        default=WorkspaceLLMProvider.OPENAI,
    )
    model_name = models.CharField(max_length=255, default="gpt-4.1-mini")
    max_parallel_workers = models.PositiveSmallIntegerField(default=1)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "user"],
                name="uniq_workspace_llm_setting_scope",
            ),
            models.CheckConstraint(
                check=models.Q(max_parallel_workers__gte=1)
                & models.Q(max_parallel_workers__lte=32),
                name="chk_workspace_llm_setting_workers_bounds",
            ),
        ]
        indexes = [
            models.Index(fields=["workspace", "user"], name="idx_ws_llm_scope"),
            models.Index(fields=["provider"], name="idx_ws_llm_provider"),
        ]

    def __str__(self) -> str:
        return f"{self.workspace} {self.user} {self.provider} {self.model_name}"


class WorkspaceReportExclusion(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        "wb_workspaces.Workspace",
        on_delete=models.CASCADE,
        related_name="report_exclusions",
    )
    report_identity = models.CharField(max_length=512)
    reason = models.TextField(blank=True)
    report_title = models.TextField(blank=True)
    report_date = models.CharField(max_length=32, blank=True)
    report_url = models.URLField(max_length=1024, blank=True)
    excluded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="workspace_report_exclusions",
    )
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "report_identity"],
                name="uniq_workspace_report_exclusion",
            ),
        ]
        indexes = [
            models.Index(fields=["workspace", "created_at"], name="idx_wre_workspace_created"),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.workspace_id}:{self.report_identity}"
