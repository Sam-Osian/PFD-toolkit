from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_action_cache_event, log_audit_event

from .credentials import WorkspaceCredentialError, decrypt_secret, encrypt_secret
from .models import (
    MembershipAccessMode,
    MembershipRole,
    RevisionChangeType,
    UserLLMCredential,
    UserLLMSetting,
    Workspace,
    WorkspaceCredential,
    WorkspaceLLMSetting,
    WorkspaceLLMProvider,
    WorkspaceMembership,
    WorkspaceReportExclusion,
    WorkspaceUserState,
)
from .permissions import can_edit_workspace, can_manage_members, can_view_workspace
from .revisions import capture_workspace_state, write_workspace_revision


class WorkspaceMembershipError(ValidationError):
    pass


class WorkspaceCredentialValidationError(ValidationError):
    pass


class WorkspaceReportExclusionError(ValidationError):
    pass


class WorkspaceLifecycleError(ValidationError):
    pass


def get_active_workspace_for_user(*, user) -> Workspace | None:
    if not user or not getattr(user, "is_authenticated", False):
        return None
    state = WorkspaceUserState.objects.select_related("active_workspace").filter(user=user).first()
    if not state or state.active_workspace is None:
        return None
    if state.active_workspace.archived_at is not None:
        return None
    if not can_view_workspace(user, state.active_workspace):
        return None
    return state.active_workspace


@transaction.atomic
def set_active_workspace_for_user(*, user, workspace: Workspace, request=None) -> WorkspaceUserState:
    if not user or not getattr(user, "is_authenticated", False):
        raise WorkspaceMembershipError("Authenticated user is required to set an active workbook.")
    if workspace.archived_at is not None:
        raise WorkspaceLifecycleError("Cannot set an archived workbook as active.")
    if not can_view_workspace(user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")

    state, _ = WorkspaceUserState.objects.get_or_create(user=user)
    if str(state.active_workspace_id or "") == str(workspace.id):
        return state

    previous_workspace_id = str(state.active_workspace_id) if state.active_workspace_id else None
    state.active_workspace = workspace
    state.save(update_fields=["active_workspace", "updated_at"])
    log_audit_event(
        action_type="workspace.active_changed",
        target_type="workspace_user_state",
        target_id=str(user.id),
        workspace=workspace,
        user=user,
        payload={
            "previous_workspace_id": previous_workspace_id,
            "active_workspace_id": str(workspace.id),
        },
        request=request,
    )
    return state


def _fallback_active_workspace_for_user(*, user, exclude_workspace_id=None) -> Workspace | None:
    if not user:
        return None
    memberships = (
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=user, workspace__archived_at__isnull=True)
        .order_by("-workspace__updated_at")
    )
    for membership in memberships:
        workspace = membership.workspace
        if exclude_workspace_id and str(workspace.id) == str(exclude_workspace_id):
            continue
        return workspace
    return None


@transaction.atomic
def archive_workspace(*, actor, workspace: Workspace, request=None) -> Workspace:
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can archive this workbook.")
    if workspace.archived_at is not None:
        return workspace

    now = timezone.now()
    workspace.archived_at = now
    workspace.save(update_fields=["archived_at", "updated_at"])

    affected_states = WorkspaceUserState.objects.select_related("user").filter(active_workspace=workspace)
    for state in affected_states:
        fallback = _fallback_active_workspace_for_user(
            user=state.user,
            exclude_workspace_id=workspace.id,
        )
        state.active_workspace = fallback
        state.save(update_fields=["active_workspace", "updated_at"])

    log_audit_event(
        action_type="workspace.archived_manual",
        target_type="workspace",
        target_id=str(workspace.id),
        workspace=workspace,
        user=actor,
        payload={
            "archived_at": now.isoformat(),
            "affected_active_users": affected_states.count(),
        },
        request=request,
    )
    return workspace


@transaction.atomic
def restore_workspace(*, actor, workspace: Workspace, request=None) -> Workspace:
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can restore this workbook.")
    if workspace.archived_at is None:
        return workspace

    archived_at = workspace.archived_at
    workspace.archived_at = None
    workspace.save(update_fields=["archived_at", "updated_at"])
    set_active_workspace_for_user(user=actor, workspace=workspace, request=request)

    log_audit_event(
        action_type="workspace.restored_manual",
        target_type="workspace",
        target_id=str(workspace.id),
        workspace=workspace,
        user=actor,
        payload={
            "archived_at_before_restore": archived_at.isoformat() if archived_at else None,
            "restored_at": timezone.now().isoformat(),
        },
        request=request,
    )
    return workspace


@transaction.atomic
def delete_workspace_immediately(*, actor, workspace: Workspace, reason: str = "", request=None) -> None:
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can permanently delete this workbook.")

    workspace_id = str(workspace.id)
    workspace_title = workspace.title
    WorkspaceUserState.objects.filter(active_workspace=workspace).update(active_workspace=None)
    workspace.delete()
    log_audit_event(
        action_type="workspace.deleted_manual_admin",
        target_type="workspace",
        target_id=workspace_id,
        workspace=None,
        user=actor,
        payload={
            "workspace_title": workspace_title,
            "reason": str(reason or "").strip(),
        },
        request=request,
    )


def _ensure_admin_for_owner_assignment(*, actor, current_role: str | None, target_role: str) -> None:
    promoting_to_owner = target_role == MembershipRole.OWNER and current_role != MembershipRole.OWNER
    if promoting_to_owner and not getattr(actor, "is_superuser", False):
        raise WorkspaceMembershipError(
            "Only the site admin can grant owner role to another user."
        )


def _owner_memberships_queryset(workspace: Workspace):
    return WorkspaceMembership.objects.filter(
        workspace=workspace,
        role=MembershipRole.OWNER,
    )


def _ensure_owner_invariants(workspace: Workspace) -> None:
    owner_qs = _owner_memberships_queryset(workspace)
    if not owner_qs.exists():
        raise WorkspaceMembershipError("Workspace must keep at least one owner.")

    managing_owner_exists = owner_qs.filter(
        access_mode=MembershipAccessMode.EDIT,
        can_manage_members=True,
    ).exists()
    if not managing_owner_exists:
        raise WorkspaceMembershipError(
            "Workspace must keep at least one owner with edit access and member-management rights."
        )


@transaction.atomic
def create_workspace_for_user(
    *,
    user,
    title: str,
    slug: str,
    description: str = "",
    seed_initial_revision: bool = True,
    request=None,
) -> Workspace:
    workspace = Workspace.objects.create(
        created_by=user,
        title=title,
        slug=slug,
        description=description,
    )
    WorkspaceMembership.objects.create(
        workspace=workspace,
        user=user,
        role=MembershipRole.OWNER,
        access_mode=MembershipAccessMode.EDIT,
        can_manage_members=True,
        can_manage_shares=True,
        can_run_workflows=True,
    )
    set_active_workspace_for_user(user=user, workspace=workspace, request=request)
    log_audit_event(
        action_type="workspace.created",
        target_type="workspace",
        target_id=str(workspace.id),
        workspace=workspace,
        user=user,
        payload={"title": workspace.title, "slug": workspace.slug},
        request=request,
    )
    if seed_initial_revision:
        write_workspace_revision(
            workspace=workspace,
            actor=user,
            change_type=RevisionChangeType.SYSTEM,
            state_json=capture_workspace_state(workspace=workspace),
            request=request,
            payload={"action": "workspace_created"},
        )
    log_action_cache_event(
        workspace=workspace,
        user=user,
        action_key="workbook.create",
        entity_type="workspace",
        entity_id=str(workspace.id),
        options={"slug": workspace.slug},
        state_before={},
        state_after={
            "title": workspace.title,
            "description": workspace.description,
            "visibility": workspace.visibility,
            "is_listed": workspace.is_listed,
        },
        context={"source": "service"},
    )
    _ensure_owner_invariants(workspace)
    return workspace


@transaction.atomic
def add_workspace_member(
    *,
    actor,
    workspace: Workspace,
    target_user,
    role: str,
    access_mode: str,
    can_run_workflows: bool,
    can_manage_members_flag: bool,
    can_manage_shares_flag: bool,
    request=None,
) -> WorkspaceMembership:
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can add members.")
    _ensure_admin_for_owner_assignment(
        actor=actor,
        current_role=None,
        target_role=role,
    )

    if WorkspaceMembership.objects.filter(workspace=workspace, user=target_user).exists():
        raise WorkspaceMembershipError("User is already a member of this workspace.")

    membership = WorkspaceMembership.objects.create(
        workspace=workspace,
        user=target_user,
        role=role,
        access_mode=access_mode,
        can_manage_members=can_manage_members_flag,
        can_manage_shares=can_manage_shares_flag,
        can_run_workflows=can_run_workflows,
    )
    _ensure_owner_invariants(workspace)
    log_audit_event(
        action_type="workspace.member_added",
        target_type="workspace_membership",
        target_id=str(membership.id),
        workspace=workspace,
        user=actor,
        payload={
            "member_user_id": str(target_user.id),
            "member_email": getattr(target_user, "email", ""),
            "role": role,
            "access_mode": access_mode,
            "can_manage_members": can_manage_members_flag,
            "can_manage_shares": can_manage_shares_flag,
            "can_run_workflows": can_run_workflows,
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="membership.add",
        entity_type="workspace_membership",
        entity_id=str(membership.id),
        options={
            "role": role,
            "access_mode": access_mode,
            "can_manage_members": can_manage_members_flag,
            "can_manage_shares": can_manage_shares_flag,
            "can_run_workflows": can_run_workflows,
        },
        state_before={},
        state_after={"member_user_id": str(target_user.id)},
        context={"member_email": getattr(target_user, "email", "")},
    )
    return membership


@transaction.atomic
def update_workspace_member(
    *,
    actor,
    workspace: Workspace,
    membership: WorkspaceMembership,
    role: str,
    access_mode: str,
    can_run_workflows: bool,
    can_manage_members_flag: bool,
    can_manage_shares_flag: bool,
    request=None,
) -> WorkspaceMembership:
    if membership.workspace_id != workspace.id:
        raise WorkspaceMembershipError("Membership does not belong to this workspace.")
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can update members.")
    _ensure_admin_for_owner_assignment(
        actor=actor,
        current_role=membership.role,
        target_role=role,
    )

    previous = {
        "role": membership.role,
        "access_mode": membership.access_mode,
        "can_run_workflows": membership.can_run_workflows,
        "can_manage_members": membership.can_manage_members,
        "can_manage_shares": membership.can_manage_shares,
    }

    membership.role = role
    membership.access_mode = access_mode
    membership.can_run_workflows = can_run_workflows
    membership.can_manage_members = can_manage_members_flag
    membership.can_manage_shares = can_manage_shares_flag
    membership.save(
        update_fields=[
            "role",
            "access_mode",
            "can_run_workflows",
            "can_manage_members",
            "can_manage_shares",
            "updated_at",
        ]
    )
    _ensure_owner_invariants(workspace)

    log_audit_event(
        action_type="workspace.member_updated",
        target_type="workspace_membership",
        target_id=str(membership.id),
        workspace=workspace,
        user=actor,
        payload={
            "member_user_id": str(membership.user_id),
            "before": previous,
            "after": {
                "role": role,
                "access_mode": access_mode,
                "can_run_workflows": can_run_workflows,
                "can_manage_members": can_manage_members_flag,
                "can_manage_shares": can_manage_shares_flag,
            },
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="membership.update",
        entity_type="workspace_membership",
        entity_id=str(membership.id),
        options={
            "role": role,
            "access_mode": access_mode,
            "can_manage_members": can_manage_members_flag,
            "can_manage_shares": can_manage_shares_flag,
            "can_run_workflows": can_run_workflows,
        },
        state_before=previous,
        state_after={
            "role": membership.role,
            "access_mode": membership.access_mode,
            "can_run_workflows": membership.can_run_workflows,
            "can_manage_members": membership.can_manage_members,
            "can_manage_shares": membership.can_manage_shares,
        },
        context={"member_user_id": str(membership.user_id)},
    )
    return membership


@transaction.atomic
def remove_workspace_member(
    *,
    actor,
    workspace: Workspace,
    membership: WorkspaceMembership,
    request=None,
) -> None:
    if membership.workspace_id != workspace.id:
        raise WorkspaceMembershipError("Membership does not belong to this workspace.")
    if not can_manage_members(actor, workspace):
        raise PermissionDenied("Only owners with member-management access can remove members.")

    payload = {
        "member_user_id": str(membership.user_id),
        "member_email": getattr(membership.user, "email", ""),
        "role": membership.role,
        "access_mode": membership.access_mode,
    }
    membership_id = membership.id
    membership.delete()
    _ensure_owner_invariants(workspace)

    log_audit_event(
        action_type="workspace.member_removed",
        target_type="workspace_membership",
        target_id=str(membership_id),
        workspace=workspace,
        user=actor,
        payload=payload,
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="membership.remove",
        entity_type="workspace_membership",
        entity_id=str(membership_id),
        options={},
        state_before=payload,
        state_after={},
        context={},
    )


def _normalise_provider(provider: str) -> str:
    raw = (provider or "").strip().lower()
    if raw not in {WorkspaceLLMProvider.OPENAI, WorkspaceLLMProvider.OPENROUTER}:
        raise WorkspaceCredentialValidationError(f"Unsupported provider '{provider}'.")
    return raw


def _key_last4(api_key: str) -> str:
    compact = (api_key or "").strip()
    if len(compact) < 4:
        raise WorkspaceCredentialValidationError("API key must be at least 4 characters.")
    return compact[-4:]


def _validate_api_key_format(*, provider: str, api_key: str) -> None:
    compact = str(api_key or "").strip()
    if not compact:
        raise WorkspaceCredentialValidationError("API key is required.")
    if provider == WorkspaceLLMProvider.OPENAI and not compact.startswith("sk-"):
        raise WorkspaceCredentialValidationError(
            "OpenAI API keys must start with 'sk-'. Please paste a valid OpenAI API key."
        )
    if provider == WorkspaceLLMProvider.OPENROUTER and not compact.startswith("sk-or-"):
        raise WorkspaceCredentialValidationError(
            "OpenRouter API keys must start with 'sk-or-'. Please paste a valid OpenRouter API key."
        )


def _normalise_model_name(model_name: str) -> str:
    cleaned = str(model_name or "").strip()
    if not cleaned:
        return "gpt-4.1-mini"
    legacy_to_current = {
        "gpt-4.1": "gpt-5.4",
        "openai/gpt-4.1": "openai/gpt-5.4",
    }
    return legacy_to_current.get(cleaned, cleaned)


def _normalise_max_parallel_workers(value) -> int:
    try:
        parsed = int(value or 1)
    except (TypeError, ValueError):
        parsed = 1
    return min(32, max(1, parsed))


@transaction.atomic
def upsert_workspace_credential(
    *,
    actor,
    workspace: Workspace,
    provider: str,
    api_key: str,
    base_url: str = "",
    request=None,
) -> WorkspaceCredential:
    resolved_provider = _normalise_provider(provider)
    _validate_api_key_format(provider=resolved_provider, api_key=str(api_key))
    try:
        encrypted = encrypt_secret(str(api_key))
    except WorkspaceCredentialError as exc:
        raise WorkspaceCredentialValidationError(str(exc)) from exc

    credential, created = WorkspaceCredential.objects.get_or_create(
        workspace=workspace,
        user=actor,
        provider=resolved_provider,
        defaults={
            "encrypted_api_key": encrypted,
            "key_last4": _key_last4(str(api_key)),
            "base_url": (base_url or "").strip(),
        },
    )
    if not created:
        credential.encrypted_api_key = encrypted
        credential.key_last4 = _key_last4(str(api_key))
        credential.base_url = (base_url or "").strip()
        credential.save(
            update_fields=[
                "encrypted_api_key",
                "key_last4",
                "base_url",
                "updated_at",
            ]
        )

    log_audit_event(
        action_type="workspace.credential_saved",
        target_type="workspace_credential",
        target_id=str(credential.id),
        workspace=workspace,
        user=actor,
        payload={
            "provider": credential.provider,
            "key_last4": credential.key_last4,
            "created": created,
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="credential.save",
        entity_type="workspace_credential",
        entity_id=str(credential.id),
        options={"provider": credential.provider},
        state_before={},
        state_after={
            "provider": credential.provider,
            "key_last4": credential.key_last4,
            "base_url": credential.base_url,
        },
        context={"created": created},
    )
    return credential


def resolve_workspace_credential(
    *,
    user,
    workspace: Workspace,
    provider: str,
    request=None,
) -> tuple[str, str]:
    resolved_provider = _normalise_provider(provider)
    credential = WorkspaceCredential.objects.filter(
        workspace=workspace,
        user=user,
        provider=resolved_provider,
    ).first()
    target_type = "workspace_credential"
    if credential is None:
        credential = UserLLMCredential.objects.filter(
            user=user,
            provider=resolved_provider,
        ).first()
        target_type = "user_llm_credential"
    if credential is None:
        raise WorkspaceCredentialValidationError(f"No saved {resolved_provider} API key available.")

    try:
        api_key = decrypt_secret(credential.encrypted_api_key)
    except WorkspaceCredentialError as exc:
        raise WorkspaceCredentialValidationError(str(exc)) from exc

    credential.last_used_at = timezone.now()
    credential.save(update_fields=["last_used_at", "updated_at"])
    log_audit_event(
        action_type="workspace.credential_used",
        target_type=target_type,
        target_id=str(credential.id),
        workspace=workspace,
        user=user if user and getattr(user, "is_authenticated", False) else None,
        payload={"provider": credential.provider, "key_last4": credential.key_last4},
        request=request,
    )
    return api_key, credential.base_url


def has_workspace_credential(*, user, workspace: Workspace, provider: str) -> bool:
    resolved_provider = _normalise_provider(provider)
    workspace_saved = WorkspaceCredential.objects.filter(
        workspace=workspace,
        user=user,
        provider=resolved_provider,
    ).exists()
    if workspace_saved:
        return True
    return UserLLMCredential.objects.filter(
        user=user,
        provider=resolved_provider,
    ).exists()


def workspace_credential_status_map(*, user, workspace: Workspace) -> dict[str, bool]:
    if not user or not getattr(user, "is_authenticated", False):
        return {
            WorkspaceLLMProvider.OPENAI: False,
            WorkspaceLLMProvider.OPENROUTER: False,
        }
    providers = set(
        WorkspaceCredential.objects.filter(
            workspace=workspace,
            user=user,
        ).values_list("provider", flat=True)
    )
    user_providers = set(
        UserLLMCredential.objects.filter(
            user=user,
        ).values_list("provider", flat=True)
    )
    merged = providers | user_providers
    return {
        WorkspaceLLMProvider.OPENAI: WorkspaceLLMProvider.OPENAI in merged,
        WorkspaceLLMProvider.OPENROUTER: WorkspaceLLMProvider.OPENROUTER in merged,
    }


def get_workspace_llm_setting(*, user, workspace: Workspace) -> dict[str, object]:
    provider = WorkspaceLLMProvider.OPENAI
    model_name = "gpt-4.1-mini"
    max_parallel_workers = 1

    setting = UserLLMSetting.objects.filter(user=user).first()
    if setting is not None:
        provider = _normalise_provider(setting.provider)
        model_name = _normalise_model_name(setting.model_name)
        max_parallel_workers = _normalise_max_parallel_workers(setting.max_parallel_workers)

    cred_map = workspace_credential_status_map(user=user, workspace=workspace)
    credential_last4 = {
        WorkspaceLLMProvider.OPENAI: "",
        WorkspaceLLMProvider.OPENROUTER: "",
    }
    for provider_key in (WorkspaceLLMProvider.OPENAI, WorkspaceLLMProvider.OPENROUTER):
        provider_credential = WorkspaceCredential.objects.filter(
            workspace=workspace,
            user=user,
            provider=provider_key,
        ).first()
        if provider_credential is None:
            provider_credential = UserLLMCredential.objects.filter(
                user=user,
                provider=provider_key,
            ).first()
        if provider_credential is not None:
            credential_last4[provider_key] = str(provider_credential.key_last4 or "").strip()

    provider_credential = WorkspaceCredential.objects.filter(workspace=workspace, user=user, provider=provider).first()
    if provider_credential is None:
        provider_credential = UserLLMCredential.objects.filter(user=user, provider=provider).first()
    return {
        "provider": provider,
        "model_name": model_name,
        "max_parallel_workers": max_parallel_workers,
        "has_provider_credential": bool(cred_map.get(provider)),
        "credentials": cred_map,
        "credential_last4": credential_last4,
        "base_url": str(provider_credential.base_url or "").strip() if provider_credential else "",
    }


@transaction.atomic
def upsert_workspace_llm_setting(
    *,
    actor,
    workspace: Workspace,
    provider: str,
    model_name: str,
    max_parallel_workers,
    request=None,
) -> WorkspaceLLMSetting:
    # Backward-compatible shim: persist in user-level setting, independent of workspace.
    user_setting = upsert_user_llm_setting(
        actor=actor,
        provider=provider,
        model_name=model_name,
        max_parallel_workers=max_parallel_workers,
        request=request,
    )
    # Ensure an object exists for older references; not used for runtime defaults.
    resolved_provider = _normalise_provider(provider)
    resolved_model_name = _normalise_model_name(model_name)
    resolved_workers = _normalise_max_parallel_workers(max_parallel_workers)

    setting, created = WorkspaceLLMSetting.objects.get_or_create(
        workspace=workspace,
        user=actor,
        defaults={
            "provider": resolved_provider,
            "model_name": resolved_model_name,
            "max_parallel_workers": resolved_workers,
        },
    )
    if not created:
        setting.provider = resolved_provider
        setting.model_name = resolved_model_name
        setting.max_parallel_workers = resolved_workers
        setting.save(
            update_fields=[
                "provider",
                "model_name",
                "max_parallel_workers",
                "updated_at",
            ]
        )

    log_audit_event(
        action_type="workspace.llm_setting_saved",
        target_type="workspace_llm_setting",
        target_id=str(setting.id),
        workspace=workspace,
        user=actor,
        payload={
            "provider": setting.provider,
            "model_name": setting.model_name,
            "max_parallel_workers": setting.max_parallel_workers,
            "created": created,
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="llm_setting.save",
        entity_type="workspace_llm_setting",
        entity_id=str(setting.id),
        options={"provider": setting.provider},
        state_before={},
        state_after={
            "provider": setting.provider,
            "model_name": setting.model_name,
            "max_parallel_workers": setting.max_parallel_workers,
        },
        context={"created": created},
    )
    setting.provider = user_setting.provider
    setting.model_name = user_setting.model_name
    setting.max_parallel_workers = user_setting.max_parallel_workers
    setting.save(update_fields=["provider", "model_name", "max_parallel_workers", "updated_at"])
    return setting


@transaction.atomic
def upsert_user_llm_credential(
    *,
    actor,
    provider: str,
    api_key: str,
    base_url: str = "",
    request=None,
) -> UserLLMCredential:
    resolved_provider = _normalise_provider(provider)
    _validate_api_key_format(provider=resolved_provider, api_key=str(api_key))
    try:
        encrypted = encrypt_secret(str(api_key))
    except WorkspaceCredentialError as exc:
        raise WorkspaceCredentialValidationError(str(exc)) from exc

    credential, created = UserLLMCredential.objects.get_or_create(
        user=actor,
        provider=resolved_provider,
        defaults={
            "encrypted_api_key": encrypted,
            "key_last4": _key_last4(str(api_key)),
            "base_url": (base_url or "").strip(),
        },
    )
    if not created:
        credential.encrypted_api_key = encrypted
        credential.key_last4 = _key_last4(str(api_key))
        credential.base_url = (base_url or "").strip()
        credential.save(update_fields=["encrypted_api_key", "key_last4", "base_url", "updated_at"])
    log_audit_event(
        action_type="user.llm_credential_saved",
        target_type="user_llm_credential",
        target_id=str(credential.id),
        workspace=None,
        user=actor,
        payload={"provider": credential.provider, "key_last4": credential.key_last4, "created": created},
        request=request,
    )
    return credential


@transaction.atomic
def upsert_user_llm_setting(
    *,
    actor,
    provider: str,
    model_name: str,
    max_parallel_workers,
    request=None,
) -> UserLLMSetting:
    resolved_provider = _normalise_provider(provider)
    resolved_model_name = _normalise_model_name(model_name)
    resolved_workers = _normalise_max_parallel_workers(max_parallel_workers)
    setting, created = UserLLMSetting.objects.get_or_create(
        user=actor,
        defaults={
            "provider": resolved_provider,
            "model_name": resolved_model_name,
            "max_parallel_workers": resolved_workers,
        },
    )
    if not created:
        setting.provider = resolved_provider
        setting.model_name = resolved_model_name
        setting.max_parallel_workers = resolved_workers
        setting.save(update_fields=["provider", "model_name", "max_parallel_workers", "updated_at"])
    log_audit_event(
        action_type="user.llm_setting_saved",
        target_type="user_llm_setting",
        target_id=str(setting.id),
        workspace=None,
        user=actor,
        payload={
            "provider": setting.provider,
            "model_name": setting.model_name,
            "max_parallel_workers": setting.max_parallel_workers,
            "created": created,
        },
        request=request,
    )
    return setting


@transaction.atomic
def delete_workspace_credential(
    *,
    actor,
    workspace: Workspace,
    provider: str,
    request=None,
) -> bool:
    resolved_provider = _normalise_provider(provider)
    credential = WorkspaceCredential.objects.filter(
        workspace=workspace,
        user=actor,
        provider=resolved_provider,
    ).first()
    if credential is None:
        return False

    credential_id = str(credential.id)
    key_last4 = credential.key_last4
    credential.delete()
    log_audit_event(
        action_type="workspace.credential_deleted",
        target_type="workspace_credential",
        target_id=credential_id,
        workspace=workspace,
        user=actor,
        payload={"provider": resolved_provider, "key_last4": key_last4},
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="credential.delete",
        entity_type="workspace_credential",
        entity_id=credential_id,
        options={"provider": resolved_provider},
        state_before={"key_last4": key_last4},
        state_after={},
        context={},
    )
    return True


@transaction.atomic
def delete_user_llm_credential(
    *,
    actor,
    provider: str,
    request=None,
) -> bool:
    resolved_provider = _normalise_provider(provider)
    credential = UserLLMCredential.objects.filter(
        user=actor,
        provider=resolved_provider,
    ).first()
    if credential is None:
        return False

    credential_id = str(credential.id)
    key_last4 = credential.key_last4
    credential.delete()
    log_audit_event(
        action_type="user.llm_credential_deleted",
        target_type="user_llm_credential",
        target_id=credential_id,
        workspace=None,
        user=actor,
        payload={"provider": resolved_provider, "key_last4": key_last4},
        request=request,
    )
    return True


@transaction.atomic
def upsert_workspace_report_exclusion(
    *,
    actor,
    workspace: Workspace,
    report_identity: str,
    reason: str = "",
    report_title: str = "",
    report_date: str = "",
    report_url: str = "",
    request=None,
) -> WorkspaceReportExclusion:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to exclude reports in this workbook.")

    identity = str(report_identity or "").strip()
    if not identity:
        raise WorkspaceReportExclusionError("Report identity is required.")

    exclusion, created = WorkspaceReportExclusion.objects.get_or_create(
        workspace=workspace,
        report_identity=identity,
        defaults={
            "reason": str(reason or "").strip(),
            "report_title": str(report_title or "").strip(),
            "report_date": str(report_date or "").strip(),
            "report_url": str(report_url or "").strip(),
            "excluded_by": actor,
        },
    )
    if not created:
        exclusion.reason = str(reason or exclusion.reason or "").strip()
        if report_title:
            exclusion.report_title = str(report_title).strip()
        if report_date:
            exclusion.report_date = str(report_date).strip()
        if report_url:
            exclusion.report_url = str(report_url).strip()
        exclusion.excluded_by = actor
        exclusion.save(
            update_fields=[
                "reason",
                "report_title",
                "report_date",
                "report_url",
                "excluded_by",
                "updated_at",
            ]
        )

    log_audit_event(
        action_type="workspace.report_excluded",
        target_type="workspace_report_exclusion",
        target_id=str(exclusion.id),
        workspace=workspace,
        user=actor,
        payload={
            "report_identity": identity,
            "created": created,
            "reason": exclusion.reason,
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="report.exclude",
        entity_type="workspace_report_exclusion",
        entity_id=str(exclusion.id),
        options={"reason": exclusion.reason},
        state_before={},
        state_after={
            "report_identity": exclusion.report_identity,
            "report_title": exclusion.report_title,
            "report_date": exclusion.report_date,
            "report_url": exclusion.report_url,
        },
        context={"created": created},
    )
    write_workspace_revision(
        workspace=workspace,
        actor=actor,
        change_type=RevisionChangeType.EDIT,
        state_json=capture_workspace_state(workspace=workspace),
        request=request,
        payload={
            "action": "report_excluded",
            "report_identity": identity,
        },
    )
    return exclusion


@transaction.atomic
def restore_workspace_report_exclusion(
    *,
    actor,
    workspace: Workspace,
    exclusion: WorkspaceReportExclusion,
    request=None,
) -> None:
    if exclusion.workspace_id != workspace.id:
        raise WorkspaceReportExclusionError("Exclusion does not belong to this workspace.")
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to restore excluded reports in this workbook.")

    payload = {
        "report_identity": exclusion.report_identity,
        "reason": exclusion.reason,
    }
    exclusion_id = exclusion.id
    exclusion.delete()
    log_audit_event(
        action_type="workspace.report_restored",
        target_type="workspace_report_exclusion",
        target_id=str(exclusion_id),
        workspace=workspace,
        user=actor,
        payload=payload,
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="report.restore",
        entity_type="workspace_report_exclusion",
        entity_id=str(exclusion_id),
        options={},
        state_before=payload,
        state_after={},
        context={},
    )
    write_workspace_revision(
        workspace=workspace,
        actor=actor,
        change_type=RevisionChangeType.RESTORE,
        state_json=capture_workspace_state(workspace=workspace),
        request=request,
        payload={
            "action": "report_restored",
            "report_identity": payload["report_identity"],
        },
    )
