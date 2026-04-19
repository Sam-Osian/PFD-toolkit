from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_audit_event

from .credentials import WorkspaceCredentialError, decrypt_secret, encrypt_secret
from .models import (
    MembershipAccessMode,
    MembershipRole,
    Workspace,
    WorkspaceCredential,
    WorkspaceLLMProvider,
    WorkspaceMembership,
)
from .permissions import can_manage_members


class WorkspaceMembershipError(ValidationError):
    pass


class WorkspaceCredentialValidationError(ValidationError):
    pass


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
    log_audit_event(
        action_type="workspace.created",
        target_type="workspace",
        target_id=str(workspace.id),
        workspace=workspace,
        user=user,
        payload={"title": workspace.title, "slug": workspace.slug},
        request=request,
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
    if not api_key or not str(api_key).strip():
        raise WorkspaceCredentialValidationError("API key is required.")
    resolved_provider = _normalise_provider(provider)
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
    if credential is None:
        raise WorkspaceCredentialValidationError(
            f"No saved {resolved_provider} API key for this workspace."
        )

    try:
        api_key = decrypt_secret(credential.encrypted_api_key)
    except WorkspaceCredentialError as exc:
        raise WorkspaceCredentialValidationError(str(exc)) from exc

    credential.last_used_at = timezone.now()
    credential.save(update_fields=["last_used_at", "updated_at"])
    log_audit_event(
        action_type="workspace.credential_used",
        target_type="workspace_credential",
        target_id=str(credential.id),
        workspace=workspace,
        user=user if user and getattr(user, "is_authenticated", False) else None,
        payload={"provider": credential.provider, "key_last4": credential.key_last4},
        request=request,
    )
    return api_key, credential.base_url


def has_workspace_credential(*, user, workspace: Workspace, provider: str) -> bool:
    resolved_provider = _normalise_provider(provider)
    return WorkspaceCredential.objects.filter(
        workspace=workspace,
        user=user,
        provider=resolved_provider,
    ).exists()


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
    return True
