from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction

from wb_auditlog.services import log_audit_event

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership
from .permissions import can_manage_members


class WorkspaceMembershipError(ValidationError):
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
