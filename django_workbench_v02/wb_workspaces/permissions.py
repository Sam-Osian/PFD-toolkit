from __future__ import annotations

from functools import wraps
from typing import Callable

from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership, WorkspaceVisibility


def _membership_for(user, workspace: Workspace) -> WorkspaceMembership | None:
    if not user or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return None
    if user.is_superuser:
        return None
    try:
        return WorkspaceMembership.objects.get(workspace=workspace, user=user)
    except WorkspaceMembership.DoesNotExist:
        return None


def can_view_workspace(user, workspace: Workspace) -> bool:
    if workspace.visibility == WorkspaceVisibility.PUBLIC:
        return True
    if user and getattr(user, "is_superuser", False):
        return True
    membership = _membership_for(user, workspace)
    return membership is not None


def can_edit_workspace(user, workspace: Workspace) -> bool:
    if user and getattr(user, "is_superuser", False):
        return True
    membership = _membership_for(user, workspace)
    if membership is None:
        return False
    if membership.access_mode != MembershipAccessMode.EDIT:
        return False
    return membership.role in {MembershipRole.OWNER, MembershipRole.EDITOR}


def can_manage_members(user, workspace: Workspace) -> bool:
    if user and getattr(user, "is_superuser", False):
        return True
    membership = _membership_for(user, workspace)
    if membership is None:
        return False
    if membership.access_mode != MembershipAccessMode.EDIT:
        return False
    return membership.role == MembershipRole.OWNER and membership.can_manage_members


def can_manage_shares(user, workspace: Workspace) -> bool:
    if user and getattr(user, "is_superuser", False):
        return True
    membership = _membership_for(user, workspace)
    if membership is None:
        return False
    if membership.access_mode != MembershipAccessMode.EDIT:
        return False
    return membership.role == MembershipRole.OWNER and membership.can_manage_shares


def can_run_workflows(user, workspace: Workspace) -> bool:
    if user and getattr(user, "is_superuser", False):
        return True
    membership = _membership_for(user, workspace)
    if membership is None:
        return False
    return membership.can_run_workflows


def workspace_permission_required(
    permission_check: Callable,
    workspace_kwarg: str = "workspace_id",
) -> Callable:
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(request: HttpRequest, *args, **kwargs) -> HttpResponse:
            workspace_id = kwargs.get(workspace_kwarg)
            workspace = get_object_or_404(Workspace, id=workspace_id)
            if not permission_check(request.user, workspace):
                raise PermissionDenied("You do not have access to this workspace.")
            request.workspace = workspace
            return view_func(request, *args, **kwargs)

        return wrapped

    return decorator
