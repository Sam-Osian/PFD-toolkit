from __future__ import annotations

from django.db import transaction

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership


@transaction.atomic
def create_workspace_for_user(*, user, title: str, slug: str, description: str = "") -> Workspace:
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
    return workspace
