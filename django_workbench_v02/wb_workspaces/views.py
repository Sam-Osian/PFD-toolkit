from django.contrib.auth import get_user_model
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .forms import WorkspaceCreateForm, WorkspaceMemberAddForm, WorkspaceMemberUpdateForm
from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership, WorkspaceVisibility
from .permissions import can_manage_members, can_view_workspace
from .services import (
    WorkspaceMembershipError,
    add_workspace_member,
    create_workspace_for_user,
    remove_workspace_member,
    update_workspace_member,
)


User = get_user_model()


@login_required
@require_http_methods(["GET", "POST"])
def dashboard(request):
    if request.method == "POST":
        form = WorkspaceCreateForm(request.POST)
        if form.is_valid():
            workspace = create_workspace_for_user(
                user=request.user,
                title=form.cleaned_data["title"],
                slug=form.cleaned_data["slug"],
                description=form.cleaned_data["description"],
                request=request,
            )
            workspace.visibility = form.cleaned_data["visibility"]
            workspace.is_listed = form.cleaned_data["is_listed"]
            workspace.save(update_fields=["visibility", "is_listed", "updated_at"])
            messages.success(request, f"Workspace '{workspace.title}' created.")
            return redirect("workspace-detail", workspace_id=workspace.id)
    else:
        form = WorkspaceCreateForm()

    memberships = (
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=request.user)
        .order_by("-workspace__updated_at")
    )

    return render(
        request,
        "wb_workspaces/dashboard.html",
        {
            "memberships": memberships,
            "form": form,
        },
    )


@require_GET
def workspace_detail(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")

    workspace.last_viewed_at = timezone.now()
    workspace.save(update_fields=["last_viewed_at"])

    membership = None
    if request.user.is_authenticated:
        membership = WorkspaceMembership.objects.filter(
            workspace=workspace, user=request.user
        ).first()

    manage_members_allowed = bool(
        request.user.is_authenticated and can_manage_members(request.user, workspace)
    )
    memberships = WorkspaceMembership.objects.filter(workspace=workspace).select_related("user")
    add_form = WorkspaceMemberAddForm(initial={"can_run_workflows": True})

    return render(
        request,
        "wb_workspaces/workspace_detail.html",
        {
            "workspace": workspace,
            "membership": membership,
            "memberships": memberships,
            "manage_members_allowed": manage_members_allowed,
            "add_form": add_form,
            "role_choices": MembershipRole.choices,
            "access_mode_choices": MembershipAccessMode.choices,
        },
    )


@require_GET
def public_workspace_list(request):
    workspaces = Workspace.objects.filter(
        visibility=WorkspaceVisibility.PUBLIC, is_listed=True, archived_at__isnull=True
    ).order_by("-updated_at")
    return render(
        request,
        "wb_workspaces/public_workspace_list.html",
        {"workspaces": workspaces},
    )


@login_required
@require_POST
def add_member(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    form = WorkspaceMemberAddForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member form submission.")
        return redirect("workspace-detail", workspace_id=workspace_id)

    target_email = form.cleaned_data["email"].strip().lower()
    target_user = User.objects.filter(email__iexact=target_email).first()
    if target_user is None:
        messages.error(request, f"No account exists for {target_email}.")
        return redirect("workspace-detail", workspace_id=workspace_id)

    try:
        add_workspace_member(
            actor=request.user,
            workspace=workspace,
            target_user=target_user,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{target_user.email} added to workspace.")
    return redirect("workspace-detail", workspace_id=workspace_id)


@login_required
@require_POST
def update_member(request, workspace_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    form = WorkspaceMemberUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member update form submission.")
        return redirect("workspace-detail", workspace_id=workspace_id)

    try:
        update_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{membership.user.email} membership updated.")
    return redirect("workspace-detail", workspace_id=workspace_id)


@login_required
@require_POST
def remove_member(request, workspace_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    try:
        remove_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workspace membership removed.")
    return redirect("workspace-detail", workspace_id=workspace_id)
