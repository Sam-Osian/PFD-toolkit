from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods

from .forms import WorkspaceCreateForm
from .models import Workspace, WorkspaceMembership, WorkspaceVisibility
from .permissions import can_view_workspace
from .services import create_workspace_for_user


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

    return render(
        request,
        "wb_workspaces/workspace_detail.html",
        {"workspace": workspace, "membership": membership},
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
