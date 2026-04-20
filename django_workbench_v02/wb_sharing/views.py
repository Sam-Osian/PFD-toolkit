from django.contrib import messages
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from wb_workspaces.permissions import can_view_workspace
from wb_workspaces.models import Workspace

from .forms import ShareCopyForm, ShareLinkCreateForm, ShareLinkUpdateForm
from .models import WorkspaceShareLink
from .services import (
    WorkspaceShareLinkError,
    copy_share_link_to_workbook,
    create_share_link,
    record_share_link_view,
    revoke_share_link,
    update_share_link,
)


def _raise_if_inactive_or_expired(share_link: WorkspaceShareLink) -> None:
    if not share_link.is_active:
        raise Http404("This share link is not active.")
    if share_link.expires_at and share_link.expires_at <= timezone.now():
        raise Http404("This share link has expired.")


@require_POST
def create_workspace_share(request, workbook_id):
    if not request.user.is_authenticated:
        return redirect("accounts-login")
    workspace = get_object_or_404(Workspace, id=workbook_id)
    form = ShareLinkCreateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid share link form submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    try:
        create_share_link(
            actor=request.user,
            workspace=workspace,
            mode=form.cleaned_data["mode"],
            is_public=form.cleaned_data["is_public"],
            expires_at=form.cleaned_data["expires_at"],
            request=request,
        )
    except (WorkspaceShareLinkError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Share link created.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@require_POST
def update_workspace_share(request, workbook_id, share_id):
    if not request.user.is_authenticated:
        return redirect("accounts-login")
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace"),
        id=share_id,
        workspace_id=workbook_id,
    )
    form = ShareLinkUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid share update form submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    try:
        update_share_link(
            actor=request.user,
            share_link=share_link,
            mode=form.cleaned_data["mode"],
            is_public=form.cleaned_data["is_public"],
            is_active=form.cleaned_data["is_active"],
            expires_at=form.cleaned_data["expires_at"],
            request=request,
        )
    except (WorkspaceShareLinkError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Share link updated.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@require_POST
def revoke_workspace_share(request, workbook_id, share_id):
    if not request.user.is_authenticated:
        return redirect("accounts-login")
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace"),
        id=share_id,
        workspace_id=workbook_id,
    )
    try:
        revoke_share_link(actor=request.user, share_link=share_link, request=request)
    except (WorkspaceShareLinkError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Share link revoked.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@require_GET
def view_share_link(request, share_id):
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace", "snapshot_revision"),
        id=share_id,
    )
    _raise_if_inactive_or_expired(share_link)

    if not share_link.is_public and not can_view_workspace(request.user, share_link.workspace):
        return redirect("accounts-login")

    record_share_link_view(share_link=share_link, user=request.user, request=request)
    return render(
        request,
        "wb_sharing/share_link_detail.html",
        {
            "share_link": share_link,
            "workspace": share_link.workspace,
            "copy_form": ShareCopyForm(
                initial={"workbook_title": f"{share_link.workspace.title} (Copy)"}
            ),
        },
    )


@require_POST
def copy_share_link_to_workbook_view(request, share_id):
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace", "snapshot_revision"),
        id=share_id,
    )
    _raise_if_inactive_or_expired(share_link)

    if not request.user.is_authenticated:
        messages.info(request, "Log in to make an editable workbook copy.")
        next_url = reverse("share-link-detail", kwargs={"share_id": share_id})
        return redirect(f"{reverse('accounts-login')}?next={next_url}")

    if not share_link.is_public and not can_view_workspace(request.user, share_link.workspace):
        return redirect("accounts-login")

    form = ShareCopyForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid copy request.")
        return redirect("share-link-detail", share_id=share_id)

    copied_workspace = copy_share_link_to_workbook(
        actor=request.user,
        share_link=share_link,
        target_title=form.cleaned_data["workbook_title"],
        request=request,
    )
    messages.success(request, "Editable workbook copy created.")
    return redirect("workbook-detail", workbook_id=copied_workspace.id)
