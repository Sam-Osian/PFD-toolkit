from django.contrib import messages
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_POST

from accounts.views import _coerce_positive_int, _explore_shared_querydict
from wb_investigations.models import Investigation
from wb_runs.artifact_storage import ArtifactStorageError, open_artifact_for_download
from wb_runs.models import ArtifactStatus, ArtifactType, RunArtifact
from wb_runs.services import record_artifact_download
from wb_workspaces.permissions import can_view_workspace
from wb_workspaces.models import Workspace
from wb_workspaces.views import _workspace_build_explore_payload, _workspace_reports_panel_context

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
    workspace = share_link.workspace
    payload = _workspace_build_explore_payload(workspace=workspace, request=request)
    explore_data = payload["explore"]
    dataset_available = bool(payload["dataset_available"])
    panel_query = _explore_shared_querydict(
        query=str(explore_data.get("query") or ""),
        ai_filter="",
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
        page=_coerce_positive_int(request.GET.get("page"), default=1),
    )
    panel_base = reverse("share-link-reports-panel", kwargs={"share_id": share_link.id})
    panel_url = f"{panel_base}?{panel_query.urlencode()}" if panel_query else panel_base
    investigation = Investigation.objects.filter(workspace=workspace).first()

    return render(
        request,
        "accounts/explore.html",
        {
            "explore": explore_data,
            "dataset_available": dataset_available,
            "explore_reports_panel_url": panel_url,
            "show_reports_requested": True,
            "is_collection_view": True,
            "is_workspace_view": True,
            "is_shared_read_only": True,
            "collection": {"title": workspace.title, "description": workspace.description},
            "collection_slug": f"workspace-{workspace.id}",
            "collection_title": workspace.title,
            "collection_description": str(workspace.description or "").strip(),
            "filter_action_url": reverse("share-link-detail", kwargs={"share_id": share_link.id}),
            "filter_reset_url": reverse("share-link-detail", kwargs={"share_id": share_link.id}),
            "explore_export_url": reverse("share-link-export-csv", kwargs={"share_id": share_link.id}),
            "workspace_id": str(workspace.id),
            "workspace_investigation": investigation,
            "can_share_workspace_investigation": False,
            "workspace_public_share_url": "",
            "workspace_latest_run": None,
            "workspace_theme_summary": {"columns": [], "rows": [], "error": "", "row_count": 0},
            "workspace_debug_runs": [],
        },
    )


@require_GET
def share_reports_panel(request, share_id):
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace"),
        id=share_id,
    )
    _raise_if_inactive_or_expired(share_link)
    if not share_link.is_public and not can_view_workspace(request.user, share_link.workspace):
        return redirect("accounts-login")

    context = _workspace_reports_panel_context(workspace=share_link.workspace, request=request)
    context["dataset_panel_base"] = reverse("share-link-reports-panel", kwargs={"share_id": share_link.id})
    context["dataset_browser_base"] = reverse("share-link-detail", kwargs={"share_id": share_link.id})
    context["scope_label"] = "shared workspace"
    return render(request, "accounts/_explore_reports_panel.html", context)


@require_GET
def export_share_link_csv(request, share_id):
    share_link = get_object_or_404(
        WorkspaceShareLink.objects.select_related("workspace"),
        id=share_id,
    )
    _raise_if_inactive_or_expired(share_link)
    if not share_link.is_public and not can_view_workspace(request.user, share_link.workspace):
        return redirect("accounts-login")

    artifact = (
        RunArtifact.objects.filter(
            workspace=share_link.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
        )
        .order_by("-created_at")
        .first()
    )
    if artifact is None:
        raise Http404("No export artifact is available for this shared investigation.")

    try:
        file_obj, _download_name = open_artifact_for_download(artifact)
    except ArtifactStorageError as exc:
        raise Http404(str(exc)) from exc

    if request.user.is_authenticated:
        record_artifact_download(artifact=artifact, user=request.user, request=request)
    slug = str(share_link.workspace.slug or share_link.workspace.title or "workspace").strip().replace(" ", "-").lower()
    filename = f"{slugify(slug) or 'workspace'}-pipeline-reports.csv"
    return FileResponse(file_obj, as_attachment=True, filename=filename)


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
