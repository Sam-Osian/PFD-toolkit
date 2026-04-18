from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from wb_investigations.models import Investigation
from wb_workspaces.permissions import can_view_workspace

from .forms import RunCancelForm, RunQueueForm
from .models import ArtifactStatus, ArtifactStorageBackend, InvestigationRun, RunArtifact
from .services import (
    queue_run,
    request_run_cancellation,
    record_artifact_download,
    record_run_view,
)


@login_required
@require_POST
def queue_investigation_run(request, workspace_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workspace_id,
    )
    form = RunQueueForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid run configuration.")
        return redirect(
            "investigation-detail",
            workspace_id=workspace_id,
            investigation_id=investigation_id,
        )

    try:
        queue_run(
            actor=request.user,
            investigation=investigation,
            run_type=form.cleaned_data["run_type"],
            input_config_json=form.cleaned_data["input_config_json"],
            query_start_date=form.cleaned_data["query_start_date"],
            query_end_date=form.cleaned_data["query_end_date"],
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Run queued.")

    return redirect(
        "investigation-detail",
        workspace_id=workspace_id,
        investigation_id=investigation_id,
    )


@require_GET
def run_detail(request, workspace_id, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation", "requested_by"),
        id=run_id,
        workspace_id=workspace_id,
    )
    if not can_view_workspace(request.user, run.workspace):
        return redirect("accounts-login")

    record_run_view(run=run, user=request.user, request=request)
    cancel_form = RunCancelForm()
    return render(
        request,
        "wb_runs/run_detail.html",
        {
            "run": run,
            "investigation": run.investigation,
            "workspace": run.workspace,
            "cancel_form": cancel_form,
            "events": run.events.order_by("-created_at"),
            "artifacts": run.artifacts.order_by("-created_at"),
        },
    )


@login_required
@require_POST
def cancel_run(request, workspace_id, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation"),
        id=run_id,
        workspace_id=workspace_id,
    )
    form = RunCancelForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid cancellation payload.")
        return redirect("run-detail", workspace_id=workspace_id, run_id=run_id)

    try:
        request_run_cancellation(
            actor=request.user,
            run=run,
            reason=form.cleaned_data["cancel_reason"],
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Run cancellation requested.")
    return redirect("run-detail", workspace_id=workspace_id, run_id=run_id)


@require_GET
def download_run_artifact(request, workspace_id, run_id, artifact_id):
    artifact = get_object_or_404(
        RunArtifact.objects.select_related("workspace", "run", "run__investigation"),
        id=artifact_id,
        run_id=run_id,
        workspace_id=workspace_id,
    )
    if not can_view_workspace(request.user, artifact.workspace):
        if request.user.is_authenticated:
            raise PermissionDenied("You do not have access to this artifact.")
        return redirect("accounts-login")

    if artifact.status != ArtifactStatus.READY:
        raise Http404("Artifact is not ready.")

    if artifact.storage_backend != ArtifactStorageBackend.FILE:
        raise Http404("Artifact backend is not yet downloadable.")
    if not artifact.storage_uri:
        raise Http404("Artifact file is unavailable.")

    source_path = Path(artifact.storage_uri)
    if not source_path.is_file():
        raise Http404("Artifact file was not found.")

    record_artifact_download(artifact=artifact, user=request.user, request=request)
    download_name = source_path.name
    return FileResponse(source_path.open("rb"), as_attachment=True, filename=download_name)
