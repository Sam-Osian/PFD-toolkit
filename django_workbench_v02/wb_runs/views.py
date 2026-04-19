from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from wb_investigations.models import Investigation
from wb_notifications.services import NotificationRequestError, create_notification_request
from wb_workspaces.permissions import can_view_workspace
from wb_workspaces.services import (
    WorkspaceCredentialValidationError,
    has_workspace_credential,
    upsert_workspace_credential,
)

from .artifact_storage import ArtifactStorageError, open_artifact_for_download
from .forms import RunCancelForm, RunQueueForm
from .models import ArtifactStatus, InvestigationRun, RunArtifact
from .services import (
    queue_run,
    request_run_cancellation,
    record_artifact_download,
    record_run_view,
)


@login_required
@require_POST
def queue_investigation_run(request, workbook_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workbook_id,
    )
    form = RunQueueForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid run configuration.")
        return redirect(
            "workbook-investigation-detail",
            workbook_id=workbook_id,
            investigation_id=investigation_id,
        )

    try:
        run_config = form.cleaned_data["input_config_json"] or {}
        execution_mode = str(run_config.get("execution_mode", "real")).strip().lower()
        provider = str(form.cleaned_data.get("provider") or "openai").strip().lower()
        api_key = str(form.cleaned_data.get("api_key") or "").strip()
        save_api_key = bool(form.cleaned_data.get("save_api_key", True))
        base_url = str(form.cleaned_data.get("base_url") or "").strip()

        if execution_mode != "simulate":
            if api_key:
                if save_api_key:
                    upsert_workspace_credential(
                        actor=request.user,
                        workspace=investigation.workspace,
                        provider=provider,
                        api_key=api_key,
                        base_url=base_url,
                        request=request,
                    )
            else:
                if not has_workspace_credential(
                    user=request.user,
                    workspace=investigation.workspace,
                    provider=provider,
                ):
                    raise WorkspaceCredentialValidationError(
                        f"No saved {provider} API key for this workbook."
                    )

        run = queue_run(
            actor=request.user,
            investigation=investigation,
            run_type=form.cleaned_data["run_type"],
            input_config_json=run_config,
            query_start_date=form.cleaned_data["query_start_date"],
            query_end_date=form.cleaned_data["query_end_date"],
            request=request,
        )
    except (PermissionDenied, ValidationError, WorkspaceCredentialValidationError) as exc:
        messages.error(request, str(exc))
    else:
        request_completion_email = bool(form.cleaned_data.get("request_completion_email"))
        if request_completion_email:
            notify_on = form.cleaned_data.get("notify_on") or "any"
            try:
                create_notification_request(
                    run=run,
                    user=request.user,
                    notify_on=notify_on,
                    request=request,
                )
            except (NotificationRequestError, ValidationError) as exc:
                messages.warning(
                    request,
                    f"Run queued, but completion notification could not be created: {exc}",
                )
            else:
                messages.success(request, "Run queued. Completion email notification requested.")
        else:
            messages.success(request, "Run queued.")

    return redirect(
        "workbook-investigation-detail",
        workbook_id=workbook_id,
        investigation_id=investigation_id,
    )


@require_GET
def run_detail(request, workbook_id, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation", "requested_by"),
        id=run_id,
        workspace_id=workbook_id,
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
            "notifications": run.notification_requests.select_related("user").order_by("-created_at"),
        },
    )


@login_required
@require_POST
def cancel_run(request, workbook_id, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation"),
        id=run_id,
        workspace_id=workbook_id,
    )
    form = RunCancelForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid cancellation payload.")
        return redirect("workbook-run-detail", workbook_id=workbook_id, run_id=run_id)

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
    return redirect("workbook-run-detail", workbook_id=workbook_id, run_id=run_id)


@require_GET
def download_run_artifact(request, workbook_id, run_id, artifact_id):
    artifact = get_object_or_404(
        RunArtifact.objects.select_related("workspace", "run", "run__investigation"),
        id=artifact_id,
        run_id=run_id,
        workspace_id=workbook_id,
    )
    if not can_view_workspace(request.user, artifact.workspace):
        if request.user.is_authenticated:
            raise PermissionDenied("You do not have access to this artifact.")
        return redirect("accounts-login")

    if artifact.status != ArtifactStatus.READY:
        raise Http404("Artifact is not ready.")

    try:
        file_obj, download_name = open_artifact_for_download(artifact)
    except ArtifactStorageError as exc:
        raise Http404(str(exc)) from exc

    record_artifact_download(artifact=artifact, user=request.user, request=request)
    response = FileResponse(file_obj, as_attachment=True, filename=download_name)
    if artifact.size_bytes is not None:
        response["Content-Length"] = str(artifact.size_bytes)
    return response
