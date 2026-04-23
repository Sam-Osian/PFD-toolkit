import json

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
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
from .models import ArtifactStatus, InvestigationRun, RunArtifact, RunStatus
from .services import (
    queue_run,
    request_run_cancellation,
    record_artifact_download,
    record_run_view,
)

TERMINAL_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}

RUN_STATUS_LABELS = dict(RunStatus.choices)
ARTIFACT_TYPE_LABELS = dict(RunArtifact._meta.get_field("artifact_type").choices)
RUN_TYPE_LABELS = dict(InvestigationRun._meta.get_field("run_type").choices)

ARTIFACT_INTENT_BY_TYPE = {
    "filtered_dataset": "Matched report subset after filtering.",
    "theme_summary": "Topline theme outputs from the theme workflow.",
    "theme_assignments": "Per-report theme assignment table.",
    "extraction_table": "Structured extract table generated from prompts.",
    "bundle_export": "Packaged workbook export bundle.",
    "preview": "Preview output produced during processing.",
}

CONFIG_SUMMARY_KEYS = [
    ("execution_mode", "Execution mode"),
    ("provider", "Provider"),
    ("model_name", "Model"),
    ("collection_slug", "Collection"),
    ("collection_query", "Collection query"),
    ("query_text", "Query text"),
    ("search_query", "Search query"),
    ("pipeline_plan", "Pipeline plan"),
    ("pipeline_continue_on_fail", "Continue on fail"),
    ("pipeline_index", "Pipeline index"),
    ("input_artifact_id", "Upstream artifact"),
]

CONFIG_SKIP_KEYS = {
    "report_identity_allowlist",
    "excluded_report_identities",
    "selected_filters",
}


def _format_size(size_bytes):
    if size_bytes is None:
        return "-"
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_config_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "-"
    if isinstance(value, list):
        if not value:
            return "none"
        if len(value) <= 4:
            return ", ".join(str(item) for item in value)
        return f"{len(value)} values"
    if isinstance(value, dict):
        if not value:
            return "none"
        keys = list(value.keys())
        preview = ", ".join(str(item) for item in keys[:3])
        if len(keys) > 3:
            preview = f"{preview}, +{len(keys) - 3} more"
        return f"{len(keys)} keys ({preview})"
    return str(value)


def _humanize_config_key(key):
    return key.replace("_", " ").capitalize()


def _build_run_journey(run):
    steps = [
        {
            "label": "Queued",
            "state": "done",
            "state_label": "done",
            "timestamp": run.queued_at,
            "note": "Run entered the queue.",
        },
        {
            "label": "Started",
            "state": "pending",
            "state_label": "pending",
            "timestamp": run.started_at,
            "note": "Worker started processing.",
        },
        {
            "label": "Running",
            "state": "pending",
            "state_label": "pending",
            "timestamp": run.started_at,
            "note": "Workflow execution in progress.",
        },
        {
            "label": "Terminal",
            "state": "pending",
            "state_label": "pending",
            "timestamp": run.finished_at,
            "note": "Final run status has been recorded.",
        },
    ]
    status = run.status
    if run.started_at:
        steps[1]["state"] = "done"
        steps[1]["state_label"] = "done"
    elif status == RunStatus.STARTING:
        steps[1]["state"] = "current"
        steps[1]["state_label"] = "current"
    elif status in TERMINAL_STATUSES or status in {RunStatus.RUNNING, RunStatus.CANCELLING}:
        steps[1]["state"] = "skipped"
        steps[1]["state_label"] = "skipped"
        steps[1]["timestamp"] = None
        steps[1]["note"] = "Run never reached an explicit start timestamp."

    if status in {RunStatus.RUNNING, RunStatus.CANCELLING}:
        steps[2]["state"] = "current"
        steps[2]["state_label"] = "current"
    elif status in TERMINAL_STATUSES:
        steps[2]["state"] = "done" if run.started_at else "skipped"
        steps[2]["state_label"] = "done" if run.started_at else "skipped"
    elif status == RunStatus.STARTING:
        steps[2]["state"] = "pending"
        steps[2]["state_label"] = "pending"

    if status in TERMINAL_STATUSES:
        steps[3]["state"] = "done"
        steps[3]["state_label"] = RUN_STATUS_LABELS.get(status, status)
        steps[3]["note"] = f"Run finished with status: {RUN_STATUS_LABELS.get(status, status)}."
    elif status == RunStatus.CANCELLING:
        steps[3]["state"] = "current"
        steps[3]["state_label"] = "waiting"
        steps[3]["note"] = "Cancellation requested. Waiting for worker to finalize."
    return steps


def _build_artifact_groups(artifacts):
    grouped = {}
    for artifact in artifacts:
        artifact_type = artifact.artifact_type
        group = grouped.setdefault(
            artifact_type,
            {
                "artifact_type": artifact_type,
                "label": ARTIFACT_TYPE_LABELS.get(artifact_type, artifact_type),
                "intent": ARTIFACT_INTENT_BY_TYPE.get(artifact_type, ""),
                "entries": [],
            },
        )
        group["entries"].append(
            {
                "artifact": artifact,
                "downloadable": (
                    artifact.status == ArtifactStatus.READY
                    and artifact.storage_backend != "db"
                    and bool(artifact.storage_uri)
                ),
                "size_display": _format_size(artifact.size_bytes),
            }
        )
    return list(grouped.values())


def _build_config_summary(run):
    config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
    rows = []
    captured_keys = set()
    for key, label in CONFIG_SUMMARY_KEYS:
        if key in config:
            rows.append({"label": label, "value": _format_config_value(config.get(key))})
            captured_keys.add(key)

    if run.query_start_date or run.query_end_date:
        rows.append(
            {
                "label": "Query date range",
                "value": f"{run.query_start_date or '-'} to {run.query_end_date or '-'}",
            }
        )

    for key in sorted(config.keys()):
        if key in captured_keys or key in CONFIG_SKIP_KEYS:
            continue
        rows.append({"label": _humanize_config_key(key), "value": _format_config_value(config.get(key))})
    return rows


def _build_outcome_summary(run, artifact_count):
    status_label = RUN_STATUS_LABELS.get(run.status, run.status)
    if run.status == RunStatus.SUCCEEDED:
        return {
            "title": "Run completed",
            "description": (
                f"{RUN_TYPE_LABELS.get(run.run_type, run.run_type)} run finished successfully "
                f"with {artifact_count} artifact(s)."
            ),
            "next_step": "Review artifacts and download outputs as needed.",
        }
    if run.status == RunStatus.CANCELLED:
        reason = run.cancel_reason.strip() or "No cancellation reason provided."
        return {
            "title": "Run cancelled",
            "description": f"Run was cancelled. Reason: {reason}",
            "next_step": "Re-launch the run when you are ready.",
        }
    if run.status == RunStatus.CANCELLING:
        return {
            "title": "Cancellation in progress",
            "description": "A cancellation request is queued with the worker.",
            "next_step": "This page updates once cancellation reaches terminal state.",
        }
    if run.status in {RunStatus.FAILED, RunStatus.TIMED_OUT}:
        error = run.error_message.strip() or "No explicit error details were recorded."
        return {
            "title": f"Run {status_label.lower()}",
            "description": error,
            "next_step": "Check events below, adjust configuration, and retry.",
        }
    return {
        "title": "Run in progress",
        "description": f"Current status: {status_label}.",
        "next_step": "You can leave this page; server-side processing will continue.",
    }


def _build_cancellation_message(run):
    if run.status == RunStatus.CANCELLING:
        return "Cancellation has been requested and will be finalized by the worker."
    if run.status == RunStatus.CANCELLED:
        return "This run has been cancelled."
    return ""


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
        return redirect("workbook-open", workbook_id=workbook_id)

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

    return redirect("workbook-open", workbook_id=workbook_id)


@require_GET
def run_detail(request, workbook_id, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation", "requested_by"),
        id=run_id,
        workspace_id=workbook_id,
    )
    if not can_view_workspace(request.user, run.workspace):
        return redirect("accounts-login")
    # Retired user-facing run detail page: canonical user view is workbook-open.
    return redirect(f"{reverse('workbook-open', kwargs={'workbook_id': workbook_id})}?open_run_logs=1&run_id={run_id}")


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
        return redirect("workbook-open", workbook_id=workbook_id)

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
    return redirect("workbook-open", workbook_id=workbook_id)


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
