from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_audit_event
from wb_workspaces.permissions import can_run_workflows, can_view_workspace

from .models import InvestigationRun, RunEvent, RunEventType, RunStatus


class RunServiceError(ValidationError):
    pass


TERMINAL_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}

ALLOWED_STATUS_TRANSITIONS = {
    RunStatus.QUEUED: {
        RunStatus.STARTING,
        RunStatus.CANCELLING,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
    },
    RunStatus.STARTING: {
        RunStatus.RUNNING,
        RunStatus.CANCELLING,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
    },
    RunStatus.RUNNING: {
        RunStatus.RUNNING,  # allows progress updates while running
        RunStatus.CANCELLING,
        RunStatus.CANCELLED,
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
    },
    RunStatus.CANCELLING: {
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
    },
}


def is_terminal_status(status: str) -> bool:
    return status in TERMINAL_STATUSES


def _validate_status_transition(current_status: str, next_status: str) -> None:
    if current_status in TERMINAL_STATUSES:
        raise RunServiceError("Cannot transition a terminal run.")
    allowed = ALLOWED_STATUS_TRANSITIONS.get(current_status, set())
    if next_status not in allowed:
        raise RunServiceError(f"Invalid run transition: {current_status} -> {next_status}.")


@transaction.atomic
def queue_run(
    *,
    actor,
    investigation,
    run_type: str,
    input_config_json: dict | None,
    query_start_date=None,
    query_end_date=None,
    request=None,
) -> InvestigationRun:
    workspace = investigation.workspace
    if not can_run_workflows(actor, workspace):
        raise PermissionDenied("You do not have permission to run workflows in this workspace.")

    run = InvestigationRun.objects.create(
        investigation=investigation,
        workspace=workspace,
        requested_by=actor,
        run_type=run_type,
        status=RunStatus.QUEUED,
        input_config_json=input_config_json or {},
        query_start_date=query_start_date,
        query_end_date=query_end_date,
    )
    RunEvent.objects.create(
        run=run,
        event_type=RunEventType.INFO,
        message="Run queued.",
        payload_json={"status": run.status},
    )
    log_audit_event(
        action_type="run.queued",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=workspace,
        user=actor,
        payload={
            "investigation_id": str(investigation.id),
            "run_type": run.run_type,
            "status": run.status,
        },
        request=request,
    )
    return run


@transaction.atomic
def set_run_status(
    *,
    run: InvestigationRun,
    status: str,
    message: str,
    actor=None,
    event_type: str = RunEventType.STAGE,
    progress_percent=None,
    error_code: str = "",
    error_message: str = "",
    request=None,
) -> InvestigationRun:
    _validate_status_transition(run.status, status)

    run.status = status
    if progress_percent is not None:
        run.progress_percent = progress_percent
    if status == RunStatus.STARTING:
        run.started_at = run.started_at or timezone.now()
    if status in TERMINAL_STATUSES:
        run.finished_at = timezone.now()
    if error_code:
        run.error_code = error_code
    if error_message:
        run.error_message = error_message
    run.save()

    RunEvent.objects.create(
        run=run,
        event_type=event_type,
        message=message,
        payload_json={
            "status": run.status,
            "progress_percent": run.progress_percent,
            "error_code": run.error_code,
            "error_message": run.error_message,
        },
    )
    log_audit_event(
        action_type="run.status_changed",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=actor if actor and getattr(actor, "is_authenticated", False) else None,
        payload={
            "status": run.status,
            "message": message,
            "progress_percent": run.progress_percent,
        },
        request=request,
    )
    return run


@transaction.atomic
def request_run_cancellation(
    *,
    actor,
    run: InvestigationRun,
    reason: str = "",
    request=None,
) -> InvestigationRun:
    allowed = (
        (actor and getattr(actor, "is_superuser", False))
        or (actor and actor.id == run.requested_by_id)
        or can_run_workflows(actor, run.workspace)
    )
    if not allowed:
        raise PermissionDenied("You do not have permission to cancel this run.")

    if is_terminal_status(run.status):
        raise RunServiceError("Run is already in a terminal state.")

    run.cancel_requested_at = timezone.now()
    run.cancel_requested_by = actor
    run.cancel_reason = reason or ""
    run.status = RunStatus.CANCELLING
    run.save(
        update_fields=[
            "cancel_requested_at",
            "cancel_requested_by",
            "cancel_reason",
            "status",
            "updated_at",
        ]
    )
    RunEvent.objects.create(
        run=run,
        event_type=RunEventType.CANCEL_CHECK,
        message="Cancellation requested.",
        payload_json={"reason": run.cancel_reason},
    )
    log_audit_event(
        action_type="run.cancel_requested",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=actor,
        payload={"reason": run.cancel_reason, "status": run.status},
        request=request,
    )
    return run


def record_run_view(*, run: InvestigationRun, user=None, request=None) -> None:
    if not can_view_workspace(user, run.workspace):
        raise PermissionDenied("You do not have permission to view this run.")
    log_audit_event(
        action_type="run.viewed",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=user if user and user.is_authenticated else None,
        payload={"status": run.status, "run_type": run.run_type},
        request=request,
    )
