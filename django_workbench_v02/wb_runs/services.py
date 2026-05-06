from __future__ import annotations

from datetime import datetime, time as dt_time

from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import PermissionDenied, ValidationError
from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.services import log_action_cache_event, log_audit_event
from wb_workspaces.activity import is_human_view_request, should_update_last_viewed
from wb_workspaces.permissions import can_run_workflows, can_view_workspace

from .models import (
    InvestigationRun,
    RunApprovalStatus,
    RunArtifact,
    RunEvent,
    RunEventType,
    RunStatus,
)
from .scope import resolve_run_scope_config


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
        RunStatus.QUEUED,  # allows automatic transient retry requeue
        RunStatus.CANCELLING,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
    },
    RunStatus.RUNNING: {
        RunStatus.RUNNING,  # allows progress updates while running
        RunStatus.QUEUED,  # allows automatic transient retry requeue
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


ACTIVE_RUN_STATUSES = {
    RunStatus.STARTING,
    RunStatus.RUNNING,
    RunStatus.CANCELLING,
}

LLM_APPROVAL_RUN_TYPES = {"filter", "themes", "extract"}
EXECUTION_ROUTE_LOCAL = "local"
EXECUTION_ROUTE_API = "api"


def _as_int(value, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _queue_requires_approval(*, run_type: str, config: dict) -> bool:
    if "requires_manual_approval" in config:
        return bool(config.get("requires_manual_approval"))

    provider = str(config.get("provider") or "").strip().lower()
    execution_mode = str(config.get("execution_mode") or "real").strip().lower()
    pipeline_index = _as_int(config.get("pipeline_index"), default=0)
    if execution_mode == "simulate":
        return False
    if provider != "local_ollama":
        return False
    if str(run_type or "").strip().lower() not in LLM_APPROVAL_RUN_TYPES:
        return False
    return pipeline_index <= 0


def _execution_route_for_config(config: dict) -> str:
    explicit = str(config.get("execution_route") or "").strip().lower()
    if explicit in {EXECUTION_ROUTE_LOCAL, EXECUTION_ROUTE_API}:
        return explicit
    provider = str(config.get("provider") or "").strip().lower()
    if provider == "local_ollama":
        return EXECUTION_ROUTE_LOCAL
    return EXECUTION_ROUTE_API


def _workspace_run_logs_url(*, run: InvestigationRun) -> str:
    base = str(getattr(settings, "WORKBENCH_BASE_URL", "") or "").rstrip("/")
    return (
        f"{base}/workbooks/{run.workspace_id}/open/"
        f"?open_run_logs=1&run_id={run.id}"
    )


def _run_admin_change_url(*, run: InvestigationRun) -> str:
    base = str(getattr(settings, "WORKBENCH_BASE_URL", "") or "").rstrip("/")
    return f"{base}{reverse('admin:wb_runs_investigationrun_change', args=[str(run.id)])}"


def _send_approval_request_email(*, run: InvestigationRun, request=None) -> bool:
    recipient = str(getattr(settings, "PFD_ADMIN_EMAIL", "") or "").strip()
    if not recipient:
        return False

    subject = f"[PFD Toolkit] Approval required: {run.investigation.title}"
    body = (
        "A local DGX investigation run is waiting for approval.\n\n"
        f"Run ID: {run.id}\n"
        f"Workspace: {run.workspace.title}\n"
        f"Investigation: {run.investigation.title}\n"
        f"Run type: {run.run_type}\n"
        f"Requested by: {getattr(run.requested_by, 'email', '')}\n"
        f"Queued at: {run.queued_at.isoformat()}\n\n"
        "Approve/schedule in admin:\n"
        f"{_run_admin_change_url(run=run)}\n\n"
        "Workspace run logs:\n"
        f"{_workspace_run_logs_url(run=run)}\n"
    )
    try:
        message = EmailMultiAlternatives(
            subject=subject,
            body=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient],
        )
        message.send(fail_silently=False)
    except Exception as exc:  # pragma: no cover - external email backend
        RunEvent.objects.create(
            run=run,
            event_type=RunEventType.WARNING,
            message="Approval request email failed to send.",
            payload_json={"recipient": recipient, "error": str(exc)},
        )
        log_audit_event(
            action_type="run.approval_email_failed",
            target_type="investigation_run",
            target_id=str(run.id),
            workspace=run.workspace,
            user=run.requested_by if getattr(run.requested_by, "is_authenticated", False) else None,
            payload={"recipient": recipient, "error": str(exc)},
            request=request,
        )
        return False

    RunEvent.objects.create(
        run=run,
        event_type=RunEventType.INFO,
        message="Approval request email sent to admin.",
        payload_json={"recipient": recipient},
    )
    log_audit_event(
        action_type="run.approval_email_sent",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=run.requested_by if getattr(run.requested_by, "is_authenticated", False) else None,
        payload={"recipient": recipient},
        request=request,
    )
    return True


def _extract_client_ip(request) -> str:
    if request is None:
        return ""
    forwarded_for = str(request.META.get("HTTP_X_FORWARDED_FOR", "")).strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return str(request.META.get("REMOTE_ADDR", "")).strip()


def _rate_limit_key(*, scope: str, identifier: str) -> str:
    return f"run_launch_rate_limit:{scope}:{identifier}"


def _check_rate_limit(*, scope: str, identifier: str, limit_per_minute: int) -> bool:
    if limit_per_minute <= 0 or not identifier:
        return False
    key = _rate_limit_key(scope=scope, identifier=identifier)
    timeout_seconds = 60
    if cache.add(key, 1, timeout=timeout_seconds):
        return False
    try:
        count = cache.incr(key)
    except ValueError:
        cache.set(key, 1, timeout=timeout_seconds)
        return False
    return count > limit_per_minute


def _enforce_launch_rate_limits(*, actor, request=None) -> None:
    if not bool(getattr(settings, "RUN_GUARDRAILS_ENABLED", True)):
        return
    if request is None:
        return
    user_limit = int(getattr(settings, "RUN_LAUNCH_RATE_LIMIT_USER_PER_MINUTE", 0))
    if actor is not None and getattr(actor, "is_authenticated", False):
        if _check_rate_limit(
            scope="user",
            identifier=str(actor.id),
            limit_per_minute=user_limit,
        ):
            raise RunServiceError("Rate limit reached for run launches. Try again in about one minute.")

    ip_limit = int(getattr(settings, "RUN_LAUNCH_RATE_LIMIT_IP_PER_MINUTE", 0))
    client_ip = _extract_client_ip(request=request)
    if _check_rate_limit(
        scope="ip",
        identifier=client_ip,
        limit_per_minute=ip_limit,
    ):
        raise RunServiceError("Too many run launch requests from this IP. Try again in about one minute.")


def _enforce_run_caps(*, actor, workspace) -> None:
    if not bool(getattr(settings, "RUN_GUARDRAILS_ENABLED", True)):
        return

    now = timezone.now()
    local_day_start = timezone.make_aware(
        datetime.combine(timezone.localdate(now), dt_time.min),
        timezone.get_current_timezone(),
    )

    if actor is not None and getattr(actor, "is_authenticated", False):
        max_runs_per_user_per_day = int(getattr(settings, "MAX_RUNS_PER_USER_PER_DAY", 0))
        if max_runs_per_user_per_day > 0:
            user_daily_count = InvestigationRun.objects.filter(
                requested_by=actor,
                created_at__gte=local_day_start,
            ).count()
            if user_daily_count >= max_runs_per_user_per_day:
                raise RunServiceError(
                    f"You reached the daily run cap ({max_runs_per_user_per_day} runs per day)."
                )

        max_concurrent_user = int(getattr(settings, "MAX_CONCURRENT_RUNS_PER_USER", 0))
        if max_concurrent_user > 0:
            user_inflight_count = InvestigationRun.objects.filter(
                requested_by=actor,
                status__in=ACTIVE_RUN_STATUSES,
            ).count()
            if user_inflight_count >= max_concurrent_user:
                raise RunServiceError(
                    f"You reached the concurrent run cap ({max_concurrent_user} in-flight runs)."
                )

    max_runs_per_workbook_per_day = int(getattr(settings, "MAX_RUNS_PER_WORKBOOK_PER_DAY", 0))
    if max_runs_per_workbook_per_day > 0:
        workbook_daily_count = InvestigationRun.objects.filter(
            workspace=workspace,
            created_at__gte=local_day_start,
        ).count()
        if workbook_daily_count >= max_runs_per_workbook_per_day:
            raise RunServiceError(
                f"This workbook reached its daily run cap ({max_runs_per_workbook_per_day} runs per day)."
            )

    max_concurrent_global = int(getattr(settings, "MAX_CONCURRENT_RUNS_GLOBAL", 0))
    if max_concurrent_global > 0:
        global_inflight_count = InvestigationRun.objects.filter(
            status__in=ACTIVE_RUN_STATUSES,
        ).count()
        if global_inflight_count >= max_concurrent_global:
            raise RunServiceError("Run queue is currently at global concurrency capacity. Try again shortly.")


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
        raise PermissionDenied("You do not have permission to run workflows in this workbook.")
    _enforce_launch_rate_limits(actor=actor, request=request)
    _enforce_run_caps(actor=actor, workspace=workspace)

    resolved_config = resolve_run_scope_config(
        investigation=investigation,
        input_config_json=input_config_json,
    )
    requires_approval = _queue_requires_approval(
        run_type=run_type,
        config=resolved_config,
    )
    resolved_config["requires_manual_approval"] = requires_approval
    resolved_config["execution_route"] = _execution_route_for_config(resolved_config)
    approval_status = (
        RunApprovalStatus.PENDING if requires_approval else RunApprovalStatus.NOT_REQUIRED
    )
    approval_requested_at = timezone.now() if requires_approval else None

    run = InvestigationRun.objects.create(
        investigation=investigation,
        workspace=workspace,
        requested_by=actor,
        run_type=run_type,
        status=RunStatus.QUEUED,
        input_config_json=resolved_config,
        requires_approval=requires_approval,
        approval_status=approval_status,
        approval_requested_at=approval_requested_at,
        query_start_date=query_start_date,
        query_end_date=query_end_date,
    )
    RunEvent.objects.create(
        run=run,
        event_type=RunEventType.INFO,
        message="Run queued.",
        payload_json={
            "status": run.status,
            "requires_approval": run.requires_approval,
            "approval_status": run.approval_status,
        },
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
            "requires_approval": run.requires_approval,
            "approval_status": run.approval_status,
        },
        request=request,
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="run.queue",
        entity_type="investigation_run",
        entity_id=str(run.id),
        query={
            "query_start_date": query_start_date.isoformat() if query_start_date else None,
            "query_end_date": query_end_date.isoformat() if query_end_date else None,
        },
        options={
            "run_type": run_type,
            "input_config_json": resolved_config,
        },
        state_before={},
        state_after={"status": run.status},
        context={"investigation_id": str(investigation.id)},
    )
    if requires_approval:
        log_audit_event(
            action_type="run.approval_requested",
            target_type="investigation_run",
            target_id=str(run.id),
            workspace=workspace,
            user=actor,
            payload={
                "provider": str(resolved_config.get("provider") or "").strip().lower(),
                "run_type": run.run_type,
                "approval_status": run.approval_status,
            },
            request=request,
        )
        _send_approval_request_email(run=run, request=request)
    return run


def _can_administer_run_approval(*, actor) -> bool:
    return bool(actor and getattr(actor, "is_authenticated", False) and getattr(actor, "is_superuser", False))


@transaction.atomic
def approve_run_for_execution(
    *,
    actor,
    run: InvestigationRun,
    note: str = "",
    scheduled_for=None,
    request=None,
) -> InvestigationRun:
    if not _can_administer_run_approval(actor=actor):
        raise PermissionDenied("Only superusers can approve queued runs.")
    if run.status != RunStatus.QUEUED:
        raise RunServiceError("Only queued runs can be approved.")
    if run.approval_status == RunApprovalStatus.REJECTED:
        raise RunServiceError("Rejected runs cannot be approved; requeue a new run instead.")

    run.requires_approval = True
    run.approval_status = RunApprovalStatus.APPROVED
    run.approved_at = timezone.now()
    run.approved_by = actor
    run.rejected_at = None
    run.rejected_by = None
    run.approval_note = str(note or "").strip()
    if scheduled_for is not None:
        run.queued_at = scheduled_for
    run.save(
        update_fields=[
            "requires_approval",
            "approval_status",
            "approved_at",
            "approved_by",
            "rejected_at",
            "rejected_by",
            "approval_note",
            "queued_at",
            "updated_at",
        ]
    )
    RunEvent.objects.create(
        run=run,
        event_type=RunEventType.INFO,
        message="Run queued.",
        payload_json={
            "approval_status": run.approval_status,
            "approved_by": str(actor.id),
            "scheduled_for": run.queued_at.isoformat() if run.queued_at else None,
        },
    )
    log_audit_event(
        action_type="run.approved",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=actor,
        payload={
            "status": run.status,
            "approval_status": run.approval_status,
            "scheduled_for": run.queued_at.isoformat() if run.queued_at else None,
        },
        request=request,
    )
    return run


@transaction.atomic
def reject_run_for_execution(
    *,
    actor,
    run: InvestigationRun,
    reason: str = "",
    request=None,
) -> InvestigationRun:
    if not _can_administer_run_approval(actor=actor):
        raise PermissionDenied("Only superusers can reject queued runs.")
    if run.status != RunStatus.QUEUED:
        raise RunServiceError("Only queued runs can be rejected.")

    run.requires_approval = True
    run.approval_status = RunApprovalStatus.REJECTED
    run.rejected_at = timezone.now()
    run.rejected_by = actor
    run.approval_note = str(reason or "").strip()
    run.save(
        update_fields=[
            "requires_approval",
            "approval_status",
            "rejected_at",
            "rejected_by",
            "approval_note",
            "updated_at",
        ]
    )
    cancelled = set_run_status(
        run=run,
        status=RunStatus.CANCELLED,
        message="Run was rejected in admin and will not be executed.",
        actor=actor,
        event_type=RunEventType.WARNING,
        request=request,
    )
    log_audit_event(
        action_type="run.rejected",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=actor,
        payload={
            "status": cancelled.status,
            "approval_status": cancelled.approval_status,
            "reason": cancelled.approval_note,
        },
        request=request,
    )
    return cancelled


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
    log_action_cache_event(
        workspace=run.workspace,
        user=actor,
        action_key="run.cancel_request",
        entity_type="investigation_run",
        entity_id=str(run.id),
        options={"reason": run.cancel_reason},
        state_before={},
        state_after={
            "status": run.status,
            "cancel_requested_at": run.cancel_requested_at.isoformat()
            if run.cancel_requested_at
            else None,
        },
        context={"investigation_id": str(run.investigation_id)},
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

    now = timezone.now()
    is_human_view = is_human_view_request(request=request)
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=run.workspace.last_viewed_at,
        now=now,
    ):
        run.workspace.last_viewed_at = now
        run.workspace.save(update_fields=["last_viewed_at", "updated_at"])

    investigation = run.investigation
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=investigation.last_viewed_at,
        now=now,
    ):
        investigation.last_viewed_at = now
        investigation.save(update_fields=["last_viewed_at", "updated_at"])

    log_audit_event(
        action_type="run.viewed",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=user if user and user.is_authenticated else None,
        payload={
            "status": run.status,
            "run_type": run.run_type,
            "is_human_view": is_human_view,
        },
        request=request,
    )


def record_artifact_download(*, artifact: RunArtifact, user=None, request=None) -> None:
    if not can_view_workspace(user, artifact.workspace):
        raise PermissionDenied("You do not have permission to download this artifact.")

    now = timezone.now()
    is_human_view = is_human_view_request(request=request)
    if is_human_view:
        if should_update_last_viewed(
            existing_last_viewed_at=artifact.last_viewed_at,
            now=now,
        ):
            artifact.last_viewed_at = now
            artifact.save(update_fields=["last_viewed_at", "updated_at"])
        workspace = artifact.workspace
        if should_update_last_viewed(
            existing_last_viewed_at=workspace.last_viewed_at,
            now=now,
        ):
            workspace.last_viewed_at = now
            workspace.save(update_fields=["last_viewed_at", "updated_at"])

        investigation = artifact.run.investigation
        if should_update_last_viewed(
            existing_last_viewed_at=investigation.last_viewed_at,
            now=now,
        ):
            investigation.last_viewed_at = now
            investigation.save(update_fields=["last_viewed_at", "updated_at"])

    log_audit_event(
        action_type="run.artifact_downloaded",
        target_type="run_artifact",
        target_id=str(artifact.id),
        workspace=artifact.workspace,
        user=user if user and user.is_authenticated else None,
        payload={
            "artifact_type": artifact.artifact_type,
            "run_id": str(artifact.run_id),
            "storage_backend": artifact.storage_backend,
            "is_human_view": is_human_view,
        },
        request=request,
    )
