from __future__ import annotations

from dataclasses import dataclass
from time import sleep

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.utils import timezone

from wb_auditlog.services import log_audit_event
from wb_runs.models import RunStatus

from .models import NotificationChannel, NotificationRequest, NotificationStatus, NotificationTrigger


TERMINAL_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}
FAILURE_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}


class NotificationRequestError(ValidationError):
    pass


@dataclass
class NotificationDispatchResult:
    scanned: int = 0
    sent: int = 0
    failed: int = 0
    cancelled: int = 0


def _should_send_for_trigger(*, notify_on: str, run_status: str) -> bool:
    if notify_on == NotificationTrigger.ANY:
        return run_status in TERMINAL_STATUSES
    if notify_on == NotificationTrigger.SUCCESS:
        return run_status == RunStatus.SUCCEEDED
    if notify_on == NotificationTrigger.FAILURE:
        return run_status in FAILURE_STATUSES
    return False


def _run_detail_url(run) -> str:
    return (
        f"{settings.WORKBENCH_BASE_URL}/workspaces/{run.workspace_id}/runs/{run.id}/"
    )


def _build_subject(notification: NotificationRequest) -> str:
    run = notification.run
    return f"[PFD Toolkit] Run {run.status}: {run.investigation.title}"


def _build_body(notification: NotificationRequest) -> str:
    run = notification.run
    return (
        "Your run has completed.\n\n"
        f"Investigation: {run.investigation.title}\n"
        f"Run ID: {run.id}\n"
        f"Run type: {run.run_type}\n"
        f"Status: {run.status}\n"
        f"Queued at: {run.queued_at}\n"
        f"Started at: {run.started_at}\n"
        f"Finished at: {run.finished_at}\n\n"
        f"View run details: {_run_detail_url(run)}\n"
    )


def create_notification_request(
    *,
    run,
    user,
    notify_on: str = NotificationTrigger.ANY,
    channel: str = NotificationChannel.EMAIL,
    request=None,
) -> NotificationRequest:
    if channel != NotificationChannel.EMAIL:
        raise NotificationRequestError("Only email notifications are currently supported.")
    if notify_on not in NotificationTrigger.values:
        raise NotificationRequestError("Invalid notification trigger.")
    if not user.email:
        raise NotificationRequestError("User account has no email address for notifications.")

    existing = NotificationRequest.objects.filter(
        run=run,
        user=user,
        channel=channel,
        notify_on=notify_on,
        status__in=[NotificationStatus.PENDING, NotificationStatus.SENT],
    ).first()
    if existing:
        return existing

    notification = NotificationRequest.objects.create(
        run=run,
        user=user,
        channel=channel,
        notify_on=notify_on,
        status=NotificationStatus.PENDING,
    )
    log_audit_event(
        action_type="notification.requested",
        target_type="notification_request",
        target_id=str(notification.id),
        workspace=run.workspace,
        user=user,
        payload={
            "run_id": str(run.id),
            "channel": notification.channel,
            "notify_on": notification.notify_on,
            "status": notification.status,
        },
        request=request,
    )
    return notification


def process_pending_notification(notification: NotificationRequest) -> str:
    if notification.status != NotificationStatus.PENDING:
        return notification.status

    run = notification.run
    if run.status not in TERMINAL_STATUSES:
        return notification.status

    if not _should_send_for_trigger(notify_on=notification.notify_on, run_status=run.status):
        notification.status = NotificationStatus.CANCELLED
        notification.error_message = (
            f"Run terminal status '{run.status}' did not match trigger '{notification.notify_on}'."
        )
        notification.save(update_fields=["status", "error_message", "updated_at"])
        log_audit_event(
            action_type="notification.cancelled_trigger_mismatch",
            target_type="notification_request",
            target_id=str(notification.id),
            workspace=run.workspace,
            user=notification.user,
            payload={
                "run_id": str(run.id),
                "run_status": run.status,
                "notify_on": notification.notify_on,
            },
        )
        return notification.status

    if not notification.user.email:
        notification.status = NotificationStatus.FAILED
        notification.error_message = "No recipient email on user account."
        notification.save(update_fields=["status", "error_message", "updated_at"])
        log_audit_event(
            action_type="notification.failed_no_email",
            target_type="notification_request",
            target_id=str(notification.id),
            workspace=run.workspace,
            user=notification.user,
            payload={"run_id": str(run.id), "run_status": run.status},
        )
        return notification.status

    try:
        send_mail(
            subject=_build_subject(notification),
            message=_build_body(notification),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[notification.user.email],
            fail_silently=False,
        )
    except Exception as exc:  # pragma: no cover - defensive external dependency path
        notification.status = NotificationStatus.FAILED
        notification.error_message = str(exc)
        notification.save(update_fields=["status", "error_message", "updated_at"])
        log_audit_event(
            action_type="notification.failed_send_error",
            target_type="notification_request",
            target_id=str(notification.id),
            workspace=run.workspace,
            user=notification.user,
            payload={"run_id": str(run.id), "run_status": run.status, "error": str(exc)},
        )
        return notification.status

    notification.status = NotificationStatus.SENT
    notification.sent_at = timezone.now()
    notification.error_message = ""
    notification.save(update_fields=["status", "sent_at", "error_message", "updated_at"])
    log_audit_event(
        action_type="notification.sent",
        target_type="notification_request",
        target_id=str(notification.id),
        workspace=run.workspace,
        user=notification.user,
        payload={
            "run_id": str(run.id),
            "run_status": run.status,
            "channel": notification.channel,
            "notify_on": notification.notify_on,
        },
    )
    return notification.status


def dispatch_pending_notifications(*, max_items: int = 50) -> NotificationDispatchResult:
    result = NotificationDispatchResult()
    pending = (
        NotificationRequest.objects.select_related(
            "run",
            "run__workspace",
            "run__investigation",
            "user",
        )
        .filter(status=NotificationStatus.PENDING, run__status__in=TERMINAL_STATUSES)
        .order_by("created_at")[:max_items]
    )
    for notification in pending:
        result.scanned += 1
        status = process_pending_notification(notification)
        if status == NotificationStatus.SENT:
            result.sent += 1
        elif status == NotificationStatus.FAILED:
            result.failed += 1
        elif status == NotificationStatus.CANCELLED:
            result.cancelled += 1
    return result


def run_notification_dispatch_loop(
    *,
    poll_seconds: float = 5.0,
    max_items_per_cycle: int = 50,
    max_cycles: int | None = None,
) -> NotificationDispatchResult:
    total = NotificationDispatchResult()
    cycles = 0
    while True:
        cycle = dispatch_pending_notifications(max_items=max_items_per_cycle)
        total.scanned += cycle.scanned
        total.sent += cycle.sent
        total.failed += cycle.failed
        total.cancelled += cycle.cancelled

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return total
        if cycle.scanned == 0:
            sleep(poll_seconds)
