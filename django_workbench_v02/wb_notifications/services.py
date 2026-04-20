from __future__ import annotations

from dataclasses import dataclass
from time import sleep

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
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
        f"{settings.WORKBENCH_BASE_URL}/workbooks/{run.workspace_id}/runs/{run.id}/"
    )


def _workspace_detail_url(run) -> str:
    return f"{settings.WORKBENCH_BASE_URL}/workbooks/{run.workspace_id}/"


def _status_presentation(run_status: str) -> dict[str, str]:
    if run_status == RunStatus.SUCCEEDED:
        return {
            "subject_label": "Run complete",
            "headline": "Run complete",
            "status_label": "RUN COMPLETED",
            "status_color": "#78D98B",
            "summary": "Your results are ready to view and download.",
            "button_label": "View results",
            "variant": "success",
        }
    if run_status == RunStatus.FAILED:
        return {
            "subject_label": "Run failed",
            "headline": "Run failed",
            "status_label": "RUN FAILED",
            "status_color": "#FF7A7A",
            "summary": "The run failed before completion. Review details and retry when ready.",
            "button_label": "Review run",
            "variant": "failure",
        }
    if run_status == RunStatus.TIMED_OUT:
        return {
            "subject_label": "Run timed out",
            "headline": "Run timed out",
            "status_label": "RUN TIMED OUT",
            "status_color": "#F6BE5A",
            "summary": "The run timed out. Review details and relaunch if needed.",
            "button_label": "Review run",
            "variant": "warning",
        }
    if run_status == RunStatus.CANCELLED:
        return {
            "subject_label": "Run cancelled",
            "headline": "Run cancelled",
            "status_label": "RUN CANCELLED",
            "status_color": "#A8AEC6",
            "summary": "This run was cancelled before completion.",
            "button_label": "Review run",
            "variant": "neutral",
        }
    return {
        "subject_label": f"Run {run_status}",
        "headline": f"Run {run_status}",
        "status_label": f"RUN {str(run_status).upper()}",
        "status_color": "#A8AEC6",
        "summary": "Run reached a terminal state.",
        "button_label": "View run",
        "variant": "neutral",
    }


def _email_context(notification: NotificationRequest) -> dict[str, str]:
    run = notification.run
    status_ui = _status_presentation(run.status)
    return {
        "subject": _build_subject(notification),
        "investigation_title": run.investigation.title,
        "run_id": str(run.id),
        "run_type": str(run.run_type),
        "run_status": str(run.status),
        "queued_at": str(run.queued_at),
        "started_at": str(run.started_at),
        "finished_at": str(run.finished_at),
        "run_detail_url": _run_detail_url(run),
        "notification_preferences_url": _workspace_detail_url(run),
        "workbench_base_url": settings.WORKBENCH_BASE_URL,
        **status_ui,
    }


def _build_subject(notification: NotificationRequest) -> str:
    run = notification.run
    status_ui = _status_presentation(run.status)
    return f"[PFD Toolkit] {status_ui['subject_label']}: {run.investigation.title}"


def _build_body(notification: NotificationRequest) -> str:
    return render_to_string(
        "wb_notifications/emails/run_status.txt",
        _email_context(notification),
    )


def _build_html_body(notification: NotificationRequest) -> str:
    return render_to_string(
        "wb_notifications/emails/run_status.html",
        _email_context(notification),
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
        message = EmailMultiAlternatives(
            subject=_build_subject(notification),
            body=_build_body(notification),
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[notification.user.email],
        )
        message.attach_alternative(_build_html_body(notification), "text/html")
        message.send(fail_silently=False)
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
