import io
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core import mail
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_runs.models import RunStatus, RunType
from wb_runs.services import queue_run
from wb_workspaces.services import create_workspace_for_user

from .models import NotificationStatus, NotificationTrigger
from .services import create_notification_request, dispatch_pending_notifications


User = get_user_model()


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    DEFAULT_FROM_EMAIL="noreply@pfdtoolkit.org",
    WORKBENCH_BASE_URL="https://pfdtoolkit.org",
)
class NotificationDispatchTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="notify-owner@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Notify Workspace",
            slug="notify-workspace",
            description="Notify Desc",
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Notify Investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        self.run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={},
        )

    def test_create_notification_request_writes_audit_event(self):
        notification = create_notification_request(
            run=self.run,
            user=self.owner,
            notify_on=NotificationTrigger.ANY,
        )
        self.assertEqual(notification.status, NotificationStatus.PENDING)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="notification.requested",
                target_id=str(notification.id),
            ).exists()
        )

    def test_dispatch_sends_email_for_matching_trigger(self):
        self.run.status = RunStatus.SUCCEEDED
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "finished_at", "updated_at"])
        notification = create_notification_request(
            run=self.run,
            user=self.owner,
            notify_on=NotificationTrigger.ANY,
        )

        result = dispatch_pending_notifications(max_items=10)
        notification.refresh_from_db()

        self.assertEqual(result.scanned, 1)
        self.assertEqual(result.sent, 1)
        self.assertEqual(notification.status, NotificationStatus.SENT)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Run succeeded", mail.outbox[0].subject)
        self.assertIn(str(self.run.id), mail.outbox[0].body)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="notification.sent",
                target_id=str(notification.id),
            ).exists()
        )

    def test_dispatch_cancels_when_trigger_does_not_match(self):
        self.run.status = RunStatus.SUCCEEDED
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "finished_at", "updated_at"])
        notification = create_notification_request(
            run=self.run,
            user=self.owner,
            notify_on=NotificationTrigger.FAILURE,
        )

        result = dispatch_pending_notifications(max_items=10)
        notification.refresh_from_db()

        self.assertEqual(result.scanned, 1)
        self.assertEqual(result.cancelled, 1)
        self.assertEqual(notification.status, NotificationStatus.CANCELLED)
        self.assertEqual(len(mail.outbox), 0)

    def test_dispatch_marks_failed_when_send_errors(self):
        self.run.status = RunStatus.SUCCEEDED
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "finished_at", "updated_at"])
        notification = create_notification_request(
            run=self.run,
            user=self.owner,
            notify_on=NotificationTrigger.ANY,
        )

        with patch("wb_notifications.services.send_mail", side_effect=RuntimeError("smtp down")):
            result = dispatch_pending_notifications(max_items=10)

        notification.refresh_from_db()
        self.assertEqual(result.scanned, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(notification.status, NotificationStatus.FAILED)
        self.assertIn("smtp down", notification.error_message)

    def test_dispatch_command_once(self):
        self.run.status = RunStatus.SUCCEEDED
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "finished_at", "updated_at"])
        create_notification_request(
            run=self.run,
            user=self.owner,
            notify_on=NotificationTrigger.ANY,
        )

        out = io.StringIO()
        call_command("run_notification_dispatcher", "--once", stdout=out)
        output = out.getvalue()
        self.assertIn("scanned=1", output)
        self.assertIn("sent=1", output)
