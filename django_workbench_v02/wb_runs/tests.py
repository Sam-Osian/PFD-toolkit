from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import TestCase
from django.urls import reverse

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_workspaces.models import MembershipAccessMode, MembershipRole, WorkspaceMembership
from wb_workspaces.services import create_workspace_for_user

from .models import RunStatus, RunType
from .services import queue_run, request_run_cancellation


User = get_user_model()


class RunServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="run-owner@example.com", password="x")
        self.viewer = User.objects.create_user(email="run-viewer@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Run Workspace",
            slug="run-workspace",
            description="desc",
        )
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=self.viewer,
            role=MembershipRole.VIEWER,
            access_mode=MembershipAccessMode.READ_ONLY,
            can_manage_members=False,
            can_manage_shares=False,
            can_run_workflows=False,
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Run Investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )

    def test_owner_can_queue_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"x": 1},
        )
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.queued",
                target_id=str(run.id),
            ).exists()
        )
        self.assertEqual(run.events.count(), 1)

    def test_viewer_cannot_queue_run(self):
        with self.assertRaises(PermissionDenied):
            queue_run(
                actor=self.viewer,
                investigation=self.investigation,
                run_type=RunType.FILTER,
                input_config_json={},
            )

    def test_cancel_changes_status_to_cancelling(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={},
        )
        request_run_cancellation(actor=self.owner, run=run, reason="Stop")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.CANCELLING)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.cancel_requested",
                target_id=str(run.id),
            ).exists()
        )

    def test_cannot_cancel_terminal_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={},
        )
        run.status = RunStatus.SUCCEEDED
        run.save(update_fields=["status", "updated_at"])
        with self.assertRaises(ValidationError):
            request_run_cancellation(actor=self.owner, run=run)


class RunViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="run-owner2@example.com", password="x")
        self.viewer = User.objects.create_user(email="run-viewer2@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Run Workspace2",
            slug="run-workspace2",
            description="desc",
        )
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=self.viewer,
            role=MembershipRole.VIEWER,
            access_mode=MembershipAccessMode.READ_ONLY,
            can_manage_members=False,
            can_manage_shares=False,
            can_run_workflows=False,
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Run Investigation 2",
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

    def test_owner_can_queue_run_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "run-queue",
                kwargs={
                    "workspace_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.EXPORT,
                "input_config_json": '{"export": true}',
                "query_start_date": "",
                "query_end_date": "",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            self.investigation.runs.filter(run_type=RunType.EXPORT).exists()
        )

    def test_viewer_cannot_queue_run_via_view(self):
        self.client.force_login(self.viewer)
        response = self.client.post(
            reverse(
                "run-queue",
                kwargs={
                    "workspace_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.EXPORT,
                "input_config_json": '{"export": true}',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(
            self.investigation.runs.filter(run_type=RunType.EXPORT).exists()
        )

    def test_run_detail_requires_workspace_view_access(self):
        stranger = User.objects.create_user(email="run-stranger@example.com", password="x")
        self.client.force_login(stranger)
        response = self.client.get(
            reverse(
                "run-detail",
                kwargs={"workspace_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_owner_can_cancel_run_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "run-cancel",
                kwargs={"workspace_id": self.workspace.id, "run_id": self.run.id},
            ),
            data={"cancel_reason": "Stop now"},
        )
        self.assertEqual(response.status_code, 302)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, RunStatus.CANCELLING)
