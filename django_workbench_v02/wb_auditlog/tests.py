from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_runs.models import RunType
from wb_runs.services import queue_run
from wb_workspaces.models import MembershipAccessMode, MembershipRole, WorkspaceMembership
from wb_workspaces.services import create_workspace_for_user

from .models import ActionCacheEvent
from .services import log_action_cache_event


User = get_user_model()


class ActionCacheServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="audit-owner@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Audit Workspace",
            slug="audit-workspace",
            description="",
        )

    def test_log_action_cache_event_persists_payloads(self):
        event = log_action_cache_event(
            workspace=self.workspace,
            user=self.owner,
            action_key="test.action",
            entity_type="workspace",
            entity_id=str(self.workspace.id),
            query={"q": "medication"},
            options={"limit": 50},
            state_before={"a": 1},
            state_after={"a": 2},
            context={"source": "test"},
        )
        self.assertEqual(event.action_key, "test.action")
        self.assertEqual(event.query_json.get("q"), "medication")
        self.assertEqual(event.state_before_json.get("a"), 1)
        self.assertEqual(event.state_after_json.get("a"), 2)

    def test_queue_run_writes_action_cache_entry(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        self.assertTrue(
            ActionCacheEvent.objects.filter(
                workspace=self.workspace,
                action_key="run.queue",
                entity_id=str(run.id),
            ).exists()
        )


class ActionCacheViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="audit-owner2@example.com", password="x")
        self.viewer = User.objects.create_user(email="audit-viewer2@example.com", password="x")
        self.admin = User.objects.create_superuser(email="audit-admin@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Audit Workspace 2",
            slug="audit-workspace-2",
            description="",
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
        log_action_cache_event(
            workspace=self.workspace,
            user=self.owner,
            action_key="test.action",
            entity_type="workspace",
            entity_id=str(self.workspace.id),
            state_after={"hello": "world"},
        )

    def test_owner_can_view_action_cache(self):
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse("workbook-action-cache", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workbook Action Cache")
        self.assertContains(response, "test.action")

    def test_admin_can_view_action_cache(self):
        self.client.force_login(self.admin)
        response = self.client.get(
            reverse("workbook-action-cache", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 200)

    def test_viewer_cannot_view_action_cache(self):
        self.client.force_login(self.viewer)
        response = self.client.get(
            reverse("workbook-action-cache", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 403)
