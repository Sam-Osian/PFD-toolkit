from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from django.urls import reverse

from wb_auditlog.models import AuditEvent
from wb_workspaces.models import MembershipAccessMode, MembershipRole, WorkspaceMembership, WorkspaceVisibility
from wb_workspaces.services import create_workspace_for_user

from .models import Investigation, InvestigationStatus
from .services import create_investigation, update_investigation


User = get_user_model()


class InvestigationServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="inv-owner@example.com", password="x")
        self.viewer = User.objects.create_user(email="inv-viewer@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Inv Workspace",
            slug="inv-workspace",
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

    def test_owner_can_create_investigation(self):
        inv = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Q1",
            question_text="What is happening?",
            scope_json={"dataset": "all"},
            method_json={"model": "x"},
            status=InvestigationStatus.DRAFT,
        )
        self.assertEqual(inv.created_by, self.owner)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="investigation.created",
                target_id=str(inv.id),
            ).exists()
        )

    def test_viewer_cannot_create_investigation(self):
        with self.assertRaises(PermissionDenied):
            create_investigation(
                actor=self.viewer,
                workspace=self.workspace,
                title="Q2",
                question_text="No",
                scope_json={},
                method_json={},
                status=InvestigationStatus.DRAFT,
            )

    def test_owner_can_update_investigation(self):
        inv = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Base",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.DRAFT,
        )
        update_investigation(
            actor=self.owner,
            investigation=inv,
            title="Updated",
            question_text="Q2",
            scope_json={"a": 1},
            method_json={"b": 2},
            status=InvestigationStatus.ACTIVE,
        )
        inv.refresh_from_db()
        self.assertEqual(inv.title, "Updated")
        self.assertEqual(inv.status, InvestigationStatus.ACTIVE)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="investigation.updated",
                target_id=str(inv.id),
            ).exists()
        )


class InvestigationViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="inv-owner2@example.com", password="x")
        self.stranger = User.objects.create_user(email="inv-stranger2@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Inv Workspace2",
            slug="inv-workspace2",
            description="desc",
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Q1",
            question_text="What is happening?",
            scope_json={"dataset": "all"},
            method_json={"model": "x"},
            status=InvestigationStatus.DRAFT,
        )

    def test_owner_can_create_investigation_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("investigation-list", kwargs={"workspace_id": self.workspace.id}),
            data={
                "title": "Created by view",
                "question_text": "A question",
                "scope_json": '{"filters": []}',
                "method_json": '{"method": "filter"}',
                "status": InvestigationStatus.DRAFT,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            Investigation.objects.filter(
                workspace=self.workspace, title="Created by view"
            ).exists()
        )

    def test_stranger_cannot_view_private_workspace_investigations(self):
        self.client.force_login(self.stranger)
        response = self.client.get(
            reverse("investigation-list", kwargs={"workspace_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_public_workspace_investigation_list_viewable_without_login(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])
        response = self.client.get(
            reverse("investigation-list", kwargs={"workspace_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 200)

    def test_bot_investigation_detail_view_does_not_update_last_viewed(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])

        self.client.get(
            reverse(
                "investigation-detail",
                kwargs={
                    "workspace_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            HTTP_USER_AGENT="Googlebot/2.1",
        )
        self.workspace.refresh_from_db()
        self.investigation.refresh_from_db()
        self.assertIsNone(self.workspace.last_viewed_at)
        self.assertIsNone(self.investigation.last_viewed_at)

    def test_human_investigation_detail_view_updates_last_viewed(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])

        self.client.get(
            reverse(
                "investigation-detail",
                kwargs={
                    "workspace_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            HTTP_USER_AGENT="Mozilla/5.0",
        )
        self.workspace.refresh_from_db()
        self.investigation.refresh_from_db()
        self.assertIsNotNone(self.workspace.last_viewed_at)
        self.assertIsNotNone(self.investigation.last_viewed_at)
