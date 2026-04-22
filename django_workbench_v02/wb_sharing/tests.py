from datetime import timedelta
from pathlib import Path
import tempfile

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_investigations.models import Investigation, InvestigationStatus
from wb_runs.models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunArtifact,
    RunEvent,
    RunEventType,
    RunStatus,
    RunType,
)
from wb_workspaces.models import (
    MembershipAccessMode,
    MembershipRole,
    WorkspaceMembership,
    WorkspaceRevision,
    WorkspaceVisibility,
)
from wb_workspaces.services import create_workspace_for_user

from .models import ShareMode, WorkspaceShareLink
from .services import (
    copy_share_link_to_workbook,
    create_share_link,
    revoke_share_link,
    update_share_link,
)


User = get_user_model()


class ShareLinkServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="share-owner@example.com", password="x")
        self.editor = User.objects.create_user(email="share-editor@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Sharing Workspace",
            slug="sharing-workspace",
            description="Sharing",
        )
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_manage_members=False,
            can_manage_shares=False,
            can_run_workflows=True,
        )

    def test_create_share_link_snapshot_success(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        self.assertEqual(share_link.mode, ShareMode.SNAPSHOT)
        self.assertIsNotNone(share_link.snapshot_revision)
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="sharing.link_created",
                target_id=str(share_link.id),
            ).exists()
        )

    def test_create_share_link_requires_manage_shares_permission(self):
        with self.assertRaises(PermissionDenied):
            create_share_link(
                actor=self.editor,
                workspace=self.workspace,
                mode=ShareMode.SNAPSHOT,
                is_public=True,
            )

    def test_create_snapshot_share_auto_seeds_revision_when_missing(self):
        WorkspaceRevision.objects.filter(workspace=self.workspace).delete()
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        self.assertIsNotNone(share_link.snapshot_revision)

    def test_update_share_and_revoke(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        updated = update_share_link(
            actor=self.owner,
            share_link=share_link,
            mode=ShareMode.LIVE,
            is_public=False,
            is_active=True,
            expires_at=timezone.now() + timedelta(days=7),
        )
        self.assertEqual(updated.mode, ShareMode.LIVE)
        self.assertIsNone(updated.snapshot_revision)
        revoked = revoke_share_link(actor=self.owner, share_link=share_link)
        self.assertFalse(revoked.is_active)
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="sharing.link_revoked",
                target_id=str(share_link.id),
            ).exists()
        )

    def test_copy_share_to_workbook_clones_history_and_sets_private_defaults(self):
        investigation = Investigation.objects.create(
            workspace=self.workspace,
            created_by=self.owner,
            title="Source Investigation",
            question_text="Question",
            scope_json={"scope": "all"},
            method_json={"method": "filter"},
            status=InvestigationStatus.ACTIVE,
        )
        run = InvestigationRun.objects.create(
            investigation=investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.SUCCEEDED,
            input_config_json={"provider": "openai"},
        )
        RunEvent.objects.create(
            run=run,
            event_type=RunEventType.INFO,
            message="Source event",
            payload_json={"k": "v"},
        )
        RunArtifact.objects.create(
            run=run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri="/tmp/source-artifact.csv",
            metadata_json={"origin": "source"},
        )
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )

        copied = copy_share_link_to_workbook(
            actor=self.editor,
            share_link=share_link,
        )

        self.assertEqual(copied.visibility, WorkspaceVisibility.PRIVATE)
        self.assertFalse(copied.is_listed)
        self.assertTrue(copied.title.endswith("(Copy)"))
        copied_investigation = Investigation.objects.get(workspace=copied)
        self.assertEqual(copied_investigation.title, investigation.title)
        copied_run = InvestigationRun.objects.get(workspace=copied, investigation=copied_investigation)
        self.assertEqual(copied_run.status, run.status)
        self.assertEqual(copied_run.run_type, run.run_type)
        self.assertEqual(copied_run.input_config_json, run.input_config_json)
        self.assertTrue(RunEvent.objects.filter(run=copied_run, message="Source event").exists())
        self.assertTrue(
            RunArtifact.objects.filter(
                run=copied_run,
                workspace=copied,
                storage_uri="/tmp/source-artifact.csv",
            ).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=copied,
                action_type="sharing.workbook_copied",
            ).exists()
        )


class ShareLinkViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="share-owner2@example.com", password="x")
        self.viewer = User.objects.create_user(email="share-viewer2@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Sharing Workspace 2",
            slug="sharing-workspace-2",
            description="Sharing",
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

    def test_owner_can_create_share_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-share-create", kwargs={"workbook_id": self.workspace.id}),
            data={"mode": ShareMode.SNAPSHOT, "is_public": "on"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            WorkspaceShareLink.objects.filter(workspace=self.workspace).exists()
        )

    def test_public_share_view_accessible_without_login(self):
        Investigation.objects.create(
            workspace=self.workspace,
            created_by=self.owner,
            title="Shared investigation",
            question_text="What happened?",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        response = self.client.get(
            reverse("share-link-detail", kwargs={"share_id": share_link.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workspace")
        self.assertContains(response, "Reports in view")
        self.assertContains(response, "Interactive report list")
        self.assertContains(response, "Export")
        self.assertNotContains(response, "Open investigation")
        self.assertNotContains(response, '<form method="get" action=')
        self.assertNotContains(response, 'id="explore-filter-form"')
        self.assertNotContains(response, "Troubleshooting and support payload")

    def test_public_share_export_csv_available_without_login(self):
        investigation = Investigation.objects.create(
            workspace=self.workspace,
            created_by=self.owner,
            title="Exportable shared investigation",
            question_text="What happened?",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = InvestigationRun.objects.create(
            investigation=investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.SUCCEEDED,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write("report_id,title\n1,Example report\n")
            artifact_path = Path(tmp.name)
        self.addCleanup(lambda: artifact_path.unlink(missing_ok=True))
        RunArtifact.objects.create(
            run=run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(artifact_path),
            metadata_json={"matched_reports": 1},
        )
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.LIVE,
            is_public=True,
        )
        response = self.client.get(
            reverse("share-link-export-csv", kwargs={"share_id": share_link.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers.get("Content-Disposition", ""))

    def test_private_share_requires_membership_or_login(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=False,
        )
        response = self.client.get(
            reverse("share-link-detail", kwargs={"share_id": share_link.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

        self.client.force_login(self.viewer)
        response_member = self.client.get(
            reverse("share-link-detail", kwargs={"share_id": share_link.id})
        )
        self.assertEqual(response_member.status_code, 200)

    def test_bot_share_view_does_not_update_last_viewed(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        self.client.get(
            reverse("share-link-detail", kwargs={"share_id": share_link.id}),
            HTTP_USER_AGENT="Googlebot/2.1",
        )
        share_link.refresh_from_db()
        self.workspace.refresh_from_db()
        self.assertIsNone(share_link.last_viewed_at)
        self.assertIsNone(self.workspace.last_viewed_at)

    def test_human_share_view_updates_last_viewed(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        self.client.get(
            reverse("share-link-detail", kwargs={"share_id": share_link.id}),
            HTTP_USER_AGENT="Mozilla/5.0",
        )
        share_link.refresh_from_db()
        self.workspace.refresh_from_db()
        self.assertIsNotNone(share_link.last_viewed_at)
        self.assertIsNotNone(self.workspace.last_viewed_at)

    def test_anonymous_user_copy_redirects_to_login(self):
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        response = self.client.post(
            reverse("share-link-copy", kwargs={"share_id": share_link.id}),
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_authenticated_user_can_create_editable_copy_from_share(self):
        Investigation.objects.create(
            workspace=self.workspace,
            created_by=self.owner,
            title="Source Investigation 2",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        share_link = create_share_link(
            actor=self.owner,
            workspace=self.workspace,
            mode=ShareMode.SNAPSHOT,
            is_public=True,
        )
        self.client.force_login(self.viewer)
        response = self.client.post(
            reverse("share-link-copy", kwargs={"share_id": share_link.id}),
        )
        self.assertEqual(response.status_code, 302)
        copied_workspace = (
            self.viewer.created_workspaces.exclude(id=self.workspace.id).order_by("-created_at").first()
        )
        self.assertIsNotNone(copied_workspace)
        self.assertEqual(copied_workspace.visibility, WorkspaceVisibility.PRIVATE)
        self.assertFalse(copied_workspace.is_listed)
