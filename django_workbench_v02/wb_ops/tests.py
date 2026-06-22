from __future__ import annotations

from datetime import timedelta
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_investigations.models import Investigation
from wb_runs.models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunStatus,
    RunType,
    RunWorkerHeartbeat,
)
from wb_workspaces.models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership, WorkspaceReportExclusion


User = get_user_model()


class OpsInterfaceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-ops@example.com", password="x")
        self.staff = User.objects.create_user(email="staff-ops@example.com", password="x", is_staff=True)
        self.admin = User.objects.create_superuser(email="admin-ops@example.com", password="x")
        self.workspace = Workspace.objects.create(
            created_by=self.owner,
            title="Ops Workspace",
            slug="ops-workspace",
        )
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=self.owner,
            role=MembershipRole.OWNER,
            access_mode=MembershipAccessMode.EDIT,
            can_manage_members=True,
            can_manage_shares=True,
            can_run_workflows=True,
        )
        self.investigation = Investigation.objects.create(
            workspace=self.workspace,
            created_by=self.owner,
            title="Ops Investigation",
            question_text="Test query",
            scope_json={"query_start_date": "2024-01-01"},
            method_json={"pipeline_plan": [RunType.FILTER]},
        )
        self.run = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.SUCCEEDED,
            input_config_json={
                "pipeline_plan": [RunType.FILTER],
                "search_query": "find delayed escalation",
            },
            started_at=timezone.now() - timedelta(minutes=5),
            finished_at=timezone.now() - timedelta(minutes=2),
        )
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(
            "id,title,date,area,receiver,url\n"
            "123,Sample report,2025-01-02,North,Trust A,https://example.com/report/123\n"
        )
        temp_file.flush()
        temp_file.close()
        self.artifact_path = Path(temp_file.name)
        self.addCleanup(self.artifact_path.unlink, missing_ok=True)
        self.run.artifacts.create(
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(self.artifact_path),
        )
        RunWorkerHeartbeat.objects.create(worker_id="ops-worker-1", state="idle")

    def test_staff_can_access_ops_dashboard(self):
        self.client.force_login(self.staff)
        response = self.client.get(reverse("ops-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Operations dashboard")
        self.assertContains(response, "Recent jobs")

    def test_non_staff_redirected_to_admin_login(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse("ops-dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])

    def test_superuser_can_exclude_row_via_ops_workspace(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("ops-workspace-exclude-row", kwargs={"workspace_id": self.workspace.id}),
            {
                "report_identity": "https://example.com/report/123",
                "report_title": "Sample report",
                "report_date": "2025-01-02",
                "report_url": "https://example.com/report/123",
                "reason": "False positive",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            WorkspaceReportExclusion.objects.filter(
                workspace=self.workspace,
                report_identity="https://example.com/report/123",
            ).exists()
        )

    def test_staff_without_workspace_permissions_cannot_exclude_row(self):
        self.client.force_login(self.staff)
        response = self.client.post(
            reverse("ops-workspace-exclude-row", kwargs={"workspace_id": self.workspace.id}),
            {
                "report_identity": "https://example.com/report/123",
                "reason": "False positive",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_workspace_detail_exposes_dataset_and_pipeline_json(self):
        self.client.force_login(self.admin)
        response = self.client.get(
            reverse("ops-workspace-detail", kwargs={"workspace_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dataset moderation")
        self.assertContains(response, "Sample report")
        self.assertContains(response, "find delayed escalation")
