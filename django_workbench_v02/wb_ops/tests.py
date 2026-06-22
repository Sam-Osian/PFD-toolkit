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
    RunApprovalStatus,
)
from wb_workspaces.models import (
    MembershipAccessMode,
    MembershipRole,
    UserLLMCredential,
    Workspace,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceReportExclusion,
)


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
        self.assertContains(response, "Run review")
        self.assertContains(response, "queued runs before approval")

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

    def test_ops_review_approve_persists_extract_feature_fields(self):
        pending = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.EXTRACT,
            status=RunStatus.QUEUED,
            approval_status=RunApprovalStatus.PENDING,
            requires_approval=True,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.EXTRACT],
                "provider": "local_ollama",
                "model_name": "gemma4:26b",
                "search_query": "example",
                "feature_fields": [{"name": "setting", "description": "Care setting", "type": "text"}],
            },
        )

        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("ops-approval-action", kwargs={"run_id": pending.id}),
            data={
                "action": "approve",
                "title": self.investigation.title,
                "question_text": self.investigation.question_text,
                "scope_option": "all_reports",
                "run_filter": "1",
                "run_extract": "1",
                "search_query": "example",
                "provider": "local_ollama",
                "model_name": "gemma4:26b",
                "max_parallel_workers": "1",
                "feature_field_name": ["setting", "age_at_death"],
                "feature_field_description": ["Care setting", "Age in years"],
                "feature_field_type": ["text", "decimal"],
                "allow_multiple": "1",
                "skip_if_present": "1",
                "approval_note": "looks good",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        pending.refresh_from_db()
        self.assertEqual(pending.approval_status, RunApprovalStatus.APPROVED)
        features = pending.input_config_json.get("feature_fields") or []
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["name"], "setting")
        self.assertEqual(features[1]["name"], "age_at_death")

    def test_ops_approvals_excludes_terminal_runs_with_stale_pending_approval_status(self):
        failed = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.FAILED,
            approval_status=RunApprovalStatus.PENDING,
            requires_approval=True,
            input_config_json={"provider": "local_ollama", "model_name": "gemma4:26b"},
        )
        queued = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.QUEUED,
            approval_status=RunApprovalStatus.PENDING,
            requires_approval=True,
            input_config_json={"provider": "local_ollama", "model_name": "gemma4:26b"},
        )

        self.client.force_login(self.admin)
        response = self.client.get(reverse("ops-approvals"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(queued.id))
        self.assertNotContains(response, str(failed.id))

    def test_staff_can_attach_one_time_openai_key_to_queued_run(self):
        queued_run = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.QUEUED,
            approval_status=RunApprovalStatus.PENDING,
            requires_approval=True,
            input_config_json={"execution_mode": "real", "provider": "local_ollama"},
        )
        self.client.force_login(self.staff)
        response = self.client.post(
            reverse(
                "ops-run-configure",
                kwargs={"workspace_id": self.workspace.id, "run_id": queued_run.id},
            ),
            data={
                "provider": "openai",
                "api_key": "sk-ops-inline-1234",
                "model_name": "gpt-4.1",
                "next_url": reverse("ops-workspace-detail", kwargs={"workspace_id": self.workspace.id}),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        queued_run.refresh_from_db()
        self.assertEqual(queued_run.input_config_json.get("provider"), "openai")
        self.assertEqual(queued_run.input_config_json.get("model_name"), "gpt-4.1")
        self.assertEqual(queued_run.ops_override_provider, "openai")
        self.assertEqual(queued_run.ops_override_key_last4, "1234")
        self.assertFalse(queued_run.requires_approval)
        self.assertEqual(queued_run.approval_status, RunApprovalStatus.NOT_REQUIRED)
        self.assertFalse(
            WorkspaceCredential.objects.filter(
                workspace=self.workspace,
                user=self.owner,
                provider="openai",
            ).exists()
        )

    def test_staff_can_queue_pending_approval_run_on_openai_with_one_time_key(self):
        pending = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.QUEUED,
            approval_status=RunApprovalStatus.PENDING,
            requires_approval=True,
            input_config_json={
                "pipeline_plan": [RunType.FILTER],
                "provider": "local_ollama",
                "model_name": "gemma4:26b",
                "search_query": "example",
                "execution_mode": "real",
            },
        )

        self.client.force_login(self.staff)
        response = self.client.post(
            reverse("ops-approval-action", kwargs={"run_id": pending.id}),
            data={
                "action": "approve",
                "title": self.investigation.title,
                "question_text": self.investigation.question_text,
                "scope_option": "all_reports",
                "run_filter": "1",
                "search_query": "example",
                "provider": "openai",
                "api_key": "sk-openai-ops-5678",
                "model_name": "gpt-4.1",
                "max_parallel_workers": "1",
                "approval_note": "switch to OpenAI",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        pending.refresh_from_db()
        self.assertEqual(pending.input_config_json.get("provider"), "openai")
        self.assertEqual(pending.input_config_json.get("model_name"), "gpt-4.1")
        self.assertEqual(pending.ops_override_provider, "openai")
        self.assertEqual(pending.ops_override_key_last4, "5678")
        self.assertFalse(pending.requires_approval)
        self.assertEqual(pending.approval_status, RunApprovalStatus.NOT_REQUIRED)
        self.assertFalse(
            UserLLMCredential.objects.filter(
                user=self.owner,
                provider="openai",
            ).exists()
        )

    def test_staff_can_switch_run_back_to_local_without_openai_model_leaking(self):
        queued_run = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.QUEUED,
            approval_status=RunApprovalStatus.NOT_REQUIRED,
            requires_approval=False,
            input_config_json={
                "execution_mode": "real",
                "provider": "openai",
                "model_name": "gpt-4.1",
            },
        )
        self.client.force_login(self.staff)
        response = self.client.post(
            reverse(
                "ops-run-configure",
                kwargs={"workspace_id": self.workspace.id, "run_id": queued_run.id},
            ),
            data={
                "provider": "local_ollama",
                "model_name": "gpt-4.1",
                "next_url": reverse("ops-workspace-detail", kwargs={"workspace_id": self.workspace.id}),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        queued_run.refresh_from_db()
        self.assertEqual(queued_run.input_config_json.get("provider"), "local_ollama")
        self.assertEqual(queued_run.input_config_json.get("model_name"), "gemma4:26b")
        self.assertTrue(queued_run.requires_approval)
        self.assertEqual(queued_run.approval_status, RunApprovalStatus.PENDING)
