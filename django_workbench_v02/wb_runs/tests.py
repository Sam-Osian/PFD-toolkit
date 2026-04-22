import io
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_notifications.models import NotificationRequest, NotificationTrigger
from wb_workspaces.models import (
    MembershipAccessMode,
    MembershipRole,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceReportExclusion,
)
from wb_workspaces.services import create_workspace_for_user

from .artifact_storage import StoredArtifactFile
from .models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    RunArtifact,
    RunStatus,
    RunType,
    RunWorkerHeartbeat,
)
from .services import queue_run, request_run_cancellation, set_run_status
from .worker import process_single_available_run, reconcile_timed_out_runs


User = get_user_model()


class RunServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="run-owner@example.com", password="x")
        self.viewer = User.objects.create_user(email="run-viewer@example.com", password="x")
        self.request_factory = RequestFactory()
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

    def test_invalid_status_transition_is_rejected(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={},
        )
        with self.assertRaises(ValidationError):
            set_run_status(
                run=run,
                status=RunStatus.SUCCEEDED,
                message="Invalid direct success from queued",
            )

    def test_queue_run_backfills_scope_from_investigation_and_enforces_exclusions(self):
        self.investigation.scope_json = {
            "collection_slug": "local-gov",
            "collection_query": "medication safety",
            "selected_filters": {"coroner": ["A"], "area": ["B"], "receiver": ["C"]},
            "report_identity_allowlist": ["https://example.com/r1"],
        }
        self.investigation.save(update_fields=["scope_json", "updated_at"])
        WorkspaceReportExclusion.objects.create(
            workspace=self.workspace,
            report_identity="https://example.com/excluded-1",
            reason="Out of scope",
        )

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        self.assertEqual(run.input_config_json.get("collection_slug"), "local-gov")
        self.assertEqual(run.input_config_json.get("collection_query"), "medication safety")
        self.assertEqual(
            run.input_config_json.get("selected_filters"),
            {"coroner": ["A"], "area": ["B"], "receiver": ["C"]},
        )
        self.assertEqual(
            run.input_config_json.get("report_identity_allowlist"),
            ["https://example.com/r1"],
        )
        self.assertEqual(
            run.input_config_json.get("excluded_report_identities"),
            ["https://example.com/excluded-1"],
        )
        self.assertEqual(run.input_config_json.get("excluded_report_count"), 1)

    def test_queue_run_respects_explicit_scope_over_investigation_scope(self):
        self.investigation.scope_json = {
            "collection_slug": "local-gov",
            "collection_query": "scope query",
            "selected_filters": {"coroner": ["Scope"], "area": [], "receiver": []},
            "report_identity_allowlist": ["https://example.com/scope"],
        }
        self.investigation.save(update_fields=["scope_json", "updated_at"])
        WorkspaceReportExclusion.objects.create(
            workspace=self.workspace,
            report_identity="https://example.com/excluded-2",
            reason="Out of scope",
        )

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "collection_slug": "custom-search",
                "collection_query": "explicit query",
                "selected_filters": {"coroner": ["Explicit"], "area": [], "receiver": []},
                "report_identity_allowlist": ["https://example.com/explicit"],
                "excluded_report_identities": ["https://example.com/ignored-manual-value"],
            },
        )
        self.assertEqual(run.input_config_json.get("collection_slug"), "custom-search")
        self.assertEqual(run.input_config_json.get("collection_query"), "explicit query")
        self.assertEqual(
            run.input_config_json.get("selected_filters"),
            {"coroner": ["Explicit"], "area": [], "receiver": []},
        )
        self.assertEqual(
            run.input_config_json.get("report_identity_allowlist"),
            ["https://example.com/explicit"],
        )
        self.assertEqual(
            run.input_config_json.get("excluded_report_identities"),
            ["https://example.com/excluded-2"],
        )

    @override_settings(MAX_RUNS_PER_USER_PER_DAY=1)
    def test_queue_run_enforces_user_daily_limit(self):
        queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        with self.assertRaisesMessage(ValidationError, "daily run cap"):
            queue_run(
                actor=self.owner,
                investigation=self.investigation,
                run_type=RunType.FILTER,
                input_config_json={"execution_mode": "simulate"},
            )

    @override_settings(MAX_RUNS_PER_WORKBOOK_PER_DAY=1)
    def test_queue_run_enforces_workbook_daily_limit(self):
        queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        another_user = User.objects.create_user(email="run-owner-2@example.com", password="x")
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=another_user,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_manage_members=False,
            can_manage_shares=False,
            can_run_workflows=True,
        )
        with self.assertRaisesMessage(ValidationError, "workbook reached its daily run cap"):
            queue_run(
                actor=another_user,
                investigation=self.investigation,
                run_type=RunType.THEMES,
                input_config_json={"execution_mode": "simulate"},
            )

    @override_settings(MAX_CONCURRENT_RUNS_PER_USER=1)
    def test_queue_run_enforces_user_concurrent_limit(self):
        active_run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        active_run.status = RunStatus.RUNNING
        active_run.started_at = timezone.now()
        active_run.save(update_fields=["status", "started_at", "updated_at"])
        with self.assertRaisesMessage(ValidationError, "concurrent run cap"):
            queue_run(
                actor=self.owner,
                investigation=self.investigation,
                run_type=RunType.EXTRACT,
                input_config_json={"execution_mode": "simulate"},
            )

    @override_settings(MAX_CONCURRENT_RUNS_GLOBAL=1)
    def test_queue_run_enforces_global_concurrent_limit(self):
        first_run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        first_run.status = RunStatus.RUNNING
        first_run.started_at = timezone.now()
        first_run.save(update_fields=["status", "started_at", "updated_at"])
        second_workspace = create_workspace_for_user(
            user=self.viewer,
            title="Global Limit Workspace",
            slug="global-limit-workspace",
            description="desc",
        )
        second_investigation = create_investigation(
            actor=self.viewer,
            workspace=second_workspace,
            title="Second investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        with self.assertRaisesMessage(ValidationError, "global concurrency capacity"):
            queue_run(
                actor=self.viewer,
                investigation=second_investigation,
                run_type=RunType.FILTER,
                input_config_json={"execution_mode": "simulate"},
            )

    @override_settings(RUN_LAUNCH_RATE_LIMIT_USER_PER_MINUTE=1)
    def test_queue_run_enforces_user_rate_limit(self):
        request = self.request_factory.post("/fake")
        request.user = self.owner
        request.META["REMOTE_ADDR"] = "203.0.113.10"
        queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
            request=request,
        )
        with self.assertRaisesMessage(ValidationError, "Rate limit reached for run launches"):
            queue_run(
                actor=self.owner,
                investigation=self.investigation,
                run_type=RunType.THEMES,
                input_config_json={"execution_mode": "simulate"},
                request=request,
            )


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
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.EXPORT,
                "input_config_json": '{"export": true, "execution_mode": "simulate"}',
                "query_start_date": "",
                "query_end_date": "",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            self.investigation.runs.filter(run_type=RunType.EXPORT).exists()
        )

    def test_queue_real_run_requires_saved_or_submitted_key(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.FILTER,
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "api_key": "",
                "input_config_json": '{"execution_mode": "real"}',
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No saved openai API key for this workbook.")
        self.assertFalse(
            self.investigation.runs.filter(
                run_type=RunType.FILTER,
                input_config_json__execution_mode="real",
            ).exists()
        )

    def test_queue_real_run_saves_workspace_credential(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.FILTER,
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "api_key": "sk-test-secret-1234",
                "save_api_key": "on",
                "input_config_json": '{"execution_mode": "real"}',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            WorkspaceCredential.objects.filter(
                workspace=self.workspace,
                user=self.owner,
                provider="openai",
                key_last4="1234",
            ).exists()
        )

    def test_queue_run_includes_scope_and_excluded_reports(self):
        self.investigation.scope_json = {
            "collection_slug": "custom-search",
            "collection_query": "medication",
            "selected_filters": {"coroner": ["A"], "area": [], "receiver": []},
            "report_identity_allowlist": ["https://example.com/r1"],
        }
        self.investigation.save(update_fields=["scope_json", "updated_at"])
        WorkspaceReportExclusion.objects.create(
            workspace=self.workspace,
            report_identity="https://example.com/r2",
            reason="Out of scope",
        )

        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.FILTER,
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "api_key": "sk-test-secret-1234",
                "save_api_key": "on",
                "input_config_json": '{"execution_mode": "real"}',
            },
        )
        self.assertEqual(response.status_code, 302)
        run = self.investigation.runs.latest("created_at")
        self.assertEqual(run.input_config_json.get("collection_slug"), "custom-search")
        self.assertEqual(run.input_config_json.get("collection_query"), "medication")
        self.assertEqual(
            run.input_config_json.get("report_identity_allowlist"),
            ["https://example.com/r1"],
        )
        self.assertEqual(
            run.input_config_json.get("excluded_report_identities"),
            ["https://example.com/r2"],
        )

    def test_owner_can_queue_run_with_completion_notification(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.EXTRACT,
                "input_config_json": '{"extract": true, "execution_mode": "simulate"}',
                "query_start_date": "",
                "query_end_date": "",
                "request_completion_email": "on",
                "notify_on": NotificationTrigger.ANY,
            },
        )
        self.assertEqual(response.status_code, 302)
        created_run = self.investigation.runs.filter(run_type=RunType.EXTRACT).latest("created_at")
        self.assertTrue(
            NotificationRequest.objects.filter(
                run=created_run,
                user=self.owner,
                notify_on=NotificationTrigger.ANY,
            ).exists()
        )

    def test_viewer_cannot_queue_run_via_view(self):
        self.client.force_login(self.viewer)
        response = self.client.post(
            reverse(
                "workbook-run-queue",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            data={
                "run_type": RunType.EXPORT,
                "input_config_json": '{"export": true, "execution_mode": "simulate"}',
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
                "workbook-run-detail",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_run_detail_includes_journey_outcome_and_config_summary(self):
        self.run.input_config_json = {
            "execution_mode": "real",
            "provider": "openai",
            "model_name": "gpt-4.1-mini",
            "pipeline_plan": ["filter", "themes", "extract"],
            "pipeline_continue_on_fail": True,
            "custom_flag": "x",
        }
        self.run.save(update_fields=["input_config_json", "updated_at"])
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-detail",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run Journey")
        self.assertContains(response, "Outcome")
        self.assertContains(response, "Configuration Snapshot")
        self.assertContains(response, "Raw configuration JSON")
        self.assertContains(response, "Pipeline plan")
        self.assertContains(response, "filter, themes, extract")
        self.assertContains(response, "Custom flag")

    def test_terminal_run_hides_cancellation_form(self):
        self.run.status = RunStatus.SUCCEEDED
        self.run.started_at = timezone.now()
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "started_at", "finished_at", "updated_at"])
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-detail",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run completed")
        self.assertNotContains(response, "<button type=\"submit\">Cancel run</button>")

    def test_cancelling_run_shows_cancellation_banner(self):
        self.run.status = RunStatus.CANCELLING
        self.run.cancel_requested_at = timezone.now()
        self.run.save(update_fields=["status", "cancel_requested_at", "updated_at"])
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-detail",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cancellation has been requested")
        self.assertContains(response, "Cancellation is not available in the current run status")

    def test_artifacts_are_grouped_with_intent_labels(self):
        output_path = Path("/tmp/test-run-detail-grouped-artifacts.csv")
        output_path.write_text("id,value\n1,ok\n", encoding="utf-8")
        RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(output_path),
            metadata_json={},
        )
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-detail",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Filtered dataset")
        self.assertContains(response, "Matched report subset after filtering.")

    def test_owner_can_cancel_run_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse(
                "workbook-run-cancel",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            ),
            data={"cancel_reason": "Stop now"},
        )
        self.assertEqual(response.status_code, 302)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, RunStatus.CANCELLING)

    def test_owner_can_download_run_artifact(self):
        output_path = Path("/tmp/test-run-detail-download.csv")
        output_path.write_text("id,value\n1,ok\n", encoding="utf-8")
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(output_path),
            metadata_json={},
        )

        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-artifact-download",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "run_id": self.run.id,
                    "artifact_id": artifact.id,
                },
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers.get("Content-Disposition", ""))
        artifact.refresh_from_db()
        self.assertIsNotNone(artifact.last_viewed_at)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.artifact_downloaded",
                target_id=str(artifact.id),
            ).exists()
        )

    def test_bot_user_agent_does_not_update_artifact_last_viewed(self):
        output_path = Path("/tmp/test-run-detail-bot-download.csv")
        output_path.write_text("id,value\n1,ok\n", encoding="utf-8")
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(output_path),
            metadata_json={},
        )

        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-run-artifact-download",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "run_id": self.run.id,
                    "artifact_id": artifact.id,
                },
            ),
            HTTP_USER_AGENT="Googlebot/2.1",
        )
        self.assertEqual(response.status_code, 200)
        artifact.refresh_from_db()
        self.assertIsNone(artifact.last_viewed_at)

    def test_authenticated_non_member_cannot_download_private_workspace_artifact(self):
        output_path = Path("/tmp/test-run-detail-no-access.csv")
        output_path.write_text("id,value\n1,ok\n", encoding="utf-8")
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(output_path),
            metadata_json={},
        )
        stranger = User.objects.create_user(email="artifact-stranger@example.com", password="x")
        self.client.force_login(stranger)
        response = self.client.get(
            reverse(
                "workbook-run-artifact-download",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "run_id": self.run.id,
                    "artifact_id": artifact.id,
                },
            )
        )
        self.assertEqual(response.status_code, 403)

    def test_owner_can_download_object_storage_artifact(self):
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.OBJECT_STORAGE,
            storage_uri="s3://fake-bucket/path/to/file.csv",
            metadata_json={},
            size_bytes=7,
        )
        self.client.force_login(self.owner)
        with patch(
            "wb_runs.views.open_artifact_for_download",
            return_value=(io.BytesIO(b"id,x\n1,2"), "file.csv"),
        ) as mocked:
            response = self.client.get(
                reverse(
                    "workbook-run-artifact-download",
                    kwargs={
                        "workbook_id": self.workspace.id,
                        "run_id": self.run.id,
                        "artifact_id": artifact.id,
                    },
                )
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers.get("Content-Disposition", ""))
        self.assertEqual(response.headers.get("Content-Length"), "7")
        mocked.assert_called_once()


class RunWorkerTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="worker-owner@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Worker Workspace",
            slug="worker-workspace",
            description="desc",
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Worker Investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )

    def test_worker_processes_queued_run_to_success(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        processed = process_single_available_run(worker_id="test-worker")
        self.assertIsNotNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertEqual(run.worker_id, "test-worker")
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.progress_percent, 100)
        self.assertTrue(run.artifacts.exists())

    def test_worker_honors_pre_requested_cancellation(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        request_run_cancellation(actor=self.owner, run=run, reason="cancel early")
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.CANCELLED)

    def test_worker_records_failure(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "simulate_failure": True,
                "simulate_failure_stage": 1,
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.error_code, "SIMULATED_FAILURE")
        self.assertTrue(run.error_message)

    def test_worker_records_timeout(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "simulate_timeout": True,
                "simulate_timeout_stage": 1,
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.TIMED_OUT)
        self.assertEqual(run.error_code, "SIMULATED_TIMEOUT")

    def test_worker_uses_real_filter_adapter_when_enabled(self):
        output_path = Path("/tmp/test-run-filter-output.csv")
        output_path.write_text("id,matches_query\n1,True\n", encoding="utf-8")

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            return_value={
                "output_path": str(output_path),
                "total_reports": 10,
                "matched_reports": 3,
                "output_reports": 3,
                "search_query": "medication safety",
                "filter_df": True,
                "produce_spans": False,
                "drop_spans": False,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "report_limit": None,
            },
        ) as mocked:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        artifact = run.artifacts.latest("created_at")
        self.assertEqual(artifact.storage_backend, ArtifactStorageBackend.FILE)
        self.assertEqual(artifact.storage_uri, str(output_path))
        self.assertEqual(artifact.status, "ready")
        mocked.assert_called_once()

    def test_worker_uses_real_themes_adapter_when_enabled(self):
        summary_path = Path("/tmp/test-run-theme-summary.csv")
        assignments_path = Path("/tmp/test-run-theme-assignments.csv")
        schema_path = Path("/tmp/test-run-theme-schema.json")
        summaries_path = Path("/tmp/test-run-report-summaries.csv")
        summary_path.write_text("theme,matched_reports\ncare_coordination,5\n", encoding="utf-8")
        assignments_path.write_text("id,care_coordination\n1,True\n", encoding="utf-8")
        schema_path.write_text('{"title":"ThemeModel"}\n', encoding="utf-8")
        summaries_path.write_text("id,summary\n1,summary\n", encoding="utf-8")

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.THEMES,
            input_config_json={"execution_mode": "real"},
        )

        with patch(
            "wb_runs.worker.execute_themes_workflow",
            return_value={
                "output_path": str(summary_path),
                "total_reports": 12,
                "discovered_themes": 4,
                "theme_summary_path": str(summary_path),
                "theme_assignments_path": str(assignments_path),
                "theme_schema_path": str(schema_path),
                "report_summaries_path": str(summaries_path),
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "report_limit": None,
            },
        ) as mocked:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertIn("Discovered 4 themes", run.events.latest("created_at").message)
        self.assertTrue(
            run.artifacts.filter(
                artifact_type=ArtifactType.THEME_SUMMARY,
                storage_uri=str(summary_path),
                storage_backend=ArtifactStorageBackend.FILE,
            ).exists()
        )
        self.assertTrue(
            run.artifacts.filter(
                artifact_type=ArtifactType.THEME_ASSIGNMENTS,
                storage_uri=str(assignments_path),
                storage_backend=ArtifactStorageBackend.FILE,
            ).exists()
        )
        mocked.assert_called_once()

    def test_worker_uses_real_extract_adapter_when_enabled(self):
        output_path = Path("/tmp/test-run-extract-output.csv")
        feature_schema_path = Path("/tmp/test-run-extract-schema.json")
        output_path.write_text("id,feature_a\n1,yes\n", encoding="utf-8")
        feature_schema_path.write_text('{"title":"RunExtractFeatures"}\n', encoding="utf-8")

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.EXTRACT,
            input_config_json={"execution_mode": "real"},
        )

        with patch(
            "wb_runs.worker.execute_extract_workflow",
            return_value={
                "output_path": str(output_path),
                "feature_schema_path": str(feature_schema_path),
                "total_reports": 20,
                "output_reports": 20,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "report_limit": None,
                "produce_spans": False,
                "drop_spans": False,
                "force_assign": False,
                "allow_multiple": False,
                "skip_if_present": True,
            },
        ) as mocked:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertTrue(
            run.artifacts.filter(
                artifact_type=ArtifactType.EXTRACTION_TABLE,
                storage_uri=str(output_path),
                storage_backend=ArtifactStorageBackend.FILE,
            ).exists()
        )
        mocked.assert_called_once()

    def test_worker_uses_real_export_adapter_when_enabled(self):
        output_path = Path("/tmp/test-run-export-output.zip")
        output_path.write_bytes(b"PK\x03\x04")

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.EXPORT,
            input_config_json={"execution_mode": "real"},
        )

        with patch(
            "wb_runs.worker.execute_export_workflow",
            return_value={
                "output_path": str(output_path),
                "bundle_name": "bundle.zip",
                "selected_artifacts": 3,
                "included_files": 2,
                "skipped_artifacts": 1,
            },
        ) as mocked:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertTrue(
            run.artifacts.filter(
                artifact_type=ArtifactType.BUNDLE_EXPORT,
                storage_uri=str(output_path),
                storage_backend=ArtifactStorageBackend.FILE,
            ).exists()
        )
        self.assertIn("Packaged 2 files", run.events.latest("created_at").message)
        mocked.assert_called_once()

    def test_pipeline_queues_next_stage_after_success(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "pipeline_plan": [RunType.FILTER, RunType.THEMES, RunType.EXTRACT],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)

        next_run = (
            self.investigation.runs.filter(run_type=RunType.THEMES)
            .exclude(id=run.id)
            .first()
        )
        self.assertIsNotNone(next_run)
        self.assertEqual(next_run.status, RunStatus.QUEUED)
        self.assertEqual(next_run.input_config_json.get("pipeline_index"), 1)
        self.assertTrue(next_run.input_config_json.get("pipeline_require_upstream_artifact"))
        self.assertTrue(bool(next_run.input_config_json.get("input_artifact_id")))

    def test_pipeline_continues_on_failure_when_enabled(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "simulate_failure": True,
                "simulate_failure_stage": 1,
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertTrue(
            self.investigation.runs.filter(run_type=RunType.THEMES).exclude(id=run.id).exists()
        )

    def test_pipeline_marks_continued_after_failed_upstream_in_queue_event(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "simulate_failure": True,
                "simulate_failure_stage": 1,
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        process_single_available_run(worker_id="test-worker")

        next_run = (
            self.investigation.runs.filter(run_type=RunType.THEMES)
            .exclude(id=run.id)
            .first()
        )
        self.assertIsNotNone(next_run)
        queue_event = next_run.events.latest("created_at")
        payload = queue_event.payload_json if isinstance(queue_event.payload_json, dict) else {}
        self.assertTrue(payload.get("continued_after_failed_upstream"))

    def test_pipeline_queues_export_after_extract(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.EXTRACT,
            input_config_json={
                "execution_mode": "simulate",
                "feature_fields": [{"name": "setting", "description": "Care setting", "type": "text"}],
                "pipeline_plan": [RunType.EXTRACT, RunType.EXPORT],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertTrue(
            self.investigation.runs.filter(run_type=RunType.EXPORT).exclude(id=run.id).exists()
        )

    def test_pipeline_does_not_continue_after_cancellation(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        request_run_cancellation(actor=self.owner, run=run, reason="stop pipeline")
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.CANCELLED)
        self.assertFalse(
            self.investigation.runs.filter(run_type=RunType.THEMES).exclude(id=run.id).exists()
        )

    def test_missing_required_upstream_artifact_uses_explicit_error_code(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.THEMES,
            input_config_json={
                "execution_mode": "real",
                "pipeline_require_upstream_artifact": True,
                "input_artifact_id": "",
            },
        )
        process_single_available_run(worker_id="test-worker")
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.error_code, "MISSING_UPSTREAM_ARTIFACT")
        self.assertIn("upstream artifact", (run.error_message or "").lower())

    @override_settings(ARTIFACT_STORAGE_BACKEND="object_storage")
    def test_worker_persists_output_using_object_storage_backend(self):
        output_path = Path("/tmp/test-run-object-storage-output.csv")
        output_path.write_text("id,matches_query\n1,True\n", encoding="utf-8")

        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            return_value={
                "output_path": str(output_path),
                "total_reports": 10,
                "matched_reports": 3,
                "output_reports": 3,
                "search_query": "medication safety",
                "filter_df": True,
                "produce_spans": False,
                "drop_spans": False,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "report_limit": None,
            },
        ), patch(
            "wb_runs.worker.store_artifact_file",
            return_value=StoredArtifactFile(
                storage_backend=ArtifactStorageBackend.OBJECT_STORAGE,
                storage_uri="s3://fake-bucket/path/filter.csv",
                size_bytes=123,
            ),
        ) as mocked_store:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        artifact = run.artifacts.latest("created_at")
        self.assertEqual(artifact.storage_backend, ArtifactStorageBackend.OBJECT_STORAGE)
        self.assertEqual(artifact.storage_uri, "s3://fake-bucket/path/filter.csv")
        self.assertEqual(artifact.size_bytes, 123)
        mocked_store.assert_called_once()

    @override_settings(
        RUN_RETRY_ENABLED=True,
        RUN_RETRY_MAX_ATTEMPTS=3,
        RUN_RETRY_BACKOFF_SECONDS=(1, 1, 1),
        RUN_RETRY_JITTER_PCT=0,
    )
    def test_transient_real_adapter_failure_requeues_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            side_effect=TimeoutError("upstream timeout"),
        ):
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(run.input_config_json.get("_retry_attempt"), 1)
        self.assertGreater(run.queued_at, timezone.now() - timedelta(seconds=1))
        self.assertFalse(run.artifacts.exists())

    @override_settings(
        RUN_RETRY_ENABLED=True,
        RUN_RETRY_MAX_ATTEMPTS=2,
        RUN_RETRY_BACKOFF_SECONDS=(0, 0),
        RUN_RETRY_JITTER_PCT=0,
    )
    def test_transient_failure_exhausts_retries_then_fails(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            side_effect=TimeoutError("upstream timeout"),
        ):
            process_single_available_run(worker_id="test-worker")
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.error_code, "FILTER_EXECUTION_ERROR")

    @override_settings(RUN_TOTAL_TIMEOUT_SECONDS=1)
    def test_reconcile_timed_out_runs_marks_stale_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "simulate"},
        )
        stale_time = timezone.now() - timedelta(seconds=5)
        run.status = RunStatus.RUNNING
        run.started_at = stale_time
        run.save(update_fields=["status", "started_at", "updated_at"])

        count = reconcile_timed_out_runs(worker_id="test-worker")
        self.assertEqual(count, 1)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.TIMED_OUT)
        self.assertEqual(run.error_code, "RUN_TOTAL_TIMEOUT")

    def test_worker_records_heartbeat_when_idle(self):
        processed = process_single_available_run(worker_id="heartbeat-worker")
        self.assertIsNone(processed)
        heartbeat = RunWorkerHeartbeat.objects.get(worker_id="heartbeat-worker")
        self.assertEqual(heartbeat.state, "idle")
        self.assertIsNone(heartbeat.last_run)

    @override_settings(WORKER_HEARTBEAT_STALE_SECONDS=120)
    def test_worker_healthcheck_passes_when_recent_heartbeat_exists(self):
        RunWorkerHeartbeat.objects.create(
            worker_id="healthy-worker",
            state="idle",
            last_seen_at=timezone.now(),
        )
        call_command("check_run_worker_health", worker_id="healthy-worker")

    @override_settings(WORKER_HEARTBEAT_STALE_SECONDS=60)
    def test_worker_healthcheck_fails_when_heartbeat_stale(self):
        RunWorkerHeartbeat.objects.create(
            worker_id="stale-worker",
            state="idle",
            last_seen_at=timezone.now() - timedelta(seconds=500),
        )
        with self.assertRaises(CommandError):
            call_command("check_run_worker_health", worker_id="stale-worker")
