import io
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import TestCase, override_settings
from django.urls import reverse

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_notifications.models import NotificationRequest, NotificationTrigger
from wb_workspaces.models import MembershipAccessMode, MembershipRole, WorkspaceMembership
from wb_workspaces.services import create_workspace_for_user

from .artifact_storage import StoredArtifactFile
from .models import ArtifactStatus, ArtifactStorageBackend, ArtifactType, RunArtifact, RunStatus, RunType
from .services import queue_run, request_run_cancellation, set_run_status
from .worker import process_single_available_run


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

    def test_owner_can_queue_run_with_completion_notification(self):
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
                "run_type": RunType.EXTRACT,
                "input_config_json": '{"extract": true}',
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
                "run-artifact-download",
                kwargs={
                    "workspace_id": self.workspace.id,
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
                "run-artifact-download",
                kwargs={
                    "workspace_id": self.workspace.id,
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
                "run-artifact-download",
                kwargs={
                    "workspace_id": self.workspace.id,
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
                    "run-artifact-download",
                    kwargs={
                        "workspace_id": self.workspace.id,
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
