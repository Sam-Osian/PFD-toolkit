import io
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.core.exceptions import PermissionDenied, ValidationError
from django.core import mail
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from pfd_toolkit.llm import GenerationCancelledError
from pydantic import Field, create_model

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_notifications.models import NotificationRequest, NotificationStatus, NotificationTrigger
from wb_workspaces.models import (
    MembershipAccessMode,
    MembershipRole,
    UserLLMCredential,
    WorkspaceCredential,
    WorkspaceLLMProvider,
    WorkspaceMembership,
    WorkspaceReportExclusion,
)
from wb_workspaces.services import (
    create_workspace_for_user,
    upsert_user_llm_credential,
    upsert_user_llm_setting,
)

from .artifact_storage import StoredArtifactFile
from .models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunArtifact,
    RunApprovalStatus,
    RunStatus,
    RunType,
    RunWorkerHeartbeat,
)
from .pfd_toolkit_adapter import (
    AdapterCancelledError,
    _build_llm_kwargs,
    _patch_generate_with_progress,
    _theme_summary_from_dataframe,
)
from .services import (
    approve_run_for_execution,
    configure_pending_run_for_ops,
    queue_run,
    reject_run_for_execution,
    request_run_cancellation,
    set_run_status,
)
from .worker import process_single_available_run, reconcile_timed_out_runs


User = get_user_model()


class RunServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="run-owner@example.com", password="x")
        self.admin_user = User.objects.create_superuser(
            email="run-admin@example.com",
            password="x",
        )
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
        self.assertFalse(run.requires_approval)
        self.assertEqual(run.approval_status, RunApprovalStatus.NOT_REQUIRED)

    def test_queue_run_marks_pending_approval_when_requested(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"requires_manual_approval": True},
        )
        self.assertTrue(run.requires_approval)
        self.assertEqual(run.approval_status, RunApprovalStatus.PENDING)
        self.assertIsNotNone(run.approval_requested_at)

    @override_settings(
        WORKBENCH_BASE_URL="https://workbench.example.com",
        PFD_ADMIN_EMAIL="sam.osian@oreliandata.co.uk",
        EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    )
    def test_queue_run_auto_requires_approval_for_local_llm_runs(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": "local_ollama",
                "execution_mode": "real",
                "model_name": "gemma4:26b",
            },
        )
        self.assertTrue(run.requires_approval)
        self.assertEqual(run.approval_status, RunApprovalStatus.PENDING)
        self.assertIsNotNone(run.approval_requested_at)
        self.assertEqual(len(mail.outbox), 1)
        sent = mail.outbox[0]
        self.assertIn("Approval required", sent.subject)
        self.assertIn("sam.osian@oreliandata.co.uk", sent.to)
        self.assertIn("/ops/approvals/?run=", sent.body)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.approval_requested",
                target_id=str(run.id),
            ).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.approval_email_sent",
                target_id=str(run.id),
            ).exists()
        )

    def test_queue_run_local_pipeline_continuation_does_not_require_second_approval(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.THEMES,
            input_config_json={
                "provider": "local_ollama",
                "execution_mode": "real",
                "model_name": "gemma4:26b",
                "pipeline_plan": ["filter", "themes", "extract"],
                "pipeline_index": 1,
            },
        )
        self.assertFalse(run.requires_approval)
        self.assertEqual(run.approval_status, RunApprovalStatus.NOT_REQUIRED)

    def test_queue_run_openai_real_does_not_require_approval(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": "openai",
                "execution_mode": "real",
                "model_name": "gpt-4.1-mini",
            },
        )
        self.assertFalse(run.requires_approval)
        self.assertEqual(run.approval_status, RunApprovalStatus.NOT_REQUIRED)

    def test_staff_can_attach_one_time_ops_openai_key_without_saving_user_credential(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "real",
                "provider": "local_ollama",
                "requires_manual_approval": True,
            },
        )
        configured = configure_pending_run_for_ops(
            actor=self.admin_user,
            run=run,
            provider="openai",
            api_key="sk-test-ops-1234",
        )
        configured.refresh_from_db()
        self.assertEqual(configured.input_config_json.get("provider"), "openai")
        self.assertEqual(configured.input_config_json.get("execution_mode"), "real")
        self.assertEqual(configured.ops_override_provider, "openai")
        self.assertEqual(configured.ops_override_key_last4, "1234")
        self.assertTrue(bool(configured.ops_override_encrypted_api_key))
        self.assertFalse(configured.requires_approval)
        self.assertEqual(configured.approval_status, RunApprovalStatus.NOT_REQUIRED)
        self.assertFalse(
            WorkspaceCredential.objects.filter(
                workspace=self.workspace,
                user=self.owner,
                provider="openai",
            ).exists()
        )
        self.assertFalse(
            UserLLMCredential.objects.filter(
                user=self.owner,
                provider="openai",
            ).exists()
        )

    def test_superuser_can_approve_queued_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"requires_manual_approval": True},
        )
        scheduled_for = timezone.now() + timedelta(minutes=15)
        approved = approve_run_for_execution(
            actor=self.admin_user,
            run=run,
            note="Approved for later execution",
            scheduled_for=scheduled_for,
        )
        self.assertEqual(approved.approval_status, RunApprovalStatus.APPROVED)
        self.assertEqual(approved.approved_by_id, self.admin_user.id)
        self.assertIsNotNone(approved.approved_at)
        self.assertEqual(approved.queued_at, scheduled_for)

    def test_superuser_can_reject_queued_run_and_mark_terminal(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"requires_manual_approval": True},
        )
        rejected = reject_run_for_execution(
            actor=self.admin_user,
            run=run,
            reason="Capacity not available",
        )
        self.assertEqual(rejected.approval_status, RunApprovalStatus.REJECTED)
        self.assertEqual(rejected.status, RunStatus.CANCELLED)
        self.assertEqual(rejected.rejected_by_id, self.admin_user.id)
        self.assertEqual(rejected.approval_note, "Capacity not available")

    def test_non_superuser_cannot_approve_or_reject(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"requires_manual_approval": True},
        )
        with self.assertRaises(PermissionDenied):
            approve_run_for_execution(actor=self.owner, run=run)
        with self.assertRaises(PermissionDenied):
            reject_run_for_execution(actor=self.owner, run=run)

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

    def test_terminal_status_normalises_stale_pending_approval_state(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"requires_manual_approval": True},
        )
        run.status = RunStatus.STARTING
        run.save(update_fields=["status", "updated_at"])

        set_run_status(
            run=run,
            status=RunStatus.FAILED,
            message="Failed after being picked up",
        )
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.approval_status, RunApprovalStatus.NOT_REQUIRED)
        self.assertFalse(run.requires_approval)

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

    def test_queue_real_local_ollama_run_does_not_require_api_key(self):
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
                "provider": "local_ollama",
                "model_name": "gemma4:26b",
                "api_key": "",
                "input_config_json": '{"execution_mode": "real"}',
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run = self.investigation.runs.filter(
            run_type=RunType.FILTER,
            input_config_json__execution_mode="real",
        ).latest("created_at")
        self.assertEqual(run.input_config_json.get("provider"), "local_ollama")
        self.assertEqual(run.input_config_json.get("model_name"), "gemma4:26b")

    def test_queue_real_run_defaults_to_saved_llm_config_when_route_omitted(self):
        upsert_user_llm_setting(
            actor=self.owner,
            provider=WorkspaceLLMProvider.OPENAI,
            model_name="gpt-5.4",
            max_parallel_workers=7,
        )
        upsert_user_llm_credential(
            actor=self.owner,
            provider=WorkspaceLLMProvider.OPENAI,
            api_key="sk-test-secret-5678",
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
                "api_key": "",
                "input_config_json": '{"execution_mode": "real"}',
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run = self.investigation.runs.filter(
            run_type=RunType.FILTER,
            input_config_json__execution_mode="real",
        ).latest("created_at")
        self.assertEqual(run.input_config_json.get("provider"), "openai")
        self.assertEqual(run.input_config_json.get("model_name"), "gpt-5.4")
        self.assertEqual(run.input_config_json.get("max_parallel_workers"), 7)

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

    def test_cancel_run_view_redirects_to_next_url_and_sets_prune_skip_flag(self):
        self.client.force_login(self.owner)
        next_url = reverse("workbook-dashboard")
        response = self.client.post(
            reverse(
                "workbook-run-cancel",
                kwargs={"workbook_id": self.workspace.id, "run_id": self.run.id},
            ),
            data={"cancel_reason": "Stop now", "next_url": next_url},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, next_url)
        session = self.client.session
        self.assertTrue(session.get("wb_skip_cancelled_prune_once"))

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


class RunAdapterTests(TestCase):
    def test_patch_generate_maps_toolkit_cancellation_to_adapter_cancellation(self):
        captured = {}

        class DummyLLMClient:
            def generate(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                raise GenerationCancelledError("cancelled")

        llm_client = DummyLLMClient()
        wrapped = _patch_generate_with_progress(
            llm_client=llm_client,
            progress_start=10,
            progress_end=90,
            progress_callback=None,
            cancellation_check=lambda: False,
            default_message="Testing cancellation",
        )

        with self.assertRaises(AdapterCancelledError):
            wrapped.generate(["prompt"])

        self.assertIn("progress_callback", captured["kwargs"])
        self.assertIn("cancellation_check", captured["kwargs"])
        self.assertTrue(callable(captured["kwargs"]["cancellation_check"]))

    @override_settings(
        LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434/v1",
        LOCAL_OLLAMA_MODEL_DEFAULT="gemma4:26b",
        LOCAL_OLLAMA_API_KEY="ollama",
        LOCAL_OLLAMA_MAX_PARALLEL_WORKERS=1,
    )
    def test_build_llm_kwargs_for_local_ollama_uses_local_defaults(self):
        with patch("wb_runs.pfd_toolkit_adapter.resolve_workspace_credential") as mocked_resolve:
            kwargs = _build_llm_kwargs(
                run=object(),
                config={
                    "provider": "local_ollama",
                    "model_name": "",
                    "max_parallel_workers": 9,
                },
            )
        mocked_resolve.assert_not_called()
        self.assertEqual(kwargs.get("api_key"), "ollama")
        self.assertEqual(kwargs.get("base_url"), "http://127.0.0.1:11434/v1")
        self.assertEqual(kwargs.get("model"), "gemma4:26b")
        self.assertEqual(kwargs.get("max_workers"), 1)
        self.assertEqual(kwargs.get("reasoning_effort"), "none")

    def test_build_llm_kwargs_prefers_run_scoped_ops_override(self):
        owner = User.objects.create_user(email="adapter-owner@example.com", password="x")
        workspace = create_workspace_for_user(
            user=owner,
            title="Adapter Workspace",
            slug="adapter-workspace",
            description="desc",
        )
        investigation = create_investigation(
            actor=owner,
            workspace=workspace,
            title="Adapter Investigation",
            question_text="Question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = InvestigationRun.objects.create(
            investigation=investigation,
            workspace=workspace,
            requested_by=owner,
            run_type=RunType.FILTER,
            ops_override_provider="openai",
            ops_override_encrypted_api_key="",
            input_config_json={"provider": "openai"},
        )
        configured = configure_pending_run_for_ops(
            actor=User.objects.create_superuser(email="adapter-admin@example.com", password="x"),
            run=run,
            provider="openai",
            api_key="sk-override-1234",
        )
        with patch("wb_runs.pfd_toolkit_adapter.resolve_workspace_credential") as mocked_resolve:
            kwargs = _build_llm_kwargs(
                run=configured,
                config={
                    "provider": "openai",
                    "model_name": "gpt-5-mini",
                    "max_parallel_workers": 3,
                },
            )
        mocked_resolve.assert_not_called()
        self.assertEqual(kwargs.get("api_key"), "sk-override-1234")
        self.assertEqual(kwargs.get("model"), "gpt-5-mini")
        self.assertEqual(kwargs.get("max_workers"), 3)


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

    def test_api_route_worker_does_not_claim_local_route_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": "local_ollama",
                "execution_mode": "simulate",
                "requires_manual_approval": False,
            },
        )
        processed = process_single_available_run(worker_id="test-worker", route_mode="api")
        self.assertIsNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)

    def test_local_route_worker_does_not_claim_api_route_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": "openai",
                "execution_mode": "simulate",
            },
        )
        processed = process_single_available_run(worker_id="test-worker", route_mode="local")
        self.assertIsNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)

    def test_local_route_worker_claims_local_route_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": "local_ollama",
                "execution_mode": "simulate",
                "requires_manual_approval": False,
            },
        )
        processed = process_single_available_run(worker_id="test-worker", route_mode="local")
        self.assertIsNotNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)

    def test_api_route_worker_does_not_claim_local_route_with_whitespace_provider(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "provider": " local_ollama ",
                "execution_mode": "simulate",
                "requires_manual_approval": False,
            },
        )
        processed = process_single_available_run(worker_id="test-worker", route_mode="api")
        self.assertIsNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(run.worker_id, "")

    def test_worker_skips_pending_approval_run(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "requires_manual_approval": True,
            },
        )
        processed = process_single_available_run(worker_id="test-worker")
        self.assertIsNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(run.approval_status, RunApprovalStatus.PENDING)

    def test_worker_processes_approved_run_that_requires_approval(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "requires_manual_approval": True,
            },
        )
        run.approval_status = RunApprovalStatus.APPROVED
        run.approved_at = timezone.now()
        run.approved_by = self.owner
        run.save(update_fields=["approval_status", "approved_at", "approved_by", "updated_at"])

        processed = process_single_available_run(worker_id="test-worker")
        self.assertIsNotNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)

    def test_worker_refuses_execution_when_claimed_run_is_pending_approval(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "requires_manual_approval": True,
            },
        )
        with patch("wb_runs.worker.claim_next_runnable_run", return_value=run):
            processed = process_single_available_run(worker_id="test-worker", route_mode="api")
        self.assertIsNone(processed)
        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(run.approval_status, RunApprovalStatus.PENDING)
        self.assertEqual(run.worker_id, "")

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

    def test_worker_marks_real_run_cancelled_when_adapter_raises_cancelled(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            side_effect=AdapterCancelledError("cancelled by test"),
        ):
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.CANCELLED)

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

    def test_theme_summary_includes_theme_descriptions(self):
        theme_model = create_model(
            "ThemeModel",
            care_coordination=(bool, Field(description="Coordination gaps between services.")),
            discharge_failures=(bool, Field(description="Unsafe or delayed discharge processes.")),
        )
        themed_df = pd.DataFrame(
            [
                {"care_coordination": True, "discharge_failures": False},
                {"care_coordination": False, "discharge_failures": True},
                {"care_coordination": True, "discharge_failures": True},
            ]
        )

        summary_df = _theme_summary_from_dataframe(themed_df, theme_model)

        self.assertEqual(
            list(summary_df.columns),
            ["theme", "description", "matched_reports"],
        )
        self.assertEqual(
            summary_df.to_dict(orient="records"),
            [
                {
                    "theme": "care_coordination",
                    "description": "Coordination gaps between services.",
                    "matched_reports": 2,
                },
                {
                    "theme": "discharge_failures",
                    "description": "Unsafe or delayed discharge processes.",
                    "matched_reports": 2,
                },
            ],
        )

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

    def test_pipeline_moves_pending_notification_to_next_stage(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "simulate",
                "pipeline_plan": [RunType.FILTER, RunType.EXTRACT],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        notification = NotificationRequest.objects.create(
            run=run,
            user=self.owner,
            notify_on=NotificationTrigger.ANY,
            status=NotificationStatus.PENDING,
        )

        process_single_available_run(worker_id="test-worker")

        next_run = (
            self.investigation.runs.filter(run_type=RunType.EXTRACT)
            .exclude(id=run.id)
            .first()
        )
        self.assertIsNotNone(next_run)
        notification.refresh_from_db()
        self.assertEqual(notification.run_id, next_run.id)
        self.assertEqual(notification.status, NotificationStatus.PENDING)
        self.assertTrue(
            next_run.events.filter(
                message="Completion notification moved to next pipeline stage.",
            ).exists()
        )

    def test_pipeline_continuation_carries_run_scoped_ops_override(self):
        staff = User.objects.create_superuser(email="worker-admin@example.com", password="x")
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
        configure_pending_run_for_ops(
            actor=staff,
            run=run,
            provider="openai",
            api_key="sk-pipeline-1234",
        )

        process_single_available_run(worker_id="test-worker")

        next_run = (
            self.investigation.runs.filter(run_type=RunType.THEMES)
            .exclude(id=run.id)
            .first()
        )
        self.assertIsNotNone(next_run)
        self.assertEqual(next_run.ops_override_provider, "openai")
        self.assertEqual(next_run.ops_override_key_last4, "1234")
        self.assertTrue(bool(next_run.ops_override_encrypted_api_key))

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

    def test_filter_run_with_zero_matches_fails_with_no_relevant_reports_error(self):
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "execution_mode": "real",
                "search_query": "query with no matches",
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )

        with patch(
            "wb_runs.worker.execute_filter_workflow",
            return_value={
                "output_path": "",
                "total_reports": 10,
                "matched_reports": 0,
                "output_reports": 0,
                "search_query": "query with no matches",
                "filter_df": True,
                "produce_spans": False,
                "drop_spans": False,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "report_limit": None,
            },
        ):
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.error_code, "NO_RELEVANT_REPORTS")
        self.assertIn("did not find any reports", (run.error_message or "").lower())
        self.assertFalse(
            self.investigation.runs.filter(run_type=RunType.THEMES).exclude(id=run.id).exists()
        )

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

    @override_settings(WORKER_ACTIVE_HEARTBEAT_SECONDS=1)
    def test_worker_refreshes_claimed_heartbeat_during_real_run_execution(self):
        output_path = Path("/tmp/test-run-heartbeat-output.csv")
        output_path.write_text("id,matches_query\n1,True\n", encoding="utf-8")
        run = queue_run(
            actor=self.owner,
            investigation=self.investigation,
            run_type=RunType.FILTER,
            input_config_json={"execution_mode": "real", "search_query": "medication safety"},
        )

        def _adapter_side_effect(*, run, progress_callback, cancellation_check):
            progress_callback(30, "Sending requests to the LLM")
            progress_callback(45, "Sending requests to the LLM")
            return {
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
            }

        with patch("wb_runs.worker.execute_filter_workflow", side_effect=_adapter_side_effect), patch(
            "wb_runs.worker.time.monotonic",
            side_effect=[0.0, 2.0, 3.0, 4.0, 5.0],
        ), patch("wb_runs.worker._record_worker_heartbeat_safe") as mocked_safe:
            process_single_available_run(worker_id="test-worker")

        run.refresh_from_db()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertGreaterEqual(mocked_safe.call_count, 1)

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
