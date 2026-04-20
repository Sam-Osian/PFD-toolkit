from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from django.urls import reverse
from datetime import date

from wb_auditlog.models import AuditEvent
from wb_runs.models import InvestigationRun, RunEvent, RunEventType, RunStatus, RunType
from wb_workspaces.models import MembershipAccessMode, MembershipRole, WorkspaceMembership, WorkspaceVisibility
from wb_workspaces.services import create_workspace_for_user

from .forms import (
    InvestigationWizardExtractConfigForm,
    InvestigationWizardMethodForm,
    InvestigationWizardState,
    InvestigationWizardThemesConfigForm,
    TemporalScopeOption,
    temporal_scope_parameters,
)
from .models import Investigation, InvestigationStatus
from .services import InvestigationServiceError, create_investigation, update_investigation


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

    def test_workspace_allows_only_one_investigation(self):
        create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Primary",
            question_text="Q1",
            scope_json={},
            method_json={},
            status=InvestigationStatus.DRAFT,
        )
        with self.assertRaises(InvestigationServiceError):
            create_investigation(
                actor=self.owner,
                workspace=self.workspace,
                title="Secondary",
                question_text="Q2",
                scope_json={},
                method_json={},
                status=InvestigationStatus.DRAFT,
            )


class InvestigationViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="inv-owner2@example.com", password="x")
        self.viewer = User.objects.create_user(email="inv-viewer2@example.com", password="x")
        self.stranger = User.objects.create_user(email="inv-stranger2@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Inv Workspace2",
            slug="inv-workspace2",
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
            title="Q1",
            question_text="What is happening?",
            scope_json={"dataset": "all"},
            method_json={"model": "x"},
            status=InvestigationStatus.DRAFT,
        )

    def test_second_investigation_create_redirects_to_existing(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-investigation-list", kwargs={"workbook_id": self.workspace.id}),
            data={
                "title": "Created by view",
                "question_text": "A question",
                "scope_json": '{"filters": []}',
                "method_json": '{"method": "filter"}',
                "status": InvestigationStatus.DRAFT,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            Investigation.objects.filter(workspace=self.workspace).count(),
            1,
        )

    def test_stranger_cannot_view_private_workspace_investigations(self):
        self.client.force_login(self.stranger)
        response = self.client.get(
            reverse("workbook-investigation-list", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    def test_public_workspace_investigation_list_viewable_without_login(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])
        response = self.client.get(
            reverse("workbook-investigation-list", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 200)

    def test_investigation_entry_redirects_editor_to_wizard(self):
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse("workbook-investigation-entry", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/wizard/", response.url)

    def test_investigation_entry_redirects_read_only_user_to_detail(self):
        self.client.force_login(self.viewer)
        response = self.client.get(
            reverse("workbook-investigation-entry", kwargs={"workbook_id": self.workspace.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/investigations/", response.url)
        self.assertNotIn("/wizard/", response.url)

    def test_investigation_entry_creates_investigation_for_editor_when_missing(self):
        self.client.force_login(self.owner)
        workspace_without_investigation = create_workspace_for_user(
            user=self.owner,
            title="No Investigation Yet",
            slug="no-investigation-yet",
            description="desc",
        )
        self.assertFalse(
            Investigation.objects.filter(workspace=workspace_without_investigation).exists()
        )
        response = self.client.get(
            reverse(
                "workbook-investigation-entry",
                kwargs={"workbook_id": workspace_without_investigation.id},
            )
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/wizard/", response.url)
        self.assertTrue(
            Investigation.objects.filter(workspace=workspace_without_investigation).exists()
        )

    def test_bot_investigation_detail_view_does_not_update_last_viewed(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])

        self.client.get(
            reverse(
                "workbook-investigation-detail",
                kwargs={
                    "workbook_id": self.workspace.id,
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
                "workbook-investigation-detail",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            ),
            HTTP_USER_AGENT="Mozilla/5.0",
        )
        self.workspace.refresh_from_db()
        self.investigation.refresh_from_db()
        self.assertIsNotNone(self.workspace.last_viewed_at)
        self.assertIsNotNone(self.investigation.last_viewed_at)

    def test_investigation_detail_shows_wizard_first_ui(self):
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse(
                "workbook-investigation-detail",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Open wizard")
        self.assertContains(response, "Configuration Snapshot")
        self.assertNotContains(response, "Scope JSON")
        self.assertNotContains(response, "Queue Run")

    def test_investigation_detail_shows_pipeline_timeline_when_pipeline_runs_exist(self):
        self.client.force_login(self.owner)
        root = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.SUCCEEDED,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        next_run = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.THEMES,
            status=RunStatus.QUEUED,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 1,
                "pipeline_continue_on_fail": True,
            },
        )
        RunEvent.objects.create(
            run=next_run,
            event_type=RunEventType.INFO,
            message="Run queued by investigation pipeline.",
            payload_json={
                "pipeline_previous_run_id": str(root.id),
                "pipeline_index": 1,
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
            },
        )
        response = self.client.get(
            reverse(
                "workbook-investigation-detail",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pipeline Timeline")
        self.assertContains(response, "continue on fail: yes")
        self.assertContains(response, "Screen and filter")
        self.assertContains(response, "Discover themes")

    def test_pipeline_timeline_flags_continued_after_failure(self):
        self.client.force_login(self.owner)
        root = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.FILTER,
            status=RunStatus.FAILED,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 0,
                "pipeline_continue_on_fail": True,
            },
        )
        next_run = InvestigationRun.objects.create(
            investigation=self.investigation,
            workspace=self.workspace,
            requested_by=self.owner,
            run_type=RunType.THEMES,
            status=RunStatus.QUEUED,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
                "pipeline_index": 1,
                "pipeline_continue_on_fail": True,
            },
        )
        RunEvent.objects.create(
            run=next_run,
            event_type=RunEventType.INFO,
            message="Run queued by investigation pipeline.",
            payload_json={
                "pipeline_previous_run_id": str(root.id),
                "pipeline_index": 1,
                "pipeline_plan": [RunType.FILTER, RunType.THEMES],
            },
        )
        response = self.client.get(
            reverse(
                "workbook-investigation-detail",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "investigation_id": self.investigation.id,
                },
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Continued after previous stage failed/timed out.")


class InvestigationWizardFormTests(TestCase):
    def test_temporal_scope_parameters(self):
        today = date(2026, 4, 20)

        all_reports = temporal_scope_parameters(
            scope_option=TemporalScopeOption.ALL_REPORTS,
            today=today,
        )
        self.assertIsNone(all_reports["query_start_date"])
        self.assertIsNone(all_reports["query_end_date"])
        self.assertIsNone(all_reports["report_limit"])

        last_year = temporal_scope_parameters(
            scope_option=TemporalScopeOption.LAST_YEAR,
            today=today,
        )
        self.assertEqual(last_year["query_start_date"], date(2025, 4, 20))
        self.assertEqual(last_year["query_end_date"], today)

        recent_100 = temporal_scope_parameters(
            scope_option=TemporalScopeOption.MOST_RECENT_100,
            today=today,
        )
        self.assertEqual(recent_100["report_limit"], 100)

    def test_method_form_pipeline_plan_fixed_order(self):
        form = InvestigationWizardMethodForm(
            data={"run_filter": "on", "run_themes": "on", "run_extract": "on"}
        )
        self.assertTrue(form.is_valid())
        self.assertEqual(form.pipeline_plan(), ["filter", "themes", "extract"])

    def test_method_form_requires_at_least_one_stage(self):
        form = InvestigationWizardMethodForm(data={})
        self.assertFalse(form.is_valid())

    def test_themes_config_rejects_invalid_min_max(self):
        form = InvestigationWizardThemesConfigForm(
            data={"enabled": "on", "min_themes": 8, "max_themes": 2}
        )
        self.assertFalse(form.is_valid())

    def test_extract_config_requires_feature_fields_when_enabled(self):
        form = InvestigationWizardExtractConfigForm(
            data={"enabled": "on", "feature_fields": []}
        )
        self.assertFalse(form.is_valid())

    def test_extract_config_accepts_valid_feature_fields(self):
        form = InvestigationWizardExtractConfigForm(
            data={
                "enabled": "on",
                "feature_fields": (
                    '[{"name":"age","description":"Age in years","type":"integer"},'
                    '{"name":"setting","description":"Care setting","type":"text"}]'
                ),
            }
        )
        self.assertTrue(form.is_valid())

    def test_extract_config_accepts_pipe_separated_lines(self):
        form = InvestigationWizardExtractConfigForm(
            data={
                "enabled": "on",
                "feature_fields": "age | Age in years | integer\nsetting | Care setting | text",
            }
        )
        self.assertTrue(form.is_valid())

    def test_wizard_state_round_trip_and_pipeline_plan(self):
        state = InvestigationWizardState.from_json(
            {
                "stage": "method",
                "title": "T",
                "question_text": "Q",
                "scope_option": TemporalScopeOption.LAST_3_YEARS,
                "run_filter": False,
                "run_themes": True,
                "run_extract": False,
                "themes_config": {"min_themes": 3},
            }
        )
        self.assertEqual(state.pipeline_plan(), ["themes"])
        payload = state.to_json()
        self.assertEqual(payload["stage"], "method")
        self.assertEqual(payload["scope_option"], TemporalScopeOption.LAST_3_YEARS)


class InvestigationWizardViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="inv-wiz-owner@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Wizard Workspace",
            slug="wizard-workspace",
            description="desc",
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Wizard Investigation",
            question_text="Initial question",
            scope_json={},
            method_json={},
            status=InvestigationStatus.DRAFT,
        )

    def _wizard_url(self) -> str:
        return reverse(
            "workbook-investigation-wizard",
            kwargs={
                "workbook_id": self.workspace.id,
                "investigation_id": self.investigation.id,
            },
        )

    def _advance_question_and_scope(self, wizard_url: str):
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "title": "Updated Investigation",
                "question_text": "What are medication safety failures?",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "scope_option": "last_year",
            },
        )

    def _launch_from_review(self, wizard_url: str):
        return self.client.post(
            wizard_url,
            data={
                "wizard_action": "launch",
                "execution_mode": "simulate",
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "request_completion_email": "",
                "notify_on": "any",
            },
        )

    def _assert_latest_run_pipeline_plan(self, expected_plan: list[str]):
        self.investigation.refresh_from_db()
        run = self.investigation.runs.latest("created_at")
        self.assertEqual(run.run_type, expected_plan[0])
        self.assertEqual(run.input_config_json.get("pipeline_plan"), expected_plan)
        self.assertEqual(run.input_config_json.get("pipeline_index"), 0)
        self.assertEqual(run.input_config_json.get("execution_mode"), "simulate")
        return run

    def test_wizard_launch_filter_themes_extract_path(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()

        self.assertEqual(self.client.get(wizard_url).status_code, 200)
        self._advance_question_and_scope(wizard_url)
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "run_filter": "on",
                "run_themes": "on",
                "run_extract": "on",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "min_themes": 2,
                "max_themes": 8,
                "seed_topics": "medication safety\nhandover",
                "extra_theme_instructions": "Focus on actionable prevention patterns.",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "feature_fields": (
                    '[{"name":"setting","description":"Care setting","type":"text"},'
                    '{"name":"age","description":"Age in years","type":"integer"}]'
                ),
                "allow_multiple": "on",
            },
        )
        launch_response = self._launch_from_review(wizard_url)
        self.assertEqual(launch_response.status_code, 302)

        run = self._assert_latest_run_pipeline_plan(["filter", "themes", "extract"])
        self.assertEqual(run.input_config_json.get("search_query"), "What are medication safety failures?")
        self.assertEqual(run.input_config_json.get("min_themes"), 2)
        self.assertEqual(run.input_config_json.get("max_themes"), 8)
        self.assertTrue(isinstance(run.input_config_json.get("feature_fields"), list))

    def test_wizard_launch_filter_only_path(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()
        self._advance_question_and_scope(wizard_url)
        review_response = self.client.post(
            wizard_url,
            data={"wizard_action": "next", "run_filter": "on"},
            follow=True,
        )
        self.assertEqual(review_response.status_code, 200)
        self.assertContains(review_response, "Final check before launch.")
        launch_response = self._launch_from_review(wizard_url)
        self.assertEqual(launch_response.status_code, 302)
        self._assert_latest_run_pipeline_plan(["filter"])

    def test_wizard_launch_filter_themes_path(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()
        self._advance_question_and_scope(wizard_url)
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "run_filter": "on",
                "run_themes": "on",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "min_themes": 2,
                "max_themes": 6,
                "seed_topics": "medication safety",
            },
        )
        launch_response = self._launch_from_review(wizard_url)
        self.assertEqual(launch_response.status_code, 302)
        run = self._assert_latest_run_pipeline_plan(["filter", "themes"])
        self.assertEqual(run.input_config_json.get("min_themes"), 2)
        self.assertEqual(run.input_config_json.get("max_themes"), 6)
        self.assertFalse(isinstance(run.input_config_json.get("feature_fields"), list))

    def test_wizard_launch_filter_extract_path(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()
        self._advance_question_and_scope(wizard_url)
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "run_filter": "on",
                "run_extract": "on",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "feature_fields": (
                    '[{"name":"setting","description":"Care setting","type":"text"}]'
                ),
                "allow_multiple": "on",
            },
        )
        launch_response = self._launch_from_review(wizard_url)
        self.assertEqual(launch_response.status_code, 302)
        run = self._assert_latest_run_pipeline_plan(["filter", "extract"])
        self.assertTrue(isinstance(run.input_config_json.get("feature_fields"), list))

    def test_method_branching_skips_optional_steps_when_not_selected(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()

        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "title": "Updated Investigation",
                "question_text": "Question",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "scope_option": "all_reports",
            },
        )
        response = self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "run_filter": "on",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "review")

    def test_wizard_launch_themes_extract_without_filter(self):
        self.client.force_login(self.owner)
        wizard_url = self._wizard_url()
        self._advance_question_and_scope(wizard_url)
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "run_themes": "on",
                "run_extract": "on",
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "min_themes": 2,
                "max_themes": 4,
            },
        )
        self.client.post(
            wizard_url,
            data={
                "wizard_action": "next",
                "enabled": "on",
                "feature_fields": (
                    '[{"name":"setting","description":"Care setting","type":"text"}]'
                ),
            },
        )
        launch_response = self._launch_from_review(wizard_url)
        self.assertEqual(launch_response.status_code, 302)
        run = self._assert_latest_run_pipeline_plan(["themes", "extract"])
        self.assertEqual(run.run_type, RunType.THEMES)
