from datetime import timedelta
import os
import re
import io
import json
import tempfile

from django.core.exceptions import PermissionDenied, ValidationError
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_investigations.models import Investigation, InvestigationStatus
from wb_investigations.services import create_investigation
from wb_runs.models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunStatus,
    RunArtifact,
    RunType,
    RunWorkerHeartbeat,
)
from wb_runs.services import queue_run

from .lifecycle import run_lifecycle_maintenance
from .models import (
    MembershipAccessMode,
    MembershipRole,
    Workspace,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceLLMSetting,
    WorkspaceVisibility,
    WorkspaceReportExclusion,
    WorkspaceRevision,
    WorkspaceUserState,
)
from .permissions import can_edit_workspace, can_manage_members, can_run_workflows, can_view_workspace
from .revisions import (
    redo_workspace_revision,
    revert_workspace_reports,
    start_over_workspace_state,
    undo_workspace_revision,
)
from .services import (
    add_workspace_member,
    archive_workspace,
    create_workspace_for_user,
    delete_workspace_immediately,
    remove_workspace_member,
    restore_workspace,
    restore_workspace_report_exclusion,
    resolve_workspace_credential,
    get_workspace_llm_setting,
    upsert_workspace_report_exclusion,
    upsert_workspace_credential,
    upsert_workspace_llm_setting,
    upsert_user_llm_credential,
    update_workspace_member,
)


User = get_user_model()


class WorkspacePermissionTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner@example.com", password="x")
        self.editor = User.objects.create_user(email="editor@example.com", password="x")
        self.viewer = User.objects.create_user(email="viewer@example.com", password="x")
        self.stranger = User.objects.create_user(email="stranger@example.com", password="x")

        self.workspace = Workspace.objects.create(
            created_by=self.owner,
            title="Workspace A",
            slug="workspace-a",
            visibility=WorkspaceVisibility.PRIVATE,
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
        WorkspaceMembership.objects.create(
            workspace=self.workspace,
            user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_manage_members=False,
            can_manage_shares=False,
            can_run_workflows=True,
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

    def test_private_workspace_visibility(self):
        self.assertTrue(can_view_workspace(self.owner, self.workspace))
        self.assertTrue(can_view_workspace(self.editor, self.workspace))
        self.assertTrue(can_view_workspace(self.viewer, self.workspace))
        self.assertFalse(can_view_workspace(self.stranger, self.workspace))

    def test_public_workspace_visibility_for_non_member(self):
        self.workspace.visibility = WorkspaceVisibility.PUBLIC
        self.workspace.save(update_fields=["visibility"])
        self.assertTrue(can_view_workspace(self.stranger, self.workspace))

    def test_edit_requires_edit_mode_and_role(self):
        self.assertTrue(can_edit_workspace(self.owner, self.workspace))
        self.assertTrue(can_edit_workspace(self.editor, self.workspace))
        self.assertFalse(can_edit_workspace(self.viewer, self.workspace))

    def test_manage_members_owner_only(self):
        self.assertTrue(can_manage_members(self.owner, self.workspace))
        self.assertFalse(can_manage_members(self.editor, self.workspace))
        self.assertFalse(can_manage_members(self.viewer, self.workspace))

    def test_run_permission_flag_controls_execution(self):
        self.assertTrue(can_run_workflows(self.owner, self.workspace))
        self.assertTrue(can_run_workflows(self.editor, self.workspace))
        self.assertFalse(can_run_workflows(self.viewer, self.workspace))


class WorkspaceServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner2@example.com", password="x")
        self.editor = User.objects.create_user(email="editor2@example.com", password="x")
        self.viewer = User.objects.create_user(email="viewer2@example.com", password="x")
        self.admin = User.objects.create_superuser(email="admin@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Service Workspace",
            slug="service-workspace",
            description="Desc",
        )

    def test_create_workspace_for_user_creates_owner_membership(self):
        membership = WorkspaceMembership.objects.get(workspace=self.workspace, user=self.owner)
        self.assertEqual(membership.role, MembershipRole.OWNER)
        self.assertEqual(membership.access_mode, MembershipAccessMode.EDIT)
        self.assertTrue(membership.can_manage_members)
        self.assertTrue(membership.can_manage_shares)
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace, action_type="workspace.created"
            ).exists()
        )
        state = WorkspaceUserState.objects.get(user=self.owner)
        self.assertEqual(state.active_workspace_id, self.workspace.id)
        self.workspace.refresh_from_db()
        self.assertIsNotNone(self.workspace.current_revision_id)
        self.assertTrue(
            WorkspaceRevision.objects.filter(
                workspace=self.workspace,
                revision_number=1,
            ).exists()
        )

    def test_add_workspace_member_creates_membership_and_audit(self):
        membership = add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )
        self.assertEqual(membership.user, self.editor)
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace, action_type="workspace.member_added"
            ).exists()
        )

    def test_non_admin_cannot_grant_owner_role_when_adding_member(self):
        with self.assertRaises(ValidationError):
            add_workspace_member(
                actor=self.owner,
                workspace=self.workspace,
                target_user=self.editor,
                role=MembershipRole.OWNER,
                access_mode=MembershipAccessMode.EDIT,
                can_run_workflows=True,
                can_manage_members_flag=True,
                can_manage_shares_flag=True,
            )

    def test_admin_can_grant_owner_role_when_adding_member(self):
        membership = add_workspace_member(
            actor=self.admin,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.OWNER,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=True,
            can_manage_shares_flag=True,
        )
        self.assertEqual(membership.role, MembershipRole.OWNER)

    def test_non_admin_cannot_promote_member_to_owner(self):
        membership = add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )
        with self.assertRaises(ValidationError):
            update_workspace_member(
                actor=self.owner,
                workspace=self.workspace,
                membership=membership,
                role=MembershipRole.OWNER,
                access_mode=MembershipAccessMode.EDIT,
                can_run_workflows=True,
                can_manage_members_flag=True,
                can_manage_shares_flag=True,
            )

    def test_admin_can_promote_member_to_owner(self):
        membership = add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )
        updated = update_workspace_member(
            actor=self.admin,
            workspace=self.workspace,
            membership=membership,
            role=MembershipRole.OWNER,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=True,
            can_manage_shares_flag=True,
        )
        self.assertEqual(updated.role, MembershipRole.OWNER)

    def test_non_owner_cannot_add_workspace_member(self):
        add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )

        with self.assertRaises(PermissionDenied):
            add_workspace_member(
                actor=self.editor,
                workspace=self.workspace,
                target_user=self.viewer,
                role=MembershipRole.VIEWER,
                access_mode=MembershipAccessMode.READ_ONLY,
                can_run_workflows=False,
                can_manage_members_flag=False,
                can_manage_shares_flag=False,
            )

    def test_update_member_enforces_owner_invariants(self):
        owner_membership = WorkspaceMembership.objects.get(
            workspace=self.workspace, user=self.owner
        )
        with self.assertRaises(ValidationError):
            update_workspace_member(
                actor=self.owner,
                workspace=self.workspace,
                membership=owner_membership,
                role=MembershipRole.OWNER,
                access_mode=MembershipAccessMode.READ_ONLY,
                can_run_workflows=True,
                can_manage_members_flag=False,
                can_manage_shares_flag=True,
            )

    def test_remove_last_owner_fails(self):
        owner_membership = WorkspaceMembership.objects.get(
            workspace=self.workspace, user=self.owner
        )
        with self.assertRaises(ValidationError):
            remove_workspace_member(
                actor=self.owner,
                workspace=self.workspace,
                membership=owner_membership,
            )

    def test_remove_member_logs_audit(self):
        target_membership = add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )
        remove_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            membership=target_membership,
        )
        self.assertFalse(
            WorkspaceMembership.objects.filter(id=target_membership.id).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="workspace.member_removed",
            ).exists()
        )


class WorkspaceCredentialTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-cred@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Credential Workspace",
            slug="credential-workspace",
            description="Credential Desc",
        )

    def test_upsert_and_resolve_workspace_credential(self):
        upsert_workspace_credential(
            actor=self.owner,
            workspace=self.workspace,
            provider="openai",
            api_key="sk-live-example-1234",
            base_url="",
        )
        api_key, base_url = resolve_workspace_credential(
            user=self.owner,
            workspace=self.workspace,
            provider="openai",
        )
        self.assertEqual(api_key, "sk-live-example-1234")
        self.assertEqual(base_url, "")
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="workspace.credential_saved",
            ).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="workspace.credential_used",
            ).exists()
        )

    def test_upsert_replaces_existing_key(self):
        upsert_workspace_credential(
            actor=self.owner,
            workspace=self.workspace,
            provider="openai",
            api_key="sk-live-example-1234",
        )
        upsert_workspace_credential(
            actor=self.owner,
            workspace=self.workspace,
            provider="openai",
            api_key="sk-live-example-9876",
        )
        api_key, _ = resolve_workspace_credential(
            user=self.owner,
            workspace=self.workspace,
            provider="openai",
        )
        self.assertEqual(api_key, "sk-live-example-9876")


class WorkspaceReportExclusionTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-excl@example.com", password="x")
        self.viewer = User.objects.create_user(email="viewer-excl@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Exclusions",
            slug="exclusions",
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

    def test_owner_can_upsert_and_restore_exclusion(self):
        exclusion = upsert_workspace_report_exclusion(
            actor=self.owner,
            workspace=self.workspace,
            report_identity="https://example.com/reports/1",
            reason="Out of scope",
            report_title="Test report",
        )
        self.assertEqual(exclusion.reason, "Out of scope")
        self.assertTrue(
            WorkspaceReportExclusion.objects.filter(
                workspace=self.workspace,
                report_identity="https://example.com/reports/1",
            ).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="workspace.report_excluded",
            ).exists()
        )

        restore_workspace_report_exclusion(
            actor=self.owner,
            workspace=self.workspace,
            exclusion=exclusion,
        )
        self.assertFalse(
            WorkspaceReportExclusion.objects.filter(id=exclusion.id).exists()
        )
        self.assertTrue(
            AuditEvent.objects.filter(
                workspace=self.workspace,
                action_type="workspace.report_restored",
            ).exists()
        )

    def test_read_only_member_cannot_exclude_report(self):
        with self.assertRaises(PermissionDenied):
            upsert_workspace_report_exclusion(
                actor=self.viewer,
                workspace=self.workspace,
                report_identity="rep-2",
            )


class WorkspaceRevisionServiceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-rev@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Revision Workspace",
            slug="revision-workspace",
            description="",
        )

    def test_undo_and_redo_restore_investigation_state(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Investigation A",
            question_text="q1",
            scope_json={"collection_slug": "local-gov"},
            method_json={"notes": "initial"},
            status=InvestigationStatus.ACTIVE,
        )
        from wb_investigations.services import update_investigation

        update_investigation(
            actor=self.owner,
            investigation=investigation,
            title=investigation.title,
            question_text="q2",
            scope_json={"collection_slug": "care-homes"},
            method_json=investigation.method_json,
            status=investigation.status,
        )

        undo_workspace_revision(actor=self.owner, workspace=self.workspace)
        investigation.refresh_from_db()
        self.assertEqual(investigation.question_text, "q1")
        self.assertEqual(investigation.scope_json.get("collection_slug"), "local-gov")

        redo_workspace_revision(actor=self.owner, workspace=self.workspace)
        investigation.refresh_from_db()
        self.assertEqual(investigation.question_text, "q2")
        self.assertEqual(investigation.scope_json.get("collection_slug"), "care-homes")

    def test_revert_reports_clears_exclusions(self):
        upsert_workspace_report_exclusion(
            actor=self.owner,
            workspace=self.workspace,
            report_identity="report-1",
            reason="Out of scope",
        )
        self.assertEqual(
            WorkspaceReportExclusion.objects.filter(workspace=self.workspace).count(),
            1,
        )

        revert_workspace_reports(actor=self.owner, workspace=self.workspace)
        self.assertEqual(
            WorkspaceReportExclusion.objects.filter(workspace=self.workspace).count(),
            0,
        )

    def test_start_over_restores_baseline_state(self):
        create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Investigation A",
            question_text="q1",
            scope_json={"collection_slug": "local-gov"},
            method_json={"notes": "initial"},
            status=InvestigationStatus.ACTIVE,
        )
        upsert_workspace_report_exclusion(
            actor=self.owner,
            workspace=self.workspace,
            report_identity="report-1",
            reason="Out of scope",
        )

        start_over_workspace_state(actor=self.owner, workspace=self.workspace)
        self.assertFalse(self.workspace.investigations.exists())
        self.assertFalse(
            WorkspaceReportExclusion.objects.filter(workspace=self.workspace).exists()
        )


class WorkspaceMemberViewsTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner3@example.com", password="x")
        self.editor = User.objects.create_user(email="editor3@example.com", password="x")
        self.viewer = User.objects.create_user(email="viewer3@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="View Workspace",
            slug="view-workspace",
            description="View Desc",
        )
        add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.viewer,
            role=MembershipRole.VIEWER,
            access_mode=MembershipAccessMode.READ_ONLY,
            can_run_workflows=False,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )

    def test_owner_can_add_member_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-member-add", kwargs={"workbook_id": self.workspace.id}),
            data={
                "email": self.editor.email,
                "role": MembershipRole.EDITOR,
                "access_mode": MembershipAccessMode.EDIT,
                "can_run_workflows": "on",
                "can_manage_members": "",
                "can_manage_shares": "",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            WorkspaceMembership.objects.filter(
                workspace=self.workspace, user=self.editor
            ).exists()
        )

    def test_non_owner_cannot_update_member_via_view(self):
        target_membership = add_workspace_member(
            actor=self.owner,
            workspace=self.workspace,
            target_user=self.editor,
            role=MembershipRole.EDITOR,
            access_mode=MembershipAccessMode.EDIT,
            can_run_workflows=True,
            can_manage_members_flag=False,
            can_manage_shares_flag=False,
        )
        self.client.force_login(self.editor)
        response = self.client.post(
            reverse(
                "workbook-member-update",
                kwargs={
                    "workbook_id": self.workspace.id,
                    "membership_id": target_membership.id,
                },
            ),
            data={
                "role": MembershipRole.OWNER,
                "access_mode": MembershipAccessMode.EDIT,
                "can_run_workflows": "on",
                "can_manage_members": "on",
                "can_manage_shares": "on",
            },
        )
        self.assertEqual(response.status_code, 302)
        target_membership.refresh_from_db()
        self.assertEqual(target_membership.role, MembershipRole.EDITOR)

    def test_owner_can_save_workspace_credential_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-credential-save", kwargs={"workbook_id": self.workspace.id}),
            data={
                "provider": "openai",
                "api_key": "sk-test-owner-1234",
                "base_url": "",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Saved openai credential ending in 1234.")
        self.assertTrue(
            WorkspaceCredential.objects.filter(
                workspace=self.workspace,
                user=self.owner,
                provider="openai",
                key_last4="1234",
            ).exists()
        )

    def test_owner_can_delete_workspace_credential_via_view(self):
        upsert_workspace_credential(
            actor=self.owner,
            workspace=self.workspace,
            provider="openai",
            api_key="sk-test-owner-1234",
        )
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-credential-remove", kwargs={"workbook_id": self.workspace.id}),
            data={"provider": "openai"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Deleted openai credential.")
        self.assertFalse(
            WorkspaceCredential.objects.filter(
                workspace=self.workspace,
                user=self.owner,
                provider="openai",
            ).exists()
        )

    def test_viewer_without_run_permission_cannot_save_workspace_credential(self):
        self.client.force_login(self.viewer)
        response = self.client.post(
            reverse("workbook-credential-save", kwargs={"workbook_id": self.workspace.id}),
            data={
                "provider": "openai",
                "api_key": "sk-test-viewer-1234",
                "base_url": "",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_owner_can_save_active_llm_config_with_optional_credential(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workspace-llm-config-save"),
            data={
                "provider": "openrouter",
                "model_name": "gpt-4.1-mini",
                "max_parallel_workers": "4",
                "api_key": "sk-or-example-1234",
                "base_url": "",
                "next_url": reverse("workbook-dashboard"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "LLM config and credential saved.")
        setting = self.owner.llm_setting
        self.assertEqual(setting.provider, "openrouter")
        self.assertEqual(setting.model_name, "gpt-4.1-mini")
        self.assertEqual(setting.max_parallel_workers, 4)
        self.assertTrue(self.owner.llm_credentials.filter(provider="openrouter", key_last4="1234").exists())

    def test_owner_can_clear_active_llm_credential(self):
        upsert_user_llm_credential(
            actor=self.owner,
            provider="openrouter",
            api_key="sk-or-example-1234",
            base_url="",
        )
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workspace-llm-config-credential-clear"),
            data={
                "provider": "openrouter",
                "next_url": reverse("workbook-dashboard"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cleared openrouter credential from your account defaults.")
        self.assertFalse(
            self.owner.llm_credentials.filter(provider="openrouter").exists()
        )


class WorkspaceLLMSettingTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-llm-setting@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="LLM Settings Workspace",
            slug="llm-settings-workspace",
            description="",
        )

    def test_upsert_and_get_workspace_llm_setting(self):
        upsert_workspace_llm_setting(
            actor=self.owner,
            workspace=self.workspace,
            provider="openrouter",
            model_name="gpt-4.1-mini",
            max_parallel_workers=3,
        )
        setting = get_workspace_llm_setting(user=self.owner, workspace=self.workspace)
        self.assertEqual(setting.get("provider"), "openrouter")
        self.assertEqual(setting.get("model_name"), "gpt-4.1-mini")
        self.assertEqual(setting.get("max_parallel_workers"), 3)


class WorkspaceActiveStateViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-active@example.com", password="x")
        RunWorkerHeartbeat.objects.update_or_create(
            worker_id="test-worker",
            defaults={"state": "idle", "last_seen_at": timezone.now()},
        )
        self.workspace_a = create_workspace_for_user(
            user=self.owner,
            title="Workspace Active A",
            slug="workspace-active-a",
            description="",
        )
        self.workspace_b = create_workspace_for_user(
            user=self.owner,
            title="Workspace Active B",
            slug="workspace-active-b",
            description="",
        )

    def test_latest_created_workspace_is_active_by_default(self):
        state = WorkspaceUserState.objects.get(user=self.owner)
        self.assertEqual(state.active_workspace_id, self.workspace_b.id)

    def test_activate_endpoint_switches_active_workspace(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-activate", kwargs={"workbook_id": self.workspace_a.id}),
            data={"next_url": reverse("workbook-dashboard")},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        state = WorkspaceUserState.objects.get(user=self.owner)
        self.assertEqual(state.active_workspace_id, self.workspace_a.id)

    def test_dashboard_shows_active_indicator(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workspace Active B")
        self.assertContains(response, "ws-card-active")

    def test_dashboard_links_active_workbooks_to_open_route(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Open")
        self.assertContains(
            response,
            reverse("workbook-open", kwargs={"workbook_id": self.workspace_a.id}),
        )
        self.assertContains(
            response,
            reverse("workbook-open", kwargs={"workbook_id": self.workspace_b.id}),
        )

    def test_dashboard_only_shows_permanent_delete_for_archived_workspaces(self):
        self.client.force_login(self.owner)
        active_response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(active_response.status_code, 200)
        active_html = active_response.content.decode("utf-8")
        self.assertIsNone(re.search(r"<button[^>]*data-open-delete-confirm", active_html))

        self.client.post(
            reverse("workbook-archive", kwargs={"workbook_id": self.workspace_a.id}),
            follow=True,
        )
        archived_response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(archived_response.status_code, 200)
        archived_html = archived_response.content.decode("utf-8")
        self.assertIsNotNone(re.search(r"<button[^>]*data-open-delete-confirm", archived_html))

    def test_dashboard_copy_action_embeds_wizard_prefill_payload(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Copied investigation",
            question_text="Original investigation description",
            scope_json={
                "temporal_scope_option": "custom_range",
                "query_start_date": "2024-01-01",
                "query_end_date": "2024-12-31",
            },
            method_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES, RunType.EXTRACT],
                "run_filter": True,
                "run_themes": True,
                "run_extract": True,
            },
            status=InvestigationStatus.ACTIVE,
        )
        queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={
                "pipeline_plan": [RunType.FILTER, RunType.THEMES, RunType.EXTRACT],
                "search_query": "screening query from run config",
                "filter_df": False,
                "selected_filters": {
                    "coroner": ["Area Coroner"],
                    "area": ["South West"],
                    "receiver": ["NHS England"],
                },
                "seed_topics": "handoffs\nrisk ownership",
                "min_themes": 4,
                "max_themes": 10,
                "extra_theme_instructions": "Prioritise process-level failures.",
                "feature_fields": [
                    {"name": "age_at_death", "description": "Age in years", "type": "integer"},
                    {"name": "had_safeguarding_referral", "description": "Safeguarding referral", "type": "boolean"},
                ],
                "allow_multiple": True,
                "force_assign": True,
                "skip_if_present": False,
                "produce_spans": True,
                "provider": "openrouter",
                "model_name": "gpt-4.1-mini",
                "max_parallel_workers": 3,
            },
        )

        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Copy into new investigation")

        html = response.content.decode("utf-8")
        script_match = re.search(
            rf'<script id="workspace-copy-prefill-{self.workspace_a.id}" type="application/json">(?P<payload>.*?)</script>',
            html,
            re.S,
        )
        self.assertIsNotNone(script_match)
        payload = json.loads(script_match.group("payload"))

        self.assertEqual(payload.get("title"), "Copied investigation")
        self.assertEqual(payload.get("question_text"), "Original investigation description")
        self.assertEqual(payload.get("scope_option"), "custom_range")
        self.assertEqual(payload.get("custom_start_date"), "2024-01-01")
        self.assertEqual(payload.get("custom_end_date"), "2024-12-31")
        self.assertTrue(payload.get("run_filter"))
        self.assertTrue(payload.get("run_themes"))
        self.assertTrue(payload.get("run_extract"))
        self.assertEqual(payload.get("search_query"), "screening query from run config")
        self.assertFalse(payload.get("filter_df"))
        self.assertEqual(payload.get("seed_topics"), "handoffs\nrisk ownership")
        self.assertEqual(payload.get("min_themes"), 4)
        self.assertEqual(payload.get("max_themes"), 10)
        self.assertEqual(payload.get("extra_theme_instructions"), "Prioritise process-level failures.")
        self.assertEqual(payload.get("provider"), "openrouter")
        self.assertEqual(payload.get("model_name"), "gpt-4.1-mini")
        self.assertEqual(payload.get("max_parallel_workers"), 3)
        self.assertNotIn("api_key", payload)
        parsed_features = json.loads(payload.get("feature_fields"))
        self.assertEqual(parsed_features[0]["type"], "decimal")
        self.assertEqual(parsed_features[1]["type"], "boolean")

    def test_dashboard_copy_action_is_available_for_archived_workspaces(self):
        self.client.force_login(self.owner)
        self.client.post(
            reverse("workbook-archive", kwargs={"workbook_id": self.workspace_a.id}),
            follow=True,
        )
        self.client.post(
            reverse("workbook-archive", kwargs={"workbook_id": self.workspace_b.id}),
            follow=True,
        )

        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        self.assertGreaterEqual(html.count("Copy into new investigation"), 2)

    def test_open_workspace_route_sets_active_and_redirects_to_explore(self):
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse("workbook-open", kwargs={"workbook_id": self.workspace_a.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workspace-scoped view of your pipeline result dataset.")
        state = WorkspaceUserState.objects.get(user=self.owner)
        self.assertEqual(state.active_workspace_id, self.workspace_a.id)

    def test_open_workspace_shows_share_button_when_investigation_exists(self):
        create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Shareable Workspace Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse("workbook-open", kwargs={"workbook_id": self.workspace_a.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Share")
        self.assertContains(
            response,
            reverse(
                "workbook-investigation-share-public",
                kwargs={
                    "workbook_id": self.workspace_a.id,
                    "investigation_id": Investigation.objects.get(workspace=self.workspace_a).id,
                },
            ),
        )

    def test_dashboard_shows_worker_offline_banner_without_heartbeat(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Queued investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        RunWorkerHeartbeat.objects.all().delete()
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Worker offline")

    def test_dashboard_hides_worker_offline_banner_without_pending_runs(self):
        RunWorkerHeartbeat.objects.all().delete()
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Worker offline")

    def test_dashboard_hides_stale_banner_when_running_run_is_recently_updated(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Running investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        run.status = RunStatus.RUNNING
        run.save(update_fields=["status", "updated_at"])
        heartbeat = RunWorkerHeartbeat.objects.create(
            worker_id="stale-worker",
            state="claimed",
            last_run=run,
        )
        heartbeat.last_seen_at = timezone.now() - timedelta(seconds=600)
        heartbeat.save(update_fields=["last_seen_at", "updated_at"])

        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Worker heartbeat is stale")

    def test_dashboard_shows_cancel_button_for_running_run(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Pending Run Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        run.status = RunStatus.RUNNING
        run.save(update_fields=["status"])
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cancel")
        self.assertContains(
            response,
            reverse("run-cancel", kwargs={"workbook_id": self.workspace_a.id, "run_id": run.id}),
        )

    def test_dashboard_pending_badge_uses_queue_and_stage_labels(self):
        queued_investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Queued Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        queued_run = queue_run(
            actor=self.owner,
            investigation=queued_investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        queued_run.status = RunStatus.QUEUED
        queued_run.save(update_fields=["status"])

        themes_investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_b,
            title="Themes Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        themes_run = queue_run(
            actor=self.owner,
            investigation=themes_investigation,
            run_type=RunType.THEMES,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        themes_run.status = RunStatus.RUNNING
        themes_run.save(update_fields=["status"])

        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Queued")
        self.assertContains(response, "Finding themes")
        self.assertNotContains(response, ">Loading<")

    def test_dashboard_shows_complete_reports_found_metric(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Complete Investigation",
            question_text="Question prompt shown as card description",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        run.status = RunStatus.SUCCEEDED
        run.save(update_fields=["status"])
        RunArtifact.objects.create(
            run=run,
            workspace=self.workspace_a,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.DB,
            storage_uri="db://artifact",
            metadata_json={"matched_reports": 47},
        )
        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Complete Investigation")
        self.assertContains(response, "Question prompt shown as card description")
        self.assertContains(response, "Reports found")
        self.assertContains(response, ">47<", html=False)

    def test_dashboard_counts_reports_from_latest_themes_dataset_artifact_rows(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Themes Investigation",
            question_text="Theme prompt",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.THEMES,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        run.status = RunStatus.SUCCEEDED
        run.save(update_fields=["status"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("id,title\n1,Report A\n2,Report B\n3,Report C\n")
            csv_path = handle.name
        self.addCleanup(lambda: os.path.exists(csv_path) and os.remove(csv_path))
        RunArtifact.objects.create(
            run=run,
            workspace=self.workspace_a,
            artifact_type=ArtifactType.THEME_ASSIGNMENTS,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=csv_path,
            metadata_json={},
        )

        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        active_rows = response.context["active_rows"]
        row = next(item for item in active_rows if item["workspace"].id == self.workspace_a.id)
        self.assertEqual(row["reports_found_count"], 3)
        self.assertTrue(row["show_reports_found_count"])

    def test_dashboard_last_edited_prefers_newer_investigation_update_over_older_run(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Recency Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        now = timezone.now()
        older_run_time = now - timedelta(hours=1, minutes=20)
        newer_investigation_time = now - timedelta(minutes=20)
        type(run).objects.filter(id=run.id).update(status=RunStatus.SUCCEEDED, updated_at=older_run_time)
        type(investigation).objects.filter(id=investigation.id).update(updated_at=newer_investigation_time)

        self.client.force_login(self.owner)
        response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(response.status_code, 200)
        active_rows = response.context["active_rows"]
        row = next(item for item in active_rows if item["workspace"].id == self.workspace_a.id)
        self.assertEqual(row["last_edited_at"], newer_investigation_time)

    def test_cancelled_runs_are_pruned_after_refresh_with_one_request_grace(self):
        investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace_a,
            title="Cancelled Investigation",
            question_text="Q",
            scope_json={},
            method_json={},
            status=InvestigationStatus.ACTIVE,
        )
        run = queue_run(
            actor=self.owner,
            investigation=investigation,
            run_type=RunType.FILTER,
            input_config_json={"provider": "openai", "model_name": "gpt-4.1-mini"},
        )
        run.status = RunStatus.CANCELLED
        run.cancel_requested_at = timezone.now()
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "cancel_requested_at", "finished_at", "updated_at"])

        self.client.force_login(self.owner)
        session = self.client.session
        session["wb_skip_cancelled_prune_once"] = True
        session.save()

        first_response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(first_response.status_code, 200)
        self.assertTrue(InvestigationRun.objects.filter(id=run.id).exists())

        second_response = self.client.get(reverse("workbook-dashboard"))
        self.assertEqual(second_response.status_code, 200)
        self.assertFalse(InvestigationRun.objects.filter(id=run.id).exists())

    def test_workspace_detail_shows_switcher_with_all_user_workbooks(self):
        self.client.force_login(self.owner)
        response = self.client.get(
            reverse("workbook-detail", kwargs={"workbook_id": self.workspace_b.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your Workbooks")
        self.assertContains(response, "Workspace Active A")
        self.assertContains(response, "Workspace Active B")
        self.assertContains(response, "Switch to this workbook")


class WorkspaceLifecycleControlTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner-lifecycle-ui@example.com", password="x")
        self.editor = User.objects.create_user(email="editor-lifecycle-ui@example.com", password="x")
        self.admin = User.objects.create_superuser(email="admin-lifecycle-ui@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Lifecycle UI Workspace",
            slug="lifecycle-ui-workspace",
            description="",
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

    def test_owner_can_archive_and_restore_via_views(self):
        self.client.force_login(self.owner)
        archive_response = self.client.post(
            reverse("workbook-archive", kwargs={"workbook_id": self.workspace.id}),
            follow=True,
        )
        self.assertEqual(archive_response.status_code, 200)
        self.workspace.refresh_from_db()
        self.assertIsNotNone(self.workspace.archived_at)
        self.assertContains(archive_response, "Archived")
        self.assertContains(archive_response, "Restore")

        restore_response = self.client.post(
            reverse("workbook-restore", kwargs={"workbook_id": self.workspace.id}),
            follow=True,
        )
        self.assertEqual(restore_response.status_code, 200)
        self.workspace.refresh_from_db()
        self.assertIsNone(self.workspace.archived_at)

    def test_owner_can_hard_delete_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workbook-delete", kwargs={"workbook_id": self.workspace.id}),
            data={"reason": "owner attempt"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Workspace.objects.filter(id=self.workspace.id).exists())

    def test_editor_cannot_hard_delete_via_view(self):
        self.client.force_login(self.editor)
        response = self.client.post(
            reverse("workbook-delete", kwargs={"workbook_id": self.workspace.id}),
            data={"reason": "editor attempt"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(Workspace.objects.filter(id=self.workspace.id).exists())

    def test_admin_can_hard_delete_via_view(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("workbook-delete", kwargs={"workbook_id": self.workspace.id}),
            data={"reason": "admin cleanup"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Workspace.objects.filter(id=self.workspace.id).exists())


class WorkspaceViewActivityTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner4@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Activity Workspace",
            slug="activity-workspace",
            description="Activity Desc",
        )

    def test_workspace_detail_bot_view_does_not_update_last_viewed(self):
        self.client.force_login(self.owner)
        self.client.get(
            reverse("workbook-detail", kwargs={"workbook_id": self.workspace.id}),
            HTTP_USER_AGENT="Googlebot/2.1",
        )
        self.workspace.refresh_from_db()
        self.assertIsNone(self.workspace.last_viewed_at)

    def test_workspace_detail_human_view_updates_last_viewed(self):
        self.client.force_login(self.owner)
        self.client.get(
            reverse("workbook-detail", kwargs={"workbook_id": self.workspace.id}),
            HTTP_USER_AGENT="Mozilla/5.0",
        )
        self.workspace.refresh_from_db()
        self.assertIsNotNone(self.workspace.last_viewed_at)


class LifecycleMaintenanceTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner5@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="Lifecycle Workspace",
            slug="lifecycle-workspace",
            description="Lifecycle Desc",
        )
        self.investigation = create_investigation(
            actor=self.owner,
            workspace=self.workspace,
            title="Lifecycle Investigation",
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

    def test_lifecycle_job_expires_stale_artifacts(self):
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri="/tmp/lifecycle-stale.csv",
            metadata_json={},
        )
        old_seen_at = timezone.now() - timedelta(days=400)
        RunArtifact.objects.filter(id=artifact.id).update(
            created_at=old_seen_at,
            last_viewed_at=old_seen_at,
        )

        result = run_lifecycle_maintenance(inactivity_days=365, archive_workspaces=False)
        artifact.refresh_from_db()

        self.assertEqual(result.artifacts_expired, 1)
        self.assertEqual(artifact.status, ArtifactStatus.EXPIRED)
        self.assertIsNotNone(artifact.expires_at)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="run.artifact_expired_inactive",
                target_id=str(artifact.id),
            ).exists()
        )

    def test_lifecycle_job_refreshes_expiry_for_recently_viewed_artifacts(self):
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri="/tmp/lifecycle-recent.csv",
            metadata_json={},
        )
        created_at = timezone.now() - timedelta(days=500)
        recently_viewed_at = timezone.now() - timedelta(days=2)
        RunArtifact.objects.filter(id=artifact.id).update(
            created_at=created_at,
            last_viewed_at=recently_viewed_at,
            expires_at=timezone.now() - timedelta(days=100),
        )

        result = run_lifecycle_maintenance(inactivity_days=365, archive_workspaces=False)
        artifact.refresh_from_db()
        expected_expires_at = recently_viewed_at + timedelta(days=365)

        self.assertEqual(result.artifacts_expired, 0)
        self.assertEqual(result.artifacts_expiry_refreshed, 1)
        self.assertEqual(artifact.status, ArtifactStatus.READY)
        self.assertEqual(artifact.expires_at, expected_expires_at)

    def test_lifecycle_job_archives_stale_workspaces(self):
        stale_at = timezone.now() - timedelta(days=500)
        Workspace.objects.filter(id=self.workspace.id).update(
            created_at=stale_at,
            last_viewed_at=None,
            archived_at=None,
        )

        result = run_lifecycle_maintenance(inactivity_days=365)
        self.workspace.refresh_from_db()

        self.assertEqual(result.workspaces_archived, 1)
        self.assertIsNotNone(self.workspace.archived_at)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="workspace.auto_archived_inactive",
                target_id=str(self.workspace.id),
            ).exists()
        )

    def test_lifecycle_job_dry_run_does_not_write(self):
        stale_at = timezone.now() - timedelta(days=500)
        Workspace.objects.filter(id=self.workspace.id).update(
            created_at=stale_at,
            last_viewed_at=None,
            archived_at=None,
        )
        artifact = RunArtifact.objects.create(
            run=self.run,
            workspace=self.workspace,
            artifact_type=ArtifactType.FILTERED_DATASET,
            status=ArtifactStatus.READY,
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri="/tmp/lifecycle-dry-run.csv",
            metadata_json={},
        )
        RunArtifact.objects.filter(id=artifact.id).update(
            created_at=stale_at,
            last_viewed_at=stale_at,
        )

        result = run_lifecycle_maintenance(inactivity_days=365, dry_run=True)
        self.workspace.refresh_from_db()
        artifact.refresh_from_db()

        self.assertEqual(result.artifacts_expired, 1)
        self.assertEqual(result.workspaces_archived, 1)
        self.assertEqual(artifact.status, ArtifactStatus.READY)
        self.assertIsNone(self.workspace.archived_at)

    def test_lifecycle_job_purges_archived_workspaces_past_retention(self):
        archived_at = timezone.now() - timedelta(days=90)
        Workspace.objects.filter(id=self.workspace.id).update(
            archived_at=archived_at,
            last_viewed_at=None,
        )
        state = WorkspaceUserState.objects.get(user=self.owner)
        state.active_workspace = self.workspace
        state.save(update_fields=["active_workspace", "updated_at"])

        result = run_lifecycle_maintenance(
            inactivity_days=365,
            archive_retention_days=60,
            archive_workspaces=False,
            purge_archived_workspaces=True,
        )

        self.assertEqual(result.workspaces_purged, 1)
        self.assertFalse(Workspace.objects.filter(id=self.workspace.id).exists())
        state.refresh_from_db()
        self.assertIsNone(state.active_workspace_id)
        self.assertTrue(
            AuditEvent.objects.filter(
                action_type="workspace.auto_purged_archived",
                target_id=str(self.workspace.id),
            ).exists()
        )

    def test_lifecycle_management_command_runs(self):
        out = io.StringIO()
        call_command(
            "run_lifecycle_maintenance",
            "--dry-run",
            "--skip-workspace-archive",
            stdout=out,
        )
        joined = out.getvalue()
        self.assertIn("DRY RUN", joined)
        self.assertIn("Artifacts scanned", joined)
