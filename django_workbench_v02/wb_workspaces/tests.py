from datetime import timedelta
import io

from django.core.exceptions import PermissionDenied, ValidationError
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_runs.models import ArtifactStatus, ArtifactStorageBackend, ArtifactType, RunArtifact, RunType
from wb_runs.services import queue_run

from .lifecycle import run_lifecycle_maintenance
from .models import (
    MembershipAccessMode,
    MembershipRole,
    Workspace,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceVisibility,
)
from .permissions import can_edit_workspace, can_manage_members, can_run_workflows, can_view_workspace
from .services import (
    add_workspace_member,
    create_workspace_for_user,
    remove_workspace_member,
    resolve_workspace_credential,
    upsert_workspace_credential,
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
