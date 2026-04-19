from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_auditlog.models import AuditEvent
from wb_workspaces.models import (
    MembershipAccessMode,
    MembershipRole,
    WorkspaceMembership,
    WorkspaceRevision,
)
from wb_workspaces.services import create_workspace_for_user

from .models import ShareMode, WorkspaceShareLink
from .services import (
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
        self.revision = WorkspaceRevision.objects.create(
            workspace=self.workspace,
            revision_number=1,
            state_json={"k": "v"},
            created_by=self.owner,
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

    def test_create_snapshot_share_fails_without_revision(self):
        WorkspaceRevision.objects.filter(workspace=self.workspace).delete()
        with self.assertRaises(ValidationError):
            create_share_link(
                actor=self.owner,
                workspace=self.workspace,
                mode=ShareMode.SNAPSHOT,
                is_public=True,
            )

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
        WorkspaceRevision.objects.create(
            workspace=self.workspace,
            revision_number=1,
            state_json={"k": "v"},
            created_by=self.owner,
        )

    def test_owner_can_create_share_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workspace-share-create", kwargs={"workspace_id": self.workspace.id}),
            data={"mode": ShareMode.SNAPSHOT, "is_public": "on"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            WorkspaceShareLink.objects.filter(workspace=self.workspace).exists()
        )

    def test_public_share_view_accessible_without_login(self):
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
