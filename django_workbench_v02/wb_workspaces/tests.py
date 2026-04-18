from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership, WorkspaceVisibility
from .permissions import can_edit_workspace, can_manage_members, can_run_workflows, can_view_workspace
from .services import create_workspace_for_user


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
    def test_create_workspace_for_user_creates_owner_membership(self):
        user = User.objects.create_user(email="new-owner@example.com", password="x")
        workspace = create_workspace_for_user(
            user=user,
            title="New Workspace",
            slug="new-workspace",
            description="Desc",
        )

        membership = WorkspaceMembership.objects.get(workspace=workspace, user=user)
        self.assertEqual(membership.role, MembershipRole.OWNER)
        self.assertEqual(membership.access_mode, MembershipAccessMode.EDIT)
        self.assertTrue(membership.can_manage_members)
        self.assertTrue(membership.can_manage_shares)
