from django.core.exceptions import PermissionDenied, ValidationError
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from wb_auditlog.models import AuditEvent

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceMembership, WorkspaceVisibility
from .permissions import can_edit_workspace, can_manage_members, can_run_workflows, can_view_workspace
from .services import (
    add_workspace_member,
    create_workspace_for_user,
    remove_workspace_member,
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


class WorkspaceMemberViewsTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="owner3@example.com", password="x")
        self.editor = User.objects.create_user(email="editor3@example.com", password="x")
        self.workspace = create_workspace_for_user(
            user=self.owner,
            title="View Workspace",
            slug="view-workspace",
            description="View Desc",
        )

    def test_owner_can_add_member_via_view(self):
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("workspace-member-add", kwargs={"workspace_id": self.workspace.id}),
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
                "workspace-member-update",
                kwargs={
                    "workspace_id": self.workspace.id,
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
