from django.contrib.auth import get_user_model
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from wb_investigations.models import Investigation
from wb_sharing.forms import ShareLinkCreateForm
from wb_sharing.models import ShareMode, WorkspaceShareLink

from .activity import is_human_view_request, should_update_last_viewed
from .forms import (
    WorkspaceCreateForm,
    WorkspaceCredentialDeleteForm,
    WorkspaceCredentialUpsertForm,
    WorkspaceMemberAddForm,
    WorkspaceReportExclusionCreateForm,
    WorkspaceMemberUpdateForm,
)
from .models import (
    MembershipAccessMode,
    MembershipRole,
    Workspace,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceReportExclusion,
    WorkspaceVisibility,
)
from .permissions import can_manage_members, can_manage_shares, can_view_workspace
from .permissions import can_edit_workspace
from .services import (
    WorkspaceCredentialValidationError,
    WorkspaceLifecycleError,
    WorkspaceMembershipError,
    WorkspaceReportExclusionError,
    add_workspace_member,
    create_workspace_for_user,
    delete_workspace_credential,
    delete_workspace_immediately,
    get_active_workspace_for_user,
    remove_workspace_member,
    archive_workspace,
    restore_workspace,
    set_active_workspace_for_user,
    restore_workspace_report_exclusion,
    upsert_workspace_report_exclusion,
    upsert_workspace_credential,
    update_workspace_member,
)
from .revisions import (
    WorkspaceRevisionError,
    redo_workspace_revision,
    revert_workspace_reports,
    start_over_workspace_state,
    undo_workspace_revision,
)


User = get_user_model()


@login_required
@require_http_methods(["GET", "POST"])
def dashboard(request):
    if request.method == "POST":
        form = WorkspaceCreateForm(request.POST)
        if form.is_valid():
            workspace = create_workspace_for_user(
                user=request.user,
                title=form.cleaned_data["title"],
                slug=form.cleaned_data["slug"],
                description=form.cleaned_data["description"],
                request=request,
            )
            workspace.visibility = form.cleaned_data["visibility"]
            workspace.is_listed = form.cleaned_data["is_listed"]
            workspace.save(update_fields=["visibility", "is_listed", "updated_at"])
            messages.success(request, f"Workbook '{workspace.title}' created.")
            return redirect("workbook-detail", workbook_id=workspace.id)
    else:
        form = WorkspaceCreateForm()

    memberships = (
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=request.user)
        .order_by("-workspace__updated_at")
    )
    memberships_list = list(memberships)
    dashboard_rows: list[dict] = []
    for membership in memberships_list:
        workspace = membership.workspace
        can_restore = bool(request.user.is_superuser) or (
            membership.role == MembershipRole.OWNER
            and membership.access_mode == MembershipAccessMode.EDIT
            and membership.can_manage_members
        )
        dashboard_rows.append(
            {
                "membership": membership,
                "workspace": workspace,
                "is_archived": workspace.archived_at is not None,
                "can_restore": can_restore,
            }
        )
    active_workspace = get_active_workspace_for_user(user=request.user)
    if active_workspace is None and memberships_list:
        for membership in memberships_list:
            if membership.workspace.archived_at is None:
                active_workspace = membership.workspace
                set_active_workspace_for_user(
                    user=request.user,
                    workspace=active_workspace,
                    request=request,
                )
                break
    active_workspace_id = str(active_workspace.id) if active_workspace is not None else ""

    return render(
        request,
        "wb_workspaces/dashboard.html",
        {
            "memberships": memberships_list,
            "active_rows": [row for row in dashboard_rows if not row["is_archived"]],
            "archived_rows": [row for row in dashboard_rows if row["is_archived"]],
            "form": form,
            "active_workspace_id": active_workspace_id,
        },
    )


@login_required
@require_POST
def activate_workspace(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    try:
        set_active_workspace_for_user(user=request.user, workspace=workspace, request=request)
    except WorkspaceLifecycleError as exc:
        messages.error(request, str(exc))
        return redirect("workbook-dashboard")
    messages.success(request, f"Active workbook set to '{workspace.title}'.")
    next_url = str(request.POST.get("next_url") or "").strip()
    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-dashboard")


@login_required
@require_POST
def archive_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        archive_workspace(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceLifecycleError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook archived. It will be auto-deleted after 60 days.")
    return redirect("workbook-dashboard")


@login_required
@require_POST
def restore_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        restore_workspace(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceLifecycleError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook restored.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@login_required
@require_POST
def delete_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    reason = str(request.POST.get("reason") or "").strip()
    try:
        delete_workspace_immediately(
            actor=request.user,
            workspace=workspace,
            reason=reason,
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
        return redirect("workbook-detail", workbook_id=workspace.id)
    messages.success(request, "Workbook permanently deleted.")
    return redirect("workbook-dashboard")


@require_GET
def workspace_detail(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")
    if request.user.is_authenticated and workspace.archived_at is None:
        set_active_workspace_for_user(user=request.user, workspace=workspace, request=request)

    now = timezone.now()
    if is_human_view_request(request=request) and should_update_last_viewed(
        existing_last_viewed_at=workspace.last_viewed_at,
        now=now,
    ):
        workspace.last_viewed_at = now
        workspace.save(update_fields=["last_viewed_at"])

    membership = None
    if request.user.is_authenticated:
        membership = WorkspaceMembership.objects.filter(
            workspace=workspace, user=request.user
        ).first()

    manage_members_allowed = bool(
        request.user.is_authenticated and can_manage_members(request.user, workspace)
    )
    can_edit_state = bool(
        request.user.is_authenticated and can_edit_workspace(request.user, workspace)
    )
    manage_shares_allowed = bool(
        request.user.is_authenticated and can_manage_shares(request.user, workspace)
    )
    memberships = WorkspaceMembership.objects.filter(workspace=workspace).select_related("user")
    user_workbook_memberships = WorkspaceMembership.objects.none()
    if request.user.is_authenticated:
        user_workbook_memberships = (
            WorkspaceMembership.objects.select_related("workspace")
            .filter(user=request.user)
            .order_by("-workspace__updated_at")
        )
    active_workspace = get_active_workspace_for_user(user=request.user) if request.user.is_authenticated else None
    active_workspace_id = str(active_workspace.id) if active_workspace is not None else ""
    add_form = WorkspaceMemberAddForm(initial={"can_run_workflows": True})
    credential_form = WorkspaceCredentialUpsertForm()
    credential_delete_form = WorkspaceCredentialDeleteForm()
    user_credentials = WorkspaceCredential.objects.none()
    can_manage_credentials = bool(
        request.user.is_authenticated and membership and membership.can_run_workflows
    ) or bool(request.user.is_authenticated and request.user.is_superuser)
    if request.user.is_authenticated:
        user_credentials = WorkspaceCredential.objects.filter(
            workspace=workspace,
            user=request.user,
        ).order_by("provider")
    share_links = WorkspaceShareLink.objects.filter(workspace=workspace).order_by("-created_at")
    investigations = Investigation.objects.filter(workspace=workspace).order_by("-updated_at")
    share_create_form = ShareLinkCreateForm(
        initial={"mode": ShareMode.SNAPSHOT, "is_public": True}
    )
    report_exclusions = WorkspaceReportExclusion.objects.filter(workspace=workspace)

    return render(
        request,
        "wb_workspaces/workspace_detail.html",
        {
            "workspace": workspace,
            "membership": membership,
            "memberships": memberships,
            "user_workbook_memberships": user_workbook_memberships,
            "active_workspace_id": active_workspace_id,
            "manage_members_allowed": manage_members_allowed,
            "can_edit_state": can_edit_state,
            "manage_shares_allowed": manage_shares_allowed,
            "add_form": add_form,
            "credential_form": credential_form,
            "credential_delete_form": credential_delete_form,
            "user_credentials": user_credentials,
            "can_manage_credentials": can_manage_credentials,
            "share_links": share_links,
            "investigations": investigations,
            "share_create_form": share_create_form,
            "report_exclusions": report_exclusions,
            "role_choices": MembershipRole.choices,
            "access_mode_choices": MembershipAccessMode.choices,
            "share_mode_choices": ShareMode.choices,
            "current_revision": workspace.current_revision,
        },
    )


@login_required
@require_POST
def undo_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        undo_workspace_revision(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Undid workbook state to previous revision.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@login_required
@require_POST
def redo_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        redo_workspace_revision(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Redid workbook state to next revision.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@login_required
@require_POST
def start_over_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        start_over_workspace_state(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook reset to baseline revision.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@login_required
@require_POST
def revert_workspace_reports_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        revert_workspace_reports(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Excluded reports have been reverted.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@require_GET
def public_workspace_list(request):
    workspaces = Workspace.objects.filter(
        visibility=WorkspaceVisibility.PUBLIC, is_listed=True, archived_at__isnull=True
    ).order_by("-updated_at")
    return render(
        request,
        "wb_workspaces/public_workspace_list.html",
        {"workspaces": workspaces},
    )


@login_required
@require_POST
def add_member(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    form = WorkspaceMemberAddForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member form submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    target_email = form.cleaned_data["email"].strip().lower()
    target_user = User.objects.filter(email__iexact=target_email).first()
    if target_user is None:
        messages.error(request, f"No account exists for {target_email}.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    try:
        add_workspace_member(
            actor=request.user,
            workspace=workspace,
            target_user=target_user,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{target_user.email} added to workbook.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def update_member(request, workbook_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    form = WorkspaceMemberUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member update form submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    try:
        update_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{membership.user.email} membership updated.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def remove_member(request, workbook_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    try:
        remove_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook membership removed.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def save_credential(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    membership = WorkspaceMembership.objects.filter(
        workspace=workspace,
        user=request.user,
    ).first()
    if not request.user.is_superuser and (membership is None or not membership.can_run_workflows):
        raise PermissionDenied("You do not have permission to manage credentials in this workbook.")

    form = WorkspaceCredentialUpsertForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid credential submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    try:
        credential = upsert_workspace_credential(
            actor=request.user,
            workspace=workspace,
            provider=form.cleaned_data["provider"],
            api_key=form.cleaned_data["api_key"],
            base_url=form.cleaned_data["base_url"],
            request=request,
        )
    except (WorkspaceCredentialValidationError, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(
            request,
            f"Saved {credential.provider} credential ending in {credential.key_last4}.",
        )
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def remove_credential(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    membership = WorkspaceMembership.objects.filter(
        workspace=workspace,
        user=request.user,
    ).first()
    if not request.user.is_superuser and (membership is None or not membership.can_run_workflows):
        raise PermissionDenied("You do not have permission to manage credentials in this workbook.")

    form = WorkspaceCredentialDeleteForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid credential delete submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)

    deleted = delete_workspace_credential(
        actor=request.user,
        workspace=workspace,
        provider=form.cleaned_data["provider"],
        request=request,
    )
    if deleted:
        messages.success(request, f"Deleted {form.cleaned_data['provider']} credential.")
    else:
        messages.warning(request, f"No {form.cleaned_data['provider']} credential found.")
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def exclude_report(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    form = WorkspaceReportExclusionCreateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid excluded-report submission.")
        return redirect("workbook-detail", workbook_id=workbook_id)
    next_url = str(form.cleaned_data.get("next_url") or "").strip()

    try:
        upsert_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            report_identity=form.cleaned_data["report_identity"],
            reason=form.cleaned_data["reason"],
            report_title=form.cleaned_data["report_title"],
            report_date=form.cleaned_data["report_date"],
            report_url=form.cleaned_data["report_url"],
            request=request,
        )
    except (WorkspaceReportExclusionError, PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Report excluded from this workbook.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-detail", workbook_id=workbook_id)


@login_required
@require_POST
def restore_excluded_report(request, workbook_id, exclusion_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    exclusion = get_object_or_404(
        WorkspaceReportExclusion,
        id=exclusion_id,
        workspace=workspace,
    )
    next_url = str(request.POST.get("next_url") or "").strip()
    try:
        restore_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            exclusion=exclusion,
            request=request,
        )
    except (WorkspaceReportExclusionError, PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Excluded report restored.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-detail", workbook_id=workbook_id)
