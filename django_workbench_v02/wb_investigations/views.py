from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from wb_runs.forms import RunQueueForm
from wb_runs.models import InvestigationRun
from wb_workspaces.models import Workspace
from wb_workspaces.permissions import can_edit_workspace, can_view_workspace

from .forms import InvestigationCreateForm, InvestigationUpdateForm
from .models import Investigation
from .services import create_investigation, record_investigation_view, update_investigation


@require_http_methods(["GET", "POST"])
def investigation_list(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")

    can_edit = bool(request.user.is_authenticated and can_edit_workspace(request.user, workspace))
    if request.method == "POST":
        if not request.user.is_authenticated:
            return redirect("accounts-login")
        form = InvestigationCreateForm(request.POST)
        if not can_edit:
            messages.error(request, "You do not have permission to create investigations.")
            return redirect("investigation-list", workspace_id=workspace.id)
        if form.is_valid():
            try:
                create_investigation(
                    actor=request.user,
                    workspace=workspace,
                    title=form.cleaned_data["title"],
                    question_text=form.cleaned_data["question_text"],
                    scope_json=form.cleaned_data["scope_json"],
                    method_json=form.cleaned_data["method_json"],
                    status=form.cleaned_data["status"],
                    request=request,
                )
            except (PermissionDenied, ValidationError) as exc:
                messages.error(request, str(exc))
            else:
                messages.success(request, "Investigation created.")
            return redirect("investigation-list", workspace_id=workspace.id)
    else:
        form = InvestigationCreateForm()

    investigations = Investigation.objects.filter(workspace=workspace).order_by("-updated_at")
    return render(
        request,
        "wb_investigations/investigation_list.html",
        {
            "workspace": workspace,
            "investigations": investigations,
            "can_edit": can_edit,
            "create_form": form,
        },
    )


@require_GET
def investigation_detail(request, workspace_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace", "created_by"),
        id=investigation_id,
        workspace_id=workspace_id,
    )
    if not can_view_workspace(request.user, investigation.workspace):
        return redirect("accounts-login")

    record_investigation_view(investigation=investigation, user=request.user, request=request)

    can_edit = bool(
        request.user.is_authenticated
        and can_edit_workspace(request.user, investigation.workspace)
    )
    update_form = InvestigationUpdateForm(
        initial={
            "title": investigation.title,
            "question_text": investigation.question_text,
            "scope_json": investigation.scope_json,
            "method_json": investigation.method_json,
            "status": investigation.status,
        }
    )
    run_form = RunQueueForm()
    runs = InvestigationRun.objects.filter(investigation=investigation).order_by("-created_at")
    return render(
        request,
        "wb_investigations/investigation_detail.html",
        {
            "investigation": investigation,
            "workspace": investigation.workspace,
            "can_edit": can_edit,
            "update_form": update_form,
            "run_form": run_form,
            "runs": runs,
        },
    )


@login_required
@require_POST
def investigation_update(request, workspace_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workspace_id,
    )
    form = InvestigationUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid investigation update form.")
        return redirect(
            "investigation-detail",
            workspace_id=workspace_id,
            investigation_id=investigation_id,
        )

    try:
        update_investigation(
            actor=request.user,
            investigation=investigation,
            title=form.cleaned_data["title"],
            question_text=form.cleaned_data["question_text"],
            scope_json=form.cleaned_data["scope_json"],
            method_json=form.cleaned_data["method_json"],
            status=form.cleaned_data["status"],
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Investigation updated.")

    return redirect(
        "investigation-detail",
        workspace_id=workspace_id,
        investigation_id=investigation_id,
    )
