from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

from wb_workspaces.models import Workspace
from wb_workspaces.permissions import can_manage_members

from .models import ActionCacheEvent


@login_required
@require_GET
def workspace_action_cache(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not (request.user.is_superuser or can_manage_members(request.user, workspace)):
        raise PermissionDenied("You do not have permission to inspect workbook action cache.")

    events = (
        ActionCacheEvent.objects.select_related("user")
        .filter(workspace=workspace)
        .order_by("-created_at")[:200]
    )
    return render(
        request,
        "wb_auditlog/action_cache_list.html",
        {"workspace": workspace, "events": events},
    )
