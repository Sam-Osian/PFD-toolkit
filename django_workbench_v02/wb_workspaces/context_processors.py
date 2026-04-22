from __future__ import annotations

from .permissions import can_edit_workspace
from .services import get_active_workspace_for_user, get_workspace_llm_setting


def active_workbook(request):
    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return {
            "active_workbook": None,
            "can_start_over_active_workbook": False,
            "active_llm_config": None,
        }

    workbook = get_active_workspace_for_user(user=user)
    llm_config = get_workspace_llm_setting(user=user, workspace=workbook) if workbook else None
    return {
        "active_workbook": workbook,
        "can_start_over_active_workbook": bool(workbook and can_edit_workspace(user, workbook)),
        "active_llm_config": llm_config,
    }
