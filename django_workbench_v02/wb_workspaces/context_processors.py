from __future__ import annotations

from django.conf import settings

from .services import get_active_workspace_for_user, get_workspace_llm_setting


def active_workbook(request):
    notice = {
        "enabled": bool(getattr(settings, "USER_NOTICE_ENABLED", False)),
        "show_on_home": bool(getattr(settings, "USER_NOTICE_SHOW_ON_HOME", True)),
        "version": str(getattr(settings, "USER_NOTICE_VERSION", "v1") or "v1"),
        "title": str(getattr(settings, "USER_NOTICE_TITLE", "Notice")).strip(),
        "body": list(getattr(settings, "USER_NOTICE_BODY", ()) or []),
        "button_label": str(getattr(settings, "USER_NOTICE_BUTTON_LABEL", "I understand")).strip(),
    }
    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return {
            "active_workbook": None,
            "active_llm_config": None,
            "user_notice": notice,
        }

    workbook = get_active_workspace_for_user(user=user)
    llm_config = get_workspace_llm_setting(user=user, workspace=workbook) if workbook else None
    return {
        "active_workbook": workbook,
        "active_llm_config": llm_config,
        "user_notice": notice,
    }
