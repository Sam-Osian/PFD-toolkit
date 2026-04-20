from django.urls import path

from . import views


urlpatterns = [
    path(
        "workbooks/<uuid:workbook_id>/action-cache/",
        views.workspace_action_cache,
        name="workbook-action-cache",
    ),
    path(
        "workspaces/<uuid:workbook_id>/action-cache/",
        views.workspace_action_cache,
        name="workspace-action-cache",
    ),
]
