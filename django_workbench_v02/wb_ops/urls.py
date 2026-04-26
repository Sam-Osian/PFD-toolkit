from django.urls import path

from . import views


urlpatterns = [
    path("ops/", views.dashboard, name="ops-dashboard"),
    path("ops/users/", views.user_list, name="ops-users"),
    path("ops/users/<int:user_id>/", views.user_detail, name="ops-user-detail"),
    path("ops/workspaces/<uuid:workspace_id>/", views.workspace_detail, name="ops-workspace-detail"),
    path(
        "ops/workspaces/<uuid:workspace_id>/exclude-row/",
        views.exclude_workspace_row,
        name="ops-workspace-exclude-row",
    ),
    path(
        "ops/workspaces/<uuid:workspace_id>/restore-exclusion/<uuid:exclusion_id>/",
        views.restore_workspace_exclusion,
        name="ops-workspace-restore-exclusion",
    ),
]
