from django.urls import path

from . import views


urlpatterns = [
    path("ops/", views.approvals, name="ops-dashboard"),
    path("ops/approvals/", views.approvals, name="ops-approvals"),
    path("ops/approvals/<uuid:run_id>/action/", views.approval_action, name="ops-approval-action"),
    path("ops/failures/", views.failures, name="ops-failures"),
    path("ops/workers/", views.workers, name="ops-workers"),
    path("ops/users/", views.user_list, name="ops-users"),
    path("ops/users/<int:user_id>/", views.user_detail, name="ops-user-detail"),
    path("ops/workspaces/", views.workspace_list, name="ops-workspaces"),
    path("ops/workspaces/<uuid:workspace_id>/", views.workspace_detail, name="ops-workspace-detail"),
    path(
        "ops/workspaces/<uuid:workspace_id>/runs/<uuid:run_id>/configure/",
        views.configure_pending_run,
        name="ops-run-configure",
    ),
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
