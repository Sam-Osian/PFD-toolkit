from django.urls import path

from . import views


urlpatterns = [
    path(
        "workbooks/<uuid:workbook_id>/shares/create/",
        views.create_workspace_share,
        name="workbook-share-create",
    ),
    path(
        "workbooks/<uuid:workbook_id>/shares/<uuid:share_id>/update/",
        views.update_workspace_share,
        name="workbook-share-update",
    ),
    path(
        "workbooks/<uuid:workbook_id>/shares/<uuid:share_id>/revoke/",
        views.revoke_workspace_share,
        name="workbook-share-revoke",
    ),
    path("s/<uuid:share_id>/copy/", views.copy_share_link_to_workbook_view, name="share-link-copy"),

    path(
        "workspaces/<uuid:workbook_id>/shares/create/",
        views.create_workspace_share,
        name="workspace-share-create",
    ),
    path(
        "workspaces/<uuid:workbook_id>/shares/<uuid:share_id>/update/",
        views.update_workspace_share,
        name="workspace-share-update",
    ),
    path(
        "workspaces/<uuid:workbook_id>/shares/<uuid:share_id>/revoke/",
        views.revoke_workspace_share,
        name="workspace-share-revoke",
    ),
    path("s/<uuid:share_id>/reports/panel/", views.share_reports_panel, name="share-link-reports-panel"),
    path("s/<uuid:share_id>/export.csv", views.export_share_link_csv, name="share-link-export-csv"),
    path("s/<uuid:share_id>/", views.view_share_link, name="share-link-detail"),
]
