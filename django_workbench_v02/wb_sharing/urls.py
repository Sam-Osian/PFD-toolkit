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
    path("s/<uuid:share_id>/", views.view_share_link, name="share-link-detail"),
]
