from django.urls import path

from . import views


urlpatterns = [
    path(
        "workspaces/<uuid:workspace_id>/shares/create/",
        views.create_workspace_share,
        name="workspace-share-create",
    ),
    path(
        "workspaces/<uuid:workspace_id>/shares/<uuid:share_id>/update/",
        views.update_workspace_share,
        name="workspace-share-update",
    ),
    path(
        "workspaces/<uuid:workspace_id>/shares/<uuid:share_id>/revoke/",
        views.revoke_workspace_share,
        name="workspace-share-revoke",
    ),
    path("s/<uuid:share_id>/", views.view_share_link, name="share-link-detail"),
]
