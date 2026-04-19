from django.urls import path

from . import views


urlpatterns = [
    path("workspaces/", views.dashboard, name="workspace-dashboard"),
    path("workspaces/public/", views.public_workspace_list, name="workspace-public-list"),
    path("workspaces/<uuid:workspace_id>/", views.workspace_detail, name="workspace-detail"),
    path(
        "workspaces/<uuid:workspace_id>/members/add/",
        views.add_member,
        name="workspace-member-add",
    ),
    path(
        "workspaces/<uuid:workspace_id>/members/<uuid:membership_id>/update/",
        views.update_member,
        name="workspace-member-update",
    ),
    path(
        "workspaces/<uuid:workspace_id>/members/<uuid:membership_id>/remove/",
        views.remove_member,
        name="workspace-member-remove",
    ),
    path(
        "workspaces/<uuid:workspace_id>/credentials/save/",
        views.save_credential,
        name="workspace-credential-save",
    ),
    path(
        "workspaces/<uuid:workspace_id>/credentials/remove/",
        views.remove_credential,
        name="workspace-credential-remove",
    ),
]
