from django.urls import path

from . import views


urlpatterns = [
    path("workbooks/", views.dashboard, name="workbook-dashboard"),
    path("workbooks/public/", views.public_workspace_list, name="workbook-public-list"),
    path("workbooks/<uuid:workbook_id>/", views.workspace_detail, name="workbook-detail"),
    path(
        "workbooks/<uuid:workbook_id>/members/add/",
        views.add_member,
        name="workbook-member-add",
    ),
    path(
        "workbooks/<uuid:workbook_id>/members/<uuid:membership_id>/update/",
        views.update_member,
        name="workbook-member-update",
    ),
    path(
        "workbooks/<uuid:workbook_id>/members/<uuid:membership_id>/remove/",
        views.remove_member,
        name="workbook-member-remove",
    ),
    path(
        "workbooks/<uuid:workbook_id>/credentials/save/",
        views.save_credential,
        name="workbook-credential-save",
    ),
    path(
        "workbooks/<uuid:workbook_id>/credentials/remove/",
        views.remove_credential,
        name="workbook-credential-remove",
    ),

    path("workspaces/", views.dashboard, name="workspace-dashboard"),
    path("workspaces/public/", views.public_workspace_list, name="workspace-public-list"),
    path("workspaces/<uuid:workbook_id>/", views.workspace_detail, name="workspace-detail"),
    path(
        "workspaces/<uuid:workbook_id>/members/add/",
        views.add_member,
        name="workspace-member-add",
    ),
    path(
        "workspaces/<uuid:workbook_id>/members/<uuid:membership_id>/update/",
        views.update_member,
        name="workspace-member-update",
    ),
    path(
        "workspaces/<uuid:workbook_id>/members/<uuid:membership_id>/remove/",
        views.remove_member,
        name="workspace-member-remove",
    ),
    path(
        "workspaces/<uuid:workbook_id>/credentials/save/",
        views.save_credential,
        name="workspace-credential-save",
    ),
    path(
        "workspaces/<uuid:workbook_id>/credentials/remove/",
        views.remove_credential,
        name="workspace-credential-remove",
    ),
]
