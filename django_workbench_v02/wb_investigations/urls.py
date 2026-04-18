from django.urls import path

from . import views


urlpatterns = [
    path(
        "workspaces/<uuid:workspace_id>/investigations/",
        views.investigation_list,
        name="investigation-list",
    ),
    path(
        "workspaces/<uuid:workspace_id>/investigations/<uuid:investigation_id>/",
        views.investigation_detail,
        name="investigation-detail",
    ),
    path(
        "workspaces/<uuid:workspace_id>/investigations/<uuid:investigation_id>/update/",
        views.investigation_update,
        name="investigation-update",
    ),
]
