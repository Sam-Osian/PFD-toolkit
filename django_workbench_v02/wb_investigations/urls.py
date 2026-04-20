from django.urls import path

from . import views


urlpatterns = [
    path(
        "workbooks/<uuid:workbook_id>/investigations/",
        views.investigation_list,
        name="workbook-investigation-list",
    ),
    path(
        "workbooks/<uuid:workbook_id>/investigations/<uuid:investigation_id>/",
        views.investigation_detail,
        name="workbook-investigation-detail",
    ),
    path(
        "workbooks/<uuid:workbook_id>/investigations/<uuid:investigation_id>/update/",
        views.investigation_update,
        name="workbook-investigation-update",
    ),
    path(
        "workbooks/<uuid:workbook_id>/investigations/<uuid:investigation_id>/wizard/",
        views.investigation_wizard,
        name="workbook-investigation-wizard",
    ),

    path(
        "workspaces/<uuid:workbook_id>/investigations/",
        views.investigation_list,
        name="investigation-list",
    ),
    path(
        "workspaces/<uuid:workbook_id>/investigations/<uuid:investigation_id>/",
        views.investigation_detail,
        name="investigation-detail",
    ),
    path(
        "workspaces/<uuid:workbook_id>/investigations/<uuid:investigation_id>/update/",
        views.investigation_update,
        name="investigation-update",
    ),
    path(
        "workspaces/<uuid:workbook_id>/investigations/<uuid:investigation_id>/wizard/",
        views.investigation_wizard,
        name="investigation-wizard",
    ),
]
