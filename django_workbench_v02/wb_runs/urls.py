from django.urls import path

from . import views


urlpatterns = [
    path(
        "workbooks/<uuid:workbook_id>/investigations/<uuid:investigation_id>/runs/queue/",
        views.queue_investigation_run,
        name="workbook-run-queue",
    ),
    path(
        "workbooks/<uuid:workbook_id>/runs/<uuid:run_id>/",
        views.run_detail,
        name="workbook-run-detail",
    ),
    path(
        "workbooks/<uuid:workbook_id>/runs/<uuid:run_id>/cancel/",
        views.cancel_run,
        name="workbook-run-cancel",
    ),
    path(
        "workbooks/<uuid:workbook_id>/runs/<uuid:run_id>/artifacts/<uuid:artifact_id>/download/",
        views.download_run_artifact,
        name="workbook-run-artifact-download",
    ),

    path(
        "workspaces/<uuid:workbook_id>/investigations/<uuid:investigation_id>/runs/queue/",
        views.queue_investigation_run,
        name="run-queue",
    ),
    path(
        "workspaces/<uuid:workbook_id>/runs/<uuid:run_id>/",
        views.run_detail,
        name="run-detail",
    ),
    path(
        "workspaces/<uuid:workbook_id>/runs/<uuid:run_id>/cancel/",
        views.cancel_run,
        name="run-cancel",
    ),
    path(
        "workspaces/<uuid:workbook_id>/runs/<uuid:run_id>/artifacts/<uuid:artifact_id>/download/",
        views.download_run_artifact,
        name="run-artifact-download",
    ),
]
