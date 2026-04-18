from django.urls import path

from . import views


urlpatterns = [
    path(
        "workspaces/<uuid:workspace_id>/investigations/<uuid:investigation_id>/runs/queue/",
        views.queue_investigation_run,
        name="run-queue",
    ),
    path(
        "workspaces/<uuid:workspace_id>/runs/<uuid:run_id>/",
        views.run_detail,
        name="run-detail",
    ),
    path(
        "workspaces/<uuid:workspace_id>/runs/<uuid:run_id>/cancel/",
        views.cancel_run,
        name="run-cancel",
    ),
    path(
        "workspaces/<uuid:workspace_id>/runs/<uuid:run_id>/artifacts/<uuid:artifact_id>/download/",
        views.download_run_artifact,
        name="run-artifact-download",
    ),
]
