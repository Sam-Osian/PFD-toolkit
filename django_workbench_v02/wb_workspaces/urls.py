from django.urls import path

from . import views


urlpatterns = [
    path("workspaces/", views.dashboard, name="workspace-dashboard"),
    path("workspaces/public/", views.public_workspace_list, name="workspace-public-list"),
    path("workspaces/<uuid:workspace_id>/", views.workspace_detail, name="workspace-detail"),
]
