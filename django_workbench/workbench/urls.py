"""URL routes for the Workbench app."""
from django.urls import path

from . import views

app_name = "workbench"

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.home, name="home"),
    path("explore-pfds/", views.explore, name="explore"),
    path("settings/", views.settings_page, name="settings"),
]
