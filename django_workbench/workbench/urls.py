"""URL routes for the Workbench app."""
from django.urls import path

from . import views

app_name = "workbench"

urlpatterns = [
    path("", views.index, name="index"),
]
