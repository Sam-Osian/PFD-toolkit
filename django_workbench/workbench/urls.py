"""URL routes for the Workbench app."""
from django.urls import path

from . import views

app_name = "workbench"

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.home, name="home"),
    path("explore-pfds/", views.explore, name="explore"),
    path("workbooks/create/", views.workbook_create, name="workbook_create"),
    path(
        "workbooks/<uuid:public_id>/save/",
        views.workbook_save,
        name="workbook_save",
    ),
    path(
        "workbooks/<int:share_number>-<slug:title_slug>/",
        views.workbook_public,
        name="workbook_public",
    ),
    path(
        "workbooks/<int:share_number>-<slug:title_slug>/download/",
        views.workbook_download,
        name="workbook_download",
    ),
    path(
        "workbooks/<int:share_number>-<slug:title_slug>/dataset-panel/",
        views.workbook_dataset_panel,
        name="workbook_dataset_panel",
    ),
    path(
        "workbooks/<int:share_number>-<slug:title_slug>/clone/",
        views.workbook_clone,
        name="workbook_clone",
    ),
    path("filter/", views.filter_page, name="filter"),
    path("analyse-themes/", views.themes_page, name="themes"),
    path("extract-data/", views.extract_page, name="extract"),
    path("for-coders", views.for_coders, name="for_coders_no_slash"),
    path("for-coders/", views.for_coders, name="for_coders"),
    path("for-coders/site/<path:file_path>", views.for_coders_site_file, name="for_coders_site_file"),
    path("for-coders/<path:doc_path>", views.for_coders_page, name="for_coders_page_no_slash"),
    path("for-coders/<path:doc_path>/", views.for_coders_page, name="for_coders_page"),
    path("settings/", views.settings_page, name="settings"),
    path("dataset-panel/", views.dataset_panel, name="dataset_panel"),
]
