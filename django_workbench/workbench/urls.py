"""URL routes for the Workbench app."""
from django.urls import path

from . import views

app_name = "workbench"

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.home, name="home"),
    path("favicon.ico", views.favicon, name="favicon"),
    path("privacy/", views.privacy_policy, name="privacy_policy"),
    path("robots.txt", views.robots_txt, name="robots_txt"),
    path("sitemap.xml", views.sitemap_xml, name="sitemap_xml"),
    path("browse/", views.browse_page, name="browse"),
    path("browse/<slug:collection_slug>/", views.browse_collection_page, name="browse_collection"),
    path(
        "browse/<slug:collection_slug>/dataset-panel/",
        views.browse_collection_dataset_panel,
        name="browse_collection_dataset_panel",
    ),
    path(
        "browse/<slug:collection_slug>/dashboard-data/",
        views.browse_collection_dashboard_data,
        name="browse_collection_dashboard_data",
    ),
    path(
        "browse/<slug:collection_slug>/clone/",
        views.browse_collection_clone,
        name="browse_collection_clone",
    ),
    path("explore-pfds/", views.explore, name="explore"),
    path("network/", views.network_page, name="network"),
    path("network/unlock/", views.network_unlock, name="network_unlock"),
    path("network-data/", views.network_data, name="network_data"),
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
    path("dashboard-data/", views.dashboard_data, name="dashboard_data"),
    path("sse/filter/", views.sse_filter_reports, name="sse_filter"),
    path("sse/themes/", views.sse_discover_themes, name="sse_themes"),
    path("sse/extract/", views.sse_extract_features, name="sse_extract"),
    path(
        "workbooks/<int:share_number>-<slug:title_slug>/dashboard-data/",
        views.workbook_dashboard_data,
        name="workbook_dashboard_data",
    ),
]
