from django.urls import path

from . import views


urlpatterns = [
    path("", views.landing, name="landing"),
    path("explore/", views.explore, name="explore"),
    path("explore/reports-panel/", views.explore_reports_panel, name="explore-reports-panel"),
    path("explore/export.csv", views.explore_export_csv, name="explore-export-csv"),
    path("about/", views.about, name="about"),
    path("research/", views.research, name="research"),
    path("services/", views.services, name="services"),
    path("privacy/", views.privacy_policy, name="privacy-policy"),
    path("cookies/", views.cookie_policy, name="cookie-policy"),
    path("settings/llm/", views.llm_config, name="llm-config"),
    path("auth/login/", views.auth_login, name="accounts-login"),
    path("auth/callback/", views.auth_callback, name="accounts-callback"),
    path("auth/logout/", views.auth_logout, name="accounts-logout"),
]
