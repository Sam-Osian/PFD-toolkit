from django.urls import path

from . import views


urlpatterns = [
    path("", views.landing, name="landing"),
    path("auth/login/", views.auth_login, name="accounts-login"),
    path("auth/callback/", views.auth_callback, name="accounts-callback"),
    path("auth/logout/", views.auth_logout, name="accounts-logout"),
]
