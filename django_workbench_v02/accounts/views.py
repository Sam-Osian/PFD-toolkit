from __future__ import annotations

import secrets
from urllib.parse import urlencode

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET

from .services import normalize_auth0_profile, sync_user_from_auth0

def _require_auth0_settings() -> bool:
    return bool(settings.AUTH0_DOMAIN and settings.AUTH0_CLIENT_ID and settings.AUTH0_CLIENT_SECRET)


@require_GET
def auth_login(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("workbook-dashboard")

    if not _require_auth0_settings():
        return HttpResponseBadRequest("Auth0 is not configured on this environment.")

    next_url = request.GET.get("next", reverse("workbook-dashboard"))
    if not url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        next_url = reverse("workbook-dashboard")

    state = secrets.token_urlsafe(32)
    request.session["auth0_oauth_state"] = state
    request.session["auth0_next"] = next_url

    params = {
        "response_type": "code",
        "client_id": settings.AUTH0_CLIENT_ID,
        "redirect_uri": settings.AUTH0_CALLBACK_URL,
        "scope": settings.AUTH0_SCOPES,
        "state": state,
    }
    authorize_url = f"https://{settings.AUTH0_DOMAIN}/authorize?{urlencode(params)}"
    return redirect(authorize_url)


@require_GET
def auth_callback(request: HttpRequest) -> HttpResponse:
    if not _require_auth0_settings():
        return HttpResponseBadRequest("Auth0 is not configured on this environment.")

    expected_state = request.session.pop("auth0_oauth_state", None)
    received_state = request.GET.get("state")
    if not expected_state or expected_state != received_state:
        return HttpResponseBadRequest("Invalid or missing OAuth state.")

    error = request.GET.get("error")
    if error:
        description = request.GET.get("error_description", "Unknown Auth0 error")
        messages.error(request, f"Auth0 login failed: {description}")
        return redirect("accounts-login")

    code = request.GET.get("code")
    if not code:
        return HttpResponseBadRequest("Missing OAuth authorization code.")

    token_response = requests.post(
        f"https://{settings.AUTH0_DOMAIN}/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": settings.AUTH0_CLIENT_ID,
            "client_secret": settings.AUTH0_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.AUTH0_CALLBACK_URL,
        },
        timeout=15,
    )
    if token_response.status_code != 200:
        return HttpResponseBadRequest("Auth0 token exchange failed.")

    tokens = token_response.json()
    access_token = tokens.get("access_token")
    if not access_token:
        return HttpResponseBadRequest("Missing Auth0 access token.")

    userinfo_response = requests.get(
        f"https://{settings.AUTH0_DOMAIN}/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if userinfo_response.status_code != 200:
        return HttpResponseBadRequest("Unable to fetch Auth0 user profile.")

    profile = normalize_auth0_profile(userinfo_response.json())
    if not profile.email:
        return HttpResponseBadRequest("Auth0 profile did not include an email address.")

    user = sync_user_from_auth0(profile)
    if not user.is_active:
        return HttpResponseBadRequest("Account is inactive. Contact the administrator.")

    login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    next_url = request.session.pop("auth0_next", reverse("workbook-dashboard"))
    if not url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        next_url = reverse("workbook-dashboard")
    return redirect(next_url)


@require_GET
def auth_logout(request: HttpRequest) -> HttpResponse:
    logout(request)

    if not settings.AUTH0_DOMAIN or not settings.AUTH0_CLIENT_ID:
        return redirect(settings.LOGOUT_REDIRECT_URL)

    params = {
        "client_id": settings.AUTH0_CLIENT_ID,
        "returnTo": settings.AUTH0_POST_LOGOUT_REDIRECT_URI,
    }
    return redirect(f"https://{settings.AUTH0_DOMAIN}/v2/logout?{urlencode(params)}")


@require_GET
def landing(request: HttpRequest) -> HttpResponse:
    return render(request, "accounts/landing.html")


@require_GET
def admin_login_proxy(request: HttpRequest) -> HttpResponse:
    admin_index = reverse("admin:index")
    return redirect(f"{reverse('accounts-login')}?next={admin_index}")


@require_GET
def explore(request: HttpRequest) -> HttpResponse:
    return render(request, "accounts/explore.html")


@require_GET
def about(request: HttpRequest) -> HttpResponse:
    return render(request, "accounts/about.html")


@require_GET
def research(request: HttpRequest) -> HttpResponse:
    return render(request, "accounts/research.html")


@require_GET
def llm_config(request: HttpRequest) -> HttpResponse:
    if not request.user.is_authenticated:
        next_url = reverse("llm-config")
        return redirect(f"{reverse('accounts-login')}?next={next_url}")

    from wb_workspaces.services import get_active_workspace_for_user

    active_workspace = get_active_workspace_for_user(user=request.user)
    if active_workspace is not None:
        detail_url = reverse("workbook-detail", kwargs={"workbook_id": active_workspace.id})
        return redirect(f"{detail_url}#llm-credentials")

    messages.info(request, "Create a workbook first, then add your LLM credential there.")
    return redirect("workbook-dashboard")
