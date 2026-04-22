from __future__ import annotations

import secrets
import logging
from math import ceil
from urllib.parse import urlencode
from io import StringIO

import pandas as pd
import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, QueryDict
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET

from .services import normalize_auth0_profile, sync_user_from_auth0
from wb_collections.services import (
    collection_cards,
    build_explore_metrics,
    load_collections_dataset,
    reports_for_collection,
    _parse_report_dates,
    _split_receivers,
    _normalise_dashboard_value,
    _theme_collection_map_from_reports,
    _network_truthy,
)

logger = logging.getLogger(__name__)

EXPLORE_AI_FILTER_EXCLUDED_SLUGS = {
    "wales",
    "nhs",
    "gov_department",
    "prisons",
    "health_regulators",
}
EXPLORE_REPORTS_PAGE_SIZE = 15
EXPLORE_FILTERS_SESSION_KEY = "explore_filters_v1"
EXPLORE_SHOW_REPORTS_SESSION_KEY = "explore_show_reports_v1"


def _dedupe_values(raw_values: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(cleaned)
    return values


def _parse_explore_request_params(request: HttpRequest) -> dict[str, object]:
    filter_keys = {"q", "ai_filter", "receiver", "area", "date_start", "date_end"}
    has_explicit_filters = any(key in request.GET for key in filter_keys)

    if has_explicit_filters:
        return {
            "query": str(request.GET.get("q") or "").strip(),
            "ai_filter": str(request.GET.get("ai_filter") or "custom").strip() or "custom",
            "selected_receivers": _dedupe_values(request.GET.getlist("receiver")),
            "selected_areas": _dedupe_values(request.GET.getlist("area")),
            "date_start_raw": str(request.GET.get("date_start") or "").strip(),
            "date_end_raw": str(request.GET.get("date_end") or "").strip(),
        }

    session_payload = request.session.get(EXPLORE_FILTERS_SESSION_KEY, {})
    if not isinstance(session_payload, dict):
        session_payload = {}

    return {
        "query": str(session_payload.get("query") or "").strip(),
        "ai_filter": str(session_payload.get("ai_filter") or "custom").strip() or "custom",
        "selected_receivers": _dedupe_values(
            [str(value or "").strip() for value in session_payload.get("selected_receivers", [])]
            if isinstance(session_payload.get("selected_receivers", []), list)
            else []
        ),
        "selected_areas": _dedupe_values(
            [str(value or "").strip() for value in session_payload.get("selected_areas", [])]
            if isinstance(session_payload.get("selected_areas", []), list)
            else []
        ),
        "date_start_raw": str(session_payload.get("date_start_raw") or "").strip(),
        "date_end_raw": str(session_payload.get("date_end_raw") or "").strip(),
    }


def _coerce_positive_int(raw_value: str | None, *, default: int = 1) -> int:
    try:
        value = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _explore_shared_querydict(
    *,
    query: str,
    ai_filter: str,
    date_start: str,
    date_end: str,
    selected_receivers: list[str],
    selected_areas: list[str],
    page: int | None = None,
) -> QueryDict:
    payload = QueryDict(mutable=True)
    if query:
        payload["q"] = query
    if ai_filter:
        payload["ai_filter"] = ai_filter
    if date_start:
        payload["date_start"] = date_start
    if date_end:
        payload["date_end"] = date_end
    for value in selected_receivers:
        payload.appendlist("receiver", value)
    for value in selected_areas:
        payload.appendlist("area", value)
    if page is not None:
        payload["page"] = str(max(1, int(page)))
    return payload


def _explore_column_label(column_name: str) -> str:
    label_map = {
        "id": "ID",
        "url": "URL",
        "report_url": "URL",
        "response_date": "Response Date",
    }
    if column_name in label_map:
        return label_map[column_name]
    return str(column_name).replace("_", " ").title()


def _explore_ordered_columns(filtered_reports: pd.DataFrame) -> list[str]:
    preferred_columns = (
        "id",
        "url",
        "date",
        "coroner",
        "area",
        "receiver",
        "title",
        "investigation",
        "circumstances",
        "concerns",
        "actions",
        "response",
        "response_date",
    )
    internal_prefixes = ("theme_", "collection_")
    internal_exact = {"_score", "__report_identity"}

    ordered_columns: list[str] = []
    for column in preferred_columns:
        if column in filtered_reports.columns and column not in ordered_columns:
            ordered_columns.append(column)
    for column in filtered_reports.columns:
        name = str(column)
        if name in internal_exact:
            continue
        if any(name.startswith(prefix) for prefix in internal_prefixes):
            continue
        if name not in ordered_columns:
            ordered_columns.append(name)
    return ordered_columns


def _explore_report_rows(
    *,
    reports_df: pd.DataFrame,
    ordered_columns: list[str],
) -> list[dict[str, object]]:
    if reports_df.empty:
        return []

    preview_df = reports_df[ordered_columns].copy()
    preview_records = preview_df.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    date_columns = {"date", "response_date"}

    for row in preview_records:
        cells: list[str] = []
        for column in ordered_columns:
            value = row.get(column)
            if column in date_columns:
                parsed = _parse_report_dates(pd.Series([value])).iloc[0]
                if pd.notna(parsed):
                    cells.append(parsed.date().isoformat())
                    continue
            cells.append(_normalise_dashboard_value(value))
        rows.append({"cells": cells})
    return rows


def _build_explore_payload(
    *,
    query: str,
    ai_filter: str,
    selected_receivers: list[str],
    selected_areas: list[str],
    date_start_raw: str,
    date_end_raw: str,
    request: HttpRequest | None = None,
    emit_messages: bool = False,
    force_collection_scope: bool = False,
) -> dict[str, object]:
    explore = {
        "query": query,
        "has_query": bool(query),
        "total_reports": 0,
        "scope_reports": 0,
        "date_range_label": "Date range unavailable",
        "scope_label": "All reports",
        "top_areas": [],
        "top_receivers": [],
        "top_themes": [],
        "selected_ai_filter": ai_filter,
        "selected_receivers": selected_receivers,
        "selected_areas": selected_areas,
        "date_start": date_start_raw,
        "date_end": date_end_raw,
        "ai_filter_options": [],
        "receiver_options": [],
        "area_options": [],
    }

    scoped_reports = pd.DataFrame()
    dataset_available = False

    try:
        reports_df = load_collections_dataset(force_refresh=False)
        ai_filter_options = [
            {
                "slug": str(card.get("slug") or "").strip(),
                "title": str(card.get("title") or "").strip(),
            }
            for card in collection_cards(reports_df)
            if str(card.get("slug") or "").strip()
            and str(card.get("slug") or "").strip() != "custom-search"
            and str(card.get("slug") or "").strip() not in EXPLORE_AI_FILTER_EXCLUDED_SLUGS
        ]
        allowed_slugs = {item["slug"] for item in ai_filter_options}
        if not force_collection_scope and ai_filter not in allowed_slugs:
            ai_filter = "custom"

        receiver_counter: dict[str, int] = {}
        for raw in reports_df.get("receiver", []):
            for receiver in _split_receivers(raw):
                if receiver:
                    receiver_counter[receiver] = receiver_counter.get(receiver, 0) + 1
        area_values = {
            _normalise_dashboard_value(value)
            for value in reports_df.get("area", [])
            if _normalise_dashboard_value(value)
        }

        scoped_reports = reports_for_collection(
            reports_df=reports_df,
            collection_slug=ai_filter,
            query=query,
            selected_filters={
                "coroner": [],
                "area": selected_areas,
                "receiver": selected_receivers,
            },
        )

        if date_start_raw or date_end_raw:
            date_series = _parse_report_dates(scoped_reports.get("date", pd.Series(dtype="object")))
            mask = pd.Series(True, index=scoped_reports.index)
            if date_start_raw:
                start_ts = pd.to_datetime(date_start_raw, errors="coerce")
                if pd.notna(start_ts):
                    mask = mask & (date_series >= start_ts)
            if date_end_raw:
                end_ts = pd.to_datetime(date_end_raw, errors="coerce")
                if pd.notna(end_ts):
                    mask = mask & (date_series <= end_ts)
            scoped_reports = scoped_reports.loc[mask].reset_index(drop=True)

        dataset_date_series = _parse_report_dates(
            reports_df.get("date", pd.Series(dtype="object"))
        ).dropna()
        if dataset_date_series.empty:
            default_start = ""
        else:
            default_start = dataset_date_series.min().date().isoformat()
        default_end = timezone.now().date().isoformat()

        explore = build_explore_metrics(
            reports_df=reports_df,
            scoped_reports_df=scoped_reports,
            query=query,
        )
        selected_ai_filter_title = next(
            (item["title"] for item in ai_filter_options if item["slug"] == ai_filter),
            "All reports",
        )
        scope_area_counts = (
            scoped_reports.get("area", pd.Series(dtype="object"))
            .map(_normalise_dashboard_value)
            .loc[lambda s: s != ""]
            .value_counts()
        )

        scope_receiver_counter: dict[str, int] = {}
        for raw in scoped_reports.get("receiver", pd.Series(dtype="object")).tolist():
            for receiver in _split_receivers(raw):
                if receiver:
                    scope_receiver_counter[receiver] = scope_receiver_counter.get(receiver, 0) + 1

        scope_theme_rows: list[tuple[str, int]] = []
        for _, meta in _theme_collection_map_from_reports(reports_df).items():
            column_name = str(meta.get("collection_column") or "").strip()
            if not column_name or column_name not in scoped_reports.columns:
                continue
            count = int(scoped_reports[column_name].map(_network_truthy).sum())
            if count <= 0:
                continue
            title = str(meta.get("title") or column_name.replace("_", " ").title())
            scope_theme_rows.append((title, count))
        scope_theme_rows.sort(key=lambda item: (-item[1], item[0].casefold()))

        explore.update(
            {
                "selected_ai_filter": ai_filter,
                "selected_ai_filter_title": selected_ai_filter_title,
                "selected_receivers": selected_receivers,
                "selected_areas": selected_areas,
                "date_start": date_start_raw,
                "date_end": date_end_raw,
                "date_start_effective": date_start_raw or default_start,
                "date_end_effective": date_end_raw or default_end,
                "dataset_min_date": default_start,
                "dataset_max_date": default_end,
                "ai_filter_options": ai_filter_options,
                "receiver_options": [
                    receiver
                    for receiver, _ in sorted(
                        receiver_counter.items(),
                        key=lambda item: (-int(item[1]), str(item[0]).casefold()),
                    )
                ],
                "area_options": sorted(area_values, key=str.casefold),
                "all_areas": [
                    {"label": str(label), "count": int(count)}
                    for label, count in scope_area_counts.items()
                ],
                "all_receivers": [
                    {"label": str(label), "count": int(count)}
                    for label, count in sorted(
                        scope_receiver_counter.items(),
                        key=lambda item: (-int(item[1]), str(item[0]).casefold()),
                    )
                ],
                "all_themes": [
                    {"label": str(label), "count": int(count)}
                    for label, count in scope_theme_rows
                ],
            }
        )
        dataset_available = True
    except Exception:
        logger.exception("explore dataset load failed")
        if emit_messages and request is not None:
            messages.error(
                request,
                "Live dataset metrics are temporarily unavailable. You can still search via Collections.",
            )

    return {
        "explore": explore,
        "dataset_available": dataset_available,
        "scoped_reports": scoped_reports,
    }


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
    reset_requested = str(request.GET.get("reset") or "").strip().lower() in {"1", "true", "yes"}
    if reset_requested:
        request.session.pop(EXPLORE_FILTERS_SESSION_KEY, None)
        request.session.pop(EXPLORE_SHOW_REPORTS_SESSION_KEY, None)
        return redirect("explore")

    params = _parse_explore_request_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=str(params["ai_filter"]),
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
        request=request,
        emit_messages=True,
    )
    explore_data = payload["explore"]
    dataset_available = bool(payload["dataset_available"])

    if "show_reports" in request.GET:
        show_reports_requested = str(request.GET.get("show_reports") or "").strip().lower() in {"1", "true", "yes"}
        request.session[EXPLORE_SHOW_REPORTS_SESSION_KEY] = bool(show_reports_requested)
    else:
        show_reports_requested = bool(request.session.get(EXPLORE_SHOW_REPORTS_SESSION_KEY, False))

    request.session[EXPLORE_FILTERS_SESSION_KEY] = {
        "query": str(explore_data.get("query") or ""),
        "ai_filter": str(explore_data.get("selected_ai_filter") or "custom"),
        "selected_receivers": list(explore_data.get("selected_receivers") or []),
        "selected_areas": list(explore_data.get("selected_areas") or []),
        "date_start_raw": str(explore_data.get("date_start") or ""),
        "date_end_raw": str(explore_data.get("date_end") or ""),
    }
    requested_page = _coerce_positive_int(request.GET.get("page"), default=1)
    panel_query = _explore_shared_querydict(
        query=str(params["query"]),
        ai_filter=str(explore_data.get("selected_ai_filter") or "custom"),
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
        page=requested_page,
    )
    panel_base = reverse("explore-reports-panel")
    panel_url = f"{panel_base}?{panel_query.urlencode()}" if panel_query else panel_base

    return render(
        request,
        "accounts/explore.html",
        {
            "explore": explore_data,
            "dataset_available": dataset_available,
            "explore_reports_panel_url": panel_url,
            "show_reports_requested": show_reports_requested and dataset_available,
        },
    )


@require_GET
def explore_reports_panel(request: HttpRequest) -> HttpResponse:
    params = _parse_explore_request_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=str(params["ai_filter"]),
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
    )

    dataset_available = bool(payload["dataset_available"])
    explore_data = payload["explore"]
    scoped_reports = payload["scoped_reports"]

    ordered_columns: list[str] = []
    ordered_column_labels: list[str] = []
    rows: list[dict[str, object]] = []
    reports_count = int(len(scoped_reports))
    page = _coerce_positive_int(request.GET.get("page"), default=1)
    total_pages = 1
    page_from = 0
    page_to = 0
    prev_page: int | None = None
    next_page: int | None = None

    if dataset_available and reports_count > 0:
        ordered_columns = _explore_ordered_columns(scoped_reports)
        ordered_column_labels = [_explore_column_label(column_name) for column_name in ordered_columns]
        display_df = scoped_reports.copy()
        if "date" in display_df.columns:
            display_df["_sort_date"] = _parse_report_dates(
                display_df.get("date", pd.Series(dtype="object"))
            )
            display_df = display_df.sort_values(
                by=["_sort_date"],
                ascending=False,
                na_position="last",
            ).drop(columns=["_sort_date"], errors="ignore")

        total_pages = max(1, int(ceil(reports_count / EXPLORE_REPORTS_PAGE_SIZE)))
        page = max(1, min(page, total_pages))
        start_index = (page - 1) * EXPLORE_REPORTS_PAGE_SIZE
        end_index = min(start_index + EXPLORE_REPORTS_PAGE_SIZE, reports_count)
        page_df = display_df.iloc[start_index:end_index].copy()
        rows = _explore_report_rows(reports_df=page_df, ordered_columns=ordered_columns)

        page_from = start_index + 1 if reports_count else 0
        page_to = end_index
        prev_page = page - 1 if page > 1 else None
        next_page = page + 1 if page < total_pages else None

    shared_query = _explore_shared_querydict(
        query=str(params["query"]),
        ai_filter=str(explore_data.get("selected_ai_filter") or "custom"),
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
    ).urlencode()

    return render(
        request,
        "accounts/_explore_reports_panel.html",
        {
            "dataset_available": dataset_available,
            "reports_columns": ordered_column_labels,
            "reports_rows": rows,
            "reports_count": reports_count,
            "reports_page": page,
            "reports_total_pages": total_pages,
            "reports_page_from": page_from,
            "reports_page_to": page_to,
            "reports_prev_page": prev_page,
            "reports_next_page": next_page,
            "dataset_panel_base": reverse("explore-reports-panel"),
            "dataset_browser_base": reverse("explore"),
            "dataset_shared_query": shared_query,
            "scope_label": "Explore",
        },
    )


@require_GET
def explore_export_csv(request: HttpRequest) -> HttpResponse:
    params = _parse_explore_request_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=str(params["ai_filter"]),
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
    )

    if not bool(payload["dataset_available"]):
        return HttpResponseBadRequest("Explore dataset is unavailable right now.")

    scoped_reports: pd.DataFrame = payload["scoped_reports"]
    ordered_columns = _explore_ordered_columns(scoped_reports)
    export_df = scoped_reports[ordered_columns].copy() if ordered_columns else scoped_reports.copy()

    if "date" in export_df.columns:
        export_df["_sort_date"] = _parse_report_dates(
            export_df.get("date", pd.Series(dtype="object"))
        )
        export_df = export_df.sort_values(
            by=["_sort_date"],
            ascending=False,
            na_position="last",
        ).drop(columns=["_sort_date"], errors="ignore")

    for column_name in ("date", "response_date"):
        if column_name in export_df.columns:
            export_df[column_name] = (
                _parse_report_dates(export_df[column_name]).dt.date.astype("string").fillna("")
            )

    buffer = StringIO()
    export_df.to_csv(buffer, index=False)
    timestamp = timezone.now().strftime("%Y%m%d")
    response = HttpResponse(buffer.getvalue(), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="explore-reports-{timestamp}.csv"'
    return response


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
    return redirect(f"{reverse('landing')}?open_llm_config=1")
