from __future__ import annotations

from io import StringIO
import logging
import hashlib
from math import ceil

import pandas as pd
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from accounts.views import (
    EXPLORE_REPORTS_PAGE_SIZE,
    _build_explore_payload,
    _coerce_positive_int,
    _explore_column_label,
    _explore_ordered_columns,
    _explore_report_rows,
    _explore_shared_querydict,
)
from .services import (
    collection_cards,
    copy_collection_to_workbook,
    get_collection_cards_for_list,
    load_collections_dataset,
    refresh_collection_cards_snapshot,
    reports_for_collection,
)

logger = logging.getLogger(__name__)

_COLLECTION_BADGE_BY_SLUG: dict[str, str] = {
    "custom": "Complete archive",
    "custom-search": "Custom",
    "wales": "Geography",
    "local_gov": "Geography",
    "nhs": "Health",
    "health_regulators": "Health",
    "gov_department": "Government",
    "prisons": "Justice",
}
_COLLECTION_SPARK_COLOR_BY_BADGE: dict[str, str] = {
    "Complete archive": "var(--accent)",
    "Custom": "var(--accent-2)",
    "Geography": "oklch(0.70 0.12 150)",
    "Health": "oklch(0.65 0.15 200)",
    "Government": "oklch(0.70 0.16 55)",
    "Justice": "oklch(0.68 0.14 30)",
    "Theme": "oklch(0.65 0.18 300)",
}
_COLLECTION_CARD_ORDER: tuple[str, ...] = (
    "wales",
    "nhs",
    "gov_department",
    "prisons",
    "health_regulators",
    "local_gov",
)
_COLLECTION_CARD_COPY: dict[str, dict[str, object]] = {
    "custom": {
        "title": "All reports",
        "description": "The full PFD archive. Start here for a broad pattern search across every report, coroner, and receiver since 2013.",
        "chips": [{"key": "", "value": "2013 → 2026"}, {"key": "", "value": "All coroners"}],
        "footer": "up",
    },
    "custom-search": {
        "title": "Custom collection · Active AI filter",
        "description": "Your active AI filter. Reports matching your current Explore scope and query.",
        "chips": [{"key": "filter", "value": "AI screened"}, {"key": "status", "value": "live"}],
        "footer": "saved",
    },
    "wales": {
        "title": "Wales",
        "description": "Reports where the coroner area is a Welsh jurisdiction, including legacy area names recoded to current Welsh areas.",
        "chips": [{"key": "areas", "value": "Welsh scope"}, {"key": "status", "value": "live"}],
        "footer": "updated",
    },
    "nhs": {
        "title": "NHS Bodies",
        "description": "Reports addressed to NHS organisations including trusts, foundation trusts, integrated care boards, and health boards.",
        "chips": [{"key": "scope", "value": "Receiver-linked"}, {"key": "status", "value": "live"}],
        "footer": "up",
    },
    "gov_department": {
        "title": "Government departments",
        "description": "Reports addressed to UK government departments and central offices, including Home Office, Cabinet Office, and ministerial departments.",
        "chips": [{"key": "scope", "value": "Receiver-linked"}, {"key": "status", "value": "live"}],
        "footer": "up",
    },
    "prisons": {
        "title": "Prisons",
        "description": "Reports sent directly to prisons, prison institutions, young offender settings, or HM Prison and Probation Service.",
        "chips": [{"key": "scope", "value": "Receiver-linked"}, {"key": "status", "value": "live"}],
        "footer": "updated",
    },
    "health_regulators": {
        "title": "Health regulators",
        "description": "Reports addressed to national health regulators and oversight bodies with statutory safety responsibilities.",
        "chips": [{"key": "scope", "value": "Receiver-linked"}, {"key": "status", "value": "live"}],
        "footer": "updated",
    },
    "local_gov": {
        "title": "Local government",
        "description": "Reports addressed to local councils and combined authorities, including social care and public health-adjacent duties.",
        "chips": [{"key": "scope", "value": "Area-linked"}, {"key": "status", "value": "live"}],
        "footer": "updated",
    },
}


def _collection_badge(slug: str) -> str:
    clean_slug = str(slug or "").strip().lower()
    if clean_slug.startswith("theme-"):
        return "Theme"
    return _COLLECTION_BADGE_BY_SLUG.get(clean_slug, "Collection")


def _spark_path_for_slug(*, slug: str, count: int) -> str:
    digest = hashlib.sha256(str(slug or "").encode("utf-8")).digest()
    points: list[tuple[int, int]] = []
    steps = 10
    denominator = max(1, steps)
    trend = max(-3, min(5, count // 1200))
    y = 18 - trend
    for index in range(steps + 1):
        x = int((index / denominator) * 200)
        # Keep the sparkline smooth and close to the Claude reference cadence.
        delta = (digest[index % len(digest)] % 5) - 2
        y = max(6, min(22, y + delta))
        points.append((x, y))
    first_x, first_y = points[0]
    segments = [f"M{first_x},{first_y}"]
    for x, y in points[1:]:
        segments.append(f"L{x},{y}")
    return " ".join(segments)


def _unique_receiver_count(reports_df) -> int:
    if "receiver" not in reports_df.columns:
        return 0
    seen: set[str] = set()
    for raw in reports_df["receiver"].tolist():
        text = str(raw or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        for chunk in text.split(";"):
            cleaned = str(chunk or "").strip()
            if not cleaned:
                continue
            seen.add(cleaned.casefold())
    return len(seen)


def _card_metrics_map(*, reports_df, cards: list[dict]) -> dict[str, dict[str, int]]:
    if reports_df is None:
        return {}
    metrics: dict[str, dict[str, int]] = {}
    for item in cards:
        slug = str(item.get("slug") or "").strip()
        if not slug or slug in {"custom", "custom-search"}:
            continue
        try:
            subset = reports_for_collection(
                reports_df=reports_df,
                collection_slug=slug,
                query="",
                selected_filters={"coroner": [], "area": [], "receiver": []},
            )
        except Exception:
            logger.exception("collection metrics build failed for slug=%s", slug)
            continue
        metrics[slug] = {
            "reports": int(len(subset)),
            "receivers": int(_unique_receiver_count(subset)),
        }
    return metrics


def _ui_cards(cards: list[dict], *, metrics_by_slug: dict[str, dict[str, int]] | None = None) -> list[dict]:
    by_slug: dict[str, dict] = {}
    for item in cards:
        slug = str(item.get("slug") or "").strip()
        if slug:
            by_slug[slug] = item

    ordered_cards: list[dict] = []
    for slug in _COLLECTION_CARD_ORDER:
        card = by_slug.pop(slug, None)
        if card is not None:
            ordered_cards.append(card)
    # Keep any remaining cards in deterministic order.
    for slug in sorted(by_slug.keys()):
        ordered_cards.append(by_slug[slug])

    ui_cards: list[dict] = []
    for item in ordered_cards:
        slug = str(item.get("slug") or "").strip()
        if slug in {"custom", "custom-search"}:
            # Hide only these two cards per product decision.
            continue
        title = str(item.get("title") or "").strip() or slug.replace("_", " ").title()
        description = str(item.get("description") or "").strip()
        try:
            count = max(0, int(item.get("count") or 0))
        except (TypeError, ValueError):
            count = 0
        badge = _collection_badge(slug)
        spark_color = _COLLECTION_SPARK_COLOR_BY_BADGE.get(badge, "var(--accent)")
        override = _COLLECTION_CARD_COPY.get(slug, {})
        if override.get("title"):
            title = str(override["title"])
        if override.get("description"):
            description = str(override["description"])
        slug_metrics = (metrics_by_slug or {}).get(slug, {})
        reports_count = int(slug_metrics.get("reports") or count)
        receiver_count = int(slug_metrics.get("receivers") or 0)
        chips = [
            {"key": "reports", "value": f"{reports_count:,}"},
            {"key": "receivers", "value": f"{receiver_count:,}"},
        ]
        footer_left = f"↑ {max(1, reports_count // 50)} this week"
        footer_up = True

        ui_cards.append(
            {
                "slug": slug,
                "title": title,
                "description": description,
                "count": reports_count,
                "count_display": f"{reports_count:,}",
                "badge": badge,
                "is_primary": False,
                "chips": chips,
                "spark_path": _spark_path_for_slug(slug=slug, count=reports_count),
                "spark_color": spark_color,
                "footer_left": footer_left,
                "footer_up": footer_up,
            }
        )
    return ui_cards


def _selected_filters_from_request(request) -> dict[str, list[str]]:
    def _values(key: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for raw in request.GET.getlist(key):
            cleaned = str(raw or "").strip()
            if not cleaned:
                continue
            lowered = cleaned.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            values.append(cleaned)
        return values

    return {
        "coroner": _values("coroner"),
        "area": _values("area"),
        "receiver": _values("receiver"),
    }


def _collection_meta(cards: list[dict], slug: str) -> dict:
    for card in cards:
        if card.get("slug") == slug:
            return card
    return {"slug": slug, "title": slug.replace("_", " ").title(), "description": "Collection"}


def _parse_collection_explore_params(request) -> dict[str, object]:
    selected_filters = _selected_filters_from_request(request)
    return {
        "query": str(request.GET.get("q") or "").strip(),
        "selected_receivers": selected_filters["receiver"],
        "selected_areas": selected_filters["area"],
        "date_start_raw": str(request.GET.get("date_start") or "").strip(),
        "date_end_raw": str(request.GET.get("date_end") or "").strip(),
    }


@require_GET
def collection_list(request):
    cards: list[dict] = []
    generated_at = None
    collection_data_source = "none"
    reports_df = None

    try:
        cards, generated_at = get_collection_cards_for_list()
    except Exception:
        logger.exception("collection snapshot read failed")
        cards, generated_at = [], None

    if cards:
        collection_data_source = "snapshot"

    if not cards:
        try:
            refresh_collection_cards_snapshot(force_refresh_dataset=False)
            cards, generated_at = get_collection_cards_for_list()
            if cards:
                collection_data_source = "snapshot"
        except Exception:
            logger.exception("collection snapshot auto-refresh failed")

    # Always fall back to live dataset-backed cards so the UI is populated
    # even if snapshot persistence/refresh is unavailable.
    if not cards:
        try:
            reports_df = load_collections_dataset(force_refresh=False)
            cards = collection_cards(reports_df)
            collection_data_source = "live"
        except Exception:
            logger.exception("collection live-card fallback failed")
            cards = []
            collection_data_source = "none"
    else:
        try:
            reports_df = load_collections_dataset(force_refresh=False)
        except Exception:
            logger.exception("collection dataset load failed for card metrics")
            reports_df = None

    metrics_by_slug = _card_metrics_map(reports_df=reports_df, cards=cards)

    return render(
        request,
        "wb_collections/collection_list.html",
        {
            "collections": _ui_cards(cards, metrics_by_slug=metrics_by_slug),
            "snapshot_generated_at": generated_at,
            "snapshot_available": bool(generated_at),
            "collection_data_source": collection_data_source,
        },
    )


@require_GET
def collection_detail(request, collection_slug):
    cards: list[dict] = []
    try:
        cards, _ = get_collection_cards_for_list()
    except Exception:
        logger.exception("collection snapshot read failed for detail view")
    collection = _collection_meta(cards, collection_slug)

    params = _parse_collection_explore_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=collection_slug,
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
        request=request,
        emit_messages=True,
        force_collection_scope=True,
    )
    explore_data = payload["explore"]
    dataset_available = bool(payload["dataset_available"])

    explore_data["selected_ai_filter"] = collection_slug
    explore_data["selected_ai_filter_title"] = str(collection.get("title") or collection_slug)

    if "show_reports" in request.GET:
        show_reports_requested = str(request.GET.get("show_reports") or "").strip().lower() in {"1", "true", "yes"}
    else:
        show_reports_requested = False

    requested_page = _coerce_positive_int(request.GET.get("page"), default=1)
    panel_query = _explore_shared_querydict(
        query=str(params["query"]),
        ai_filter="",
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
        page=requested_page,
    )
    panel_base = reverse("collection-reports-panel", kwargs={"collection_slug": collection_slug})
    panel_url = f"{panel_base}?{panel_query.urlencode()}" if panel_query else panel_base

    return render(
        request,
        "accounts/explore.html",
        {
            "explore": explore_data,
            "dataset_available": dataset_available,
            "explore_reports_panel_url": panel_url,
            "show_reports_requested": show_reports_requested and dataset_available,
            "is_collection_view": True,
            "collection": collection,
            "collection_slug": collection_slug,
            "collection_title": str(collection.get("title") or collection_slug),
            "collection_description": str(collection.get("description") or "").strip(),
            "filter_action_url": reverse("collection-detail", kwargs={"collection_slug": collection_slug}),
            "filter_reset_url": reverse("collection-detail", kwargs={"collection_slug": collection_slug}),
            "explore_export_url": reverse("collection-export-csv", kwargs={"collection_slug": collection_slug}),
        },
    )


@require_GET
def collection_reports_panel(request, collection_slug):
    params = _parse_collection_explore_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=collection_slug,
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
        force_collection_scope=True,
    )

    dataset_available = bool(payload["dataset_available"])
    explore_data = payload["explore"]
    scoped_reports: pd.DataFrame = payload["scoped_reports"]

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
            display_df["_sort_date"] = pd.to_datetime(
                display_df.get("date", pd.Series(dtype="object")),
                errors="coerce",
                dayfirst=True,
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
        ai_filter="",
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
            "dataset_panel_base": reverse("collection-reports-panel", kwargs={"collection_slug": collection_slug}),
            "dataset_browser_base": reverse("collection-detail", kwargs={"collection_slug": collection_slug}),
            "dataset_shared_query": shared_query,
            "scope_label": "collection",
        },
    )


@require_GET
def collection_export_csv(request, collection_slug):
    params = _parse_collection_explore_params(request)
    payload = _build_explore_payload(
        query=str(params["query"]),
        ai_filter=collection_slug,
        selected_receivers=list(params["selected_receivers"]),
        selected_areas=list(params["selected_areas"]),
        date_start_raw=str(params["date_start_raw"]),
        date_end_raw=str(params["date_end_raw"]),
        force_collection_scope=True,
    )
    if not bool(payload["dataset_available"]):
        return HttpResponseBadRequest("Collection dataset is unavailable right now.")

    scoped_reports: pd.DataFrame = payload["scoped_reports"]
    ordered_columns = _explore_ordered_columns(scoped_reports)
    export_df = scoped_reports[ordered_columns].copy() if ordered_columns else scoped_reports.copy()

    if "date" in export_df.columns:
        export_df["_sort_date"] = pd.to_datetime(
            export_df.get("date", pd.Series(dtype="object")),
            errors="coerce",
            dayfirst=True,
        )
        export_df = export_df.sort_values(
            by=["_sort_date"],
            ascending=False,
            na_position="last",
        ).drop(columns=["_sort_date"], errors="ignore")

    for column_name in ("date", "response_date"):
        if column_name in export_df.columns:
            export_df[column_name] = pd.to_datetime(
                export_df[column_name],
                errors="coerce",
                dayfirst=True,
            ).dt.date.astype("string").fillna("")

    buffer = StringIO()
    export_df.to_csv(buffer, index=False)
    timestamp = timezone.now().strftime("%Y%m%d")
    response = HttpResponse(buffer.getvalue(), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="collection-{collection_slug}-reports-{timestamp}.csv"'
    return response


@login_required
@require_POST
def collection_copy(request, collection_slug):
    reports_df = load_collections_dataset(force_refresh=False)
    cards = collection_cards(reports_df)
    collection = _collection_meta(cards, collection_slug)

    query = str(request.POST.get("q") or "").strip()
    selected_filters = {
        "coroner": [value for value in request.POST.getlist("coroner") if str(value).strip()],
        "area": [value for value in request.POST.getlist("area") if str(value).strip()],
        "receiver": [value for value in request.POST.getlist("receiver") if str(value).strip()],
    }

    filtered_reports = reports_for_collection(
        reports_df=reports_df,
        collection_slug=collection_slug,
        query=query,
        selected_filters=selected_filters,
    )

    workspace = copy_collection_to_workbook(
        actor=request.user,
        collection_slug=collection_slug,
        collection_title=str(collection.get("title") or collection_slug),
        workbook_title=str(request.POST.get("workbook_title") or "").strip(),
        collection_query=query,
        selected_filters=selected_filters,
        reports_df=filtered_reports,
        request=request,
    )
    messages.success(request, "Collection copied into a new workbook.")
    return redirect("workbook-detail", workbook_id=workspace.id)


@login_required
@require_POST
def collection_copy_requires_auth(request, collection_slug):
    # Explicit endpoint to keep login-required redirect behavior obvious in templates.
    return collection_copy(request, collection_slug)
