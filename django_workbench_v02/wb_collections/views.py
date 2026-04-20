from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from wb_workspaces.models import Workspace
from wb_workspaces.permissions import can_edit_workspace

from .services import (
    collection_cards,
    copy_collection_to_workbook,
    load_collections_dataset,
    reports_for_collection,
)


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


@require_GET
def collection_list(request):
    reports_df = load_collections_dataset(force_refresh=False)
    cards = collection_cards(reports_df)
    return render(
        request,
        "wb_collections/collection_list.html",
        {
            "collections": cards,
        },
    )


@require_GET
def collection_detail(request, collection_slug):
    reports_df = load_collections_dataset(force_refresh=False)
    cards = collection_cards(reports_df)

    query = str(request.GET.get("q") or "").strip()
    selected_filters = _selected_filters_from_request(request)
    filtered_reports = reports_for_collection(
        reports_df=reports_df,
        collection_slug=collection_slug,
        query=query,
        selected_filters=selected_filters,
    )

    workbook = None
    workbook_id = str(request.GET.get("workbook") or "").strip()
    workbook_edit_allowed = False
    if workbook_id:
        workspace = Workspace.objects.filter(id=workbook_id).first()
        if workspace is not None and can_edit_workspace(request.user, workspace):
            workbook = workspace
            workbook_edit_allowed = True

    preferred_columns = (
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
        "url",
    )
    internal_prefixes = ("theme_", "collection_")
    internal_exact = {"__report_identity", "_score"}
    ordered_preview_columns: list[str] = []
    for column in preferred_columns:
        if column in filtered_reports.columns and column not in ordered_preview_columns:
            ordered_preview_columns.append(column)
    for column in filtered_reports.columns:
        name = str(column)
        if name in internal_exact:
            continue
        if any(name.startswith(prefix) for prefix in internal_prefixes):
            continue
        if name not in ordered_preview_columns:
            ordered_preview_columns.append(name)

    preview_columns = ["__report_identity", *ordered_preview_columns]
    preview_df = filtered_reports[preview_columns].head(200).copy()
    preview_df = preview_df.rename(columns={"__report_identity": "report_identity"})
    preview_records = preview_df.to_dict(orient="records")
    preview_rows = []
    for row in preview_records:
        row_url = row.get("url")
        cells = []
        for column in ordered_preview_columns:
            value = row.get(column)
            link_href = ""
            if column == "url" and row_url:
                link_href = str(row_url)
            elif column == "title" and row_url:
                link_href = str(row_url)
            cells.append(
                {
                    "value": value,
                    "link_href": link_href,
                }
            )
        preview_rows.append(
            {
                "report_identity": row.get("report_identity"),
                "title": row.get("title"),
                "date": row.get("date"),
                "url": row_url,
                "cells": cells,
            }
        )

    return render(
        request,
        "wb_collections/collection_detail.html",
        {
            "collection": _collection_meta(cards, collection_slug),
            "collection_slug": collection_slug,
            "query": query,
            "selected_filters": selected_filters,
            "reports_count": len(filtered_reports),
            "preview_columns": ordered_preview_columns,
            "preview_rows": preview_rows,
            "workbook": workbook,
            "workbook_edit_allowed": workbook_edit_allowed,
        },
    )


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
