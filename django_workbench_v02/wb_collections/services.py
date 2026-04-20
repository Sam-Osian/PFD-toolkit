from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any

import pandas as pd
from django.db import transaction
from django.utils import timezone
from pfd_toolkit import load_reports
from pfd_toolkit.collections import COLLECTION_COLUMNS, apply_collection_columns

from wb_auditlog.services import log_audit_event
from wb_investigations.models import InvestigationStatus
from wb_investigations.services import create_investigation
from wb_workspaces.report_identity import REPORT_IDENTITY_COLUMN, with_report_identities
from wb_workspaces.services import create_workspace_for_user

_COLLECTIONS_CACHE_LOCK = Lock()
_COLLECTIONS_CACHE_DF: pd.DataFrame | None = None
_COLLECTIONS_CACHE_UPDATED_AT: datetime | None = None
_COLLECTIONS_CACHE_TTL_SECONDS = 6 * 60 * 60


def _cache_stale(now_utc: datetime) -> bool:
    if _COLLECTIONS_CACHE_DF is None or _COLLECTIONS_CACHE_UPDATED_AT is None:
        return True
    return (now_utc - _COLLECTIONS_CACHE_UPDATED_AT).total_seconds() >= _COLLECTIONS_CACHE_TTL_SECONDS


def _normalise_dashboard_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none"} else text


def _split_receivers(value: Any) -> list[str]:
    text = _normalise_dashboard_value(value)
    if not text:
        return []
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in text.split(";"):
        cleaned = _normalise_dashboard_value(chunk)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(cleaned)
    return parts


def _network_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return False
        except TypeError:
            pass
        return value != 0
    cleaned = str(value).strip().lower()
    return cleaned in {"1", "true", "t", "yes", "y"}


def _lexical_search_score(row: pd.Series, query: str) -> float:
    query_norm = str(query or "").strip().lower()
    if not query_norm:
        return 0.0

    fields: tuple[tuple[str, int], ...] = (
        ("coroner", 3),
        ("area", 3),
        ("concerns", 8),
        ("circumstances", 7),
        ("investigation", 5),
    )
    score = 0.0
    for field_name, weight in fields:
        text = str(row.get(field_name) or "").strip().lower()
        if not text:
            continue
        if query_norm in text:
            score += weight * 6
            continue
        for token in query_norm.split():
            if token and token in text:
                score += weight * 2
    return score


def load_collections_dataset(*, force_refresh: bool = False) -> pd.DataFrame:
    global _COLLECTIONS_CACHE_DF, _COLLECTIONS_CACHE_UPDATED_AT
    now_utc = datetime.utcnow()
    with _COLLECTIONS_CACHE_LOCK:
        if not force_refresh and not _cache_stale(now_utc) and _COLLECTIONS_CACHE_DF is not None:
            return _COLLECTIONS_CACHE_DF.copy()

    reports_df = load_reports(refresh=bool(force_refresh))
    reports_df = reports_df.copy()
    apply_collection_columns(reports_df)
    reports_df = with_report_identities(reports_df)

    with _COLLECTIONS_CACHE_LOCK:
        _COLLECTIONS_CACHE_DF = reports_df.copy()
        _COLLECTIONS_CACHE_UPDATED_AT = datetime.utcnow()
    return reports_df


def collection_cards(reports_df: pd.DataFrame) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = [
        {
            "slug": "custom",
            "title": "All reports",
            "description": "Open the full PFD archive.",
            "count": int(len(reports_df)),
        },
        {
            "slug": "custom-search",
            "title": "Custom collection",
            "description": "Create a collection by searching report text.",
            "count": int(len(reports_df)),
        },
    ]

    for slug, column_name in COLLECTION_COLUMNS.items():
        if column_name not in reports_df.columns:
            continue
        count = int(reports_df[column_name].map(_network_truthy).sum())
        cards.append(
            {
                "slug": slug,
                "title": str(slug).replace("_", " ").title(),
                "description": f"Reports in the {str(slug).replace('_', ' ')} collection.",
                "count": count,
            }
        )
    return cards


def _apply_collection_slug(reports_df: pd.DataFrame, collection_slug: str, query: str) -> pd.DataFrame:
    if collection_slug == "custom":
        return reports_df.copy()
    if collection_slug == "custom-search":
        if not query:
            return reports_df.iloc[0:0].copy()
        scored = reports_df.copy()
        scored["_score"] = scored.apply(lambda row: _lexical_search_score(row, query), axis=1)
        scored = scored.loc[scored["_score"] >= 20].copy()
        if scored.empty:
            return scored.drop(columns=["_score"], errors="ignore")
        scored = scored.sort_values(by=["_score"], ascending=False)
        return scored.drop(columns=["_score"], errors="ignore")

    column_name = COLLECTION_COLUMNS.get(collection_slug, "")
    if not column_name or column_name not in reports_df.columns:
        return reports_df.iloc[0:0].copy()
    return reports_df.loc[reports_df[column_name].map(_network_truthy)].copy()


def _apply_filters(reports_df: pd.DataFrame, selected_filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = reports_df.copy()

    selected_coroners = {str(value).casefold() for value in selected_filters.get("coroner", []) if str(value).strip()}
    selected_areas = {str(value).casefold() for value in selected_filters.get("area", []) if str(value).strip()}
    selected_receivers = {str(value).casefold() for value in selected_filters.get("receiver", []) if str(value).strip()}

    if selected_coroners:
        filtered = filtered.loc[
            filtered.get("coroner", pd.Series(dtype="object")).map(
                lambda value: _normalise_dashboard_value(value).casefold() in selected_coroners
            )
        ]
    if selected_areas:
        filtered = filtered.loc[
            filtered.get("area", pd.Series(dtype="object")).map(
                lambda value: _normalise_dashboard_value(value).casefold() in selected_areas
            )
        ]
    if selected_receivers:
        filtered = filtered.loc[
            filtered.get("receiver", pd.Series(dtype="object")).map(
                lambda value: any(
                    receiver.casefold() in selected_receivers for receiver in _split_receivers(value)
                )
            )
        ]

    return filtered.reset_index(drop=True)


def reports_for_collection(
    *,
    reports_df: pd.DataFrame,
    collection_slug: str,
    query: str,
    selected_filters: dict[str, list[str]],
) -> pd.DataFrame:
    subset = _apply_collection_slug(reports_df, collection_slug, query)
    subset = _apply_filters(subset, selected_filters)
    if REPORT_IDENTITY_COLUMN not in subset.columns:
        subset = with_report_identities(subset)
    return subset


@transaction.atomic
def copy_collection_to_workbook(
    *,
    actor,
    collection_slug: str,
    collection_title: str,
    collection_query: str,
    selected_filters: dict[str, list[str]],
    reports_df: pd.DataFrame,
    request=None,
):
    workspace = create_workspace_for_user(
        user=actor,
        title=f"{collection_title} workbook",
        slug=f"{collection_slug}-{timezone.now().strftime('%Y%m%d%H%M%S')}",
        description=f"Created from collection '{collection_title}'.",
        request=request,
    )
    scope_json = {
        "dataset_source": "collection",
        "collection_slug": collection_slug,
        "collection_query": collection_query,
        "selected_filters": selected_filters,
        "report_identity_allowlist": reports_df[REPORT_IDENTITY_COLUMN].astype(str).tolist(),
    }
    create_investigation(
        actor=actor,
        workspace=workspace,
        title=f"{collection_title} investigation",
        question_text="",
        scope_json=scope_json,
        method_json={},
        status=InvestigationStatus.ACTIVE,
        request=request,
    )

    log_audit_event(
        action_type="collection.copied_to_workbook",
        target_type="workspace",
        target_id=str(workspace.id),
        workspace=workspace,
        user=actor,
        payload={
            "collection_slug": collection_slug,
            "collection_query": collection_query,
            "selected_filters": selected_filters,
            "report_count": len(reports_df),
        },
        request=request,
    )
    return workspace
