from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any
import logging

import pandas as pd
from django.db import transaction
from django.utils import timezone
from django.utils.text import slugify
from pfd_toolkit import load_reports
from pfd_toolkit.collections import COLLECTION_COLUMNS, apply_collection_columns

from wb_auditlog.services import log_audit_event
from wb_workspaces.report_identity import REPORT_IDENTITY_COLUMN, with_report_identities
from wb_workspaces.services import create_workspace_for_user
from .models import CollectionCardSnapshot

_COLLECTIONS_CACHE_LOCK = Lock()
_COLLECTIONS_CACHE_DF: pd.DataFrame | None = None
_COLLECTIONS_CACHE_UPDATED_AT: datetime | None = None
_COLLECTIONS_CACHE_TTL_SECONDS = 6 * 60 * 60
logger = logging.getLogger(__name__)

THEME_COLLECTION_SLUG_PREFIX = "theme"
THEME_COLLECTION_TITLE_OVERRIDES: dict[str, str] = {
    "suicide_risk": "Suicide",
    "care_home_safety": "Care home deaths",
    "acute_hospital_wards": "Acute Hospital Care",
    "alarm_alert_failures": "Alarms and Alerts",
    "capacity_best_interests_failures": "Capacity and Best Interests",
    "consent_decision_making_failures": "Consent and Decision-making",
    "failure_recognise_escalate_deterioration": "Recognising and Escalating Deterioration",
    "family_carer_concerns_not_acted_on": "Family and Carer Concerns",
    "failure_learn_previous_deaths_incidents": "Failure to Learn",
    "follow_up_failures": "Follow-up",
    "housing_homelessness": "Housing and Homelessness",
    "inter_agency_working": "Inter-agency Working",
    "investigation_incident_review_failures": "Investigations and Incident Reviews",
    "it_digital_system_failures": "IT and Digital Systems",
    "language_interpreter_barriers": "Language and Interpreter Barriers",
    "maternity_neonatal_perinatal_care": "Maternity, Neonatal and Perinatal Care",
    "children_young_people": "Children and Young People",
    "missed_appointments_non_attendance": "Missed Appointments and Non-attendance",
    "observation_monitoring_failures": "Observation and Monitoring",
    "policy_procedure_failures": "Policy and Procedure",
    "record_sharing_failures": "Record Sharing",
    "referral_failures": "Referrals",
    "risk_assessment_failures": "Risk Assessment",
    "roads_highways": "Roads and Highways",
    "sepsis_infection": "Sepsis and Infection",
    "staffing_shortages_workload_pressure": "Staffing Shortages and Workload Pressure",
    "suicide_self_harm": "Suicide and Self-harm",
    "test_result_management_failures": "Test Result Management",
    "training_competence_gaps": "Training and Competence",
    "transitions_discharge_failures": "Transitions and Discharge",
    "violence_homicide_related_systems_failures": "Violence and Homicide-related Systems",
    "falls_frailty": "Falls and Frailty",
    "equipment_failures": "Equipment",
    "environmental_design_failures": "Environmental Design",
}
EXCLUDED_THEME_KEYS: set[str] = {
    "policy_procedure_failures",
    "failure_recognise_escalate_deterioration",
    "nutrition",
}
APPROVED_THEME_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "theme_collections" / "approved_themes.json"
)
_APPROVED_THEME_COLUMNS_CACHE: set[str] | None = None
_APPROVED_THEME_COLUMNS_MTIME_NS: int | None = None
COLLECTION_SNAPSHOT_KEY = "default"


def _cache_stale(now_utc: datetime) -> bool:
    if _COLLECTIONS_CACHE_DF is None or _COLLECTIONS_CACHE_UPDATED_AT is None:
        return True
    return (now_utc - _COLLECTIONS_CACHE_UPDATED_AT).total_seconds() >= _COLLECTIONS_CACHE_TTL_SECONDS


def _normalise_copy_title(raw_title: str, fallback_title: str) -> str:
    title = str(raw_title or "").strip()
    if not title:
        return fallback_title
    return title[:255].strip() or fallback_title


def _workspace_slug_for_title(*, title: str, collection_slug: str) -> str:
    base = slugify(title)[:84]
    if not base:
        base = slugify(collection_slug)[:84] or "workbook"
    return base


def _next_workspace_slug_for_actor(*, actor, title: str, collection_slug: str) -> str:
    base = _workspace_slug_for_title(title=title, collection_slug=collection_slug)
    candidate = base
    counter = 2
    from wb_workspaces.models import Workspace

    while Workspace.objects.filter(created_by=actor, slug=candidate).exists():
        suffix = f"-{counter}"
        candidate = f"{base[: max(1, 100 - len(suffix))]}{suffix}"
        counter += 1
    return candidate


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


def _theme_collection_slug(theme_name: str) -> str:
    return f"{THEME_COLLECTION_SLUG_PREFIX}-{slugify(str(theme_name or '')).replace('-', '_')}"


def _approved_theme_columns_with_status() -> tuple[list[str], bool]:
    """
    Return approved theme columns and whether schema parse/load was valid.

    valid=False means schema was unavailable or unreadable; callers may apply
    safe fallback behavior.
    """
    global _APPROVED_THEME_COLUMNS_CACHE, _APPROVED_THEME_COLUMNS_MTIME_NS
    try:
        stat = APPROVED_THEME_SCHEMA_PATH.stat()
    except FileNotFoundError:
        return [], False

    mtime_ns = int(stat.st_mtime_ns)
    if _APPROVED_THEME_COLUMNS_CACHE is not None and _APPROVED_THEME_COLUMNS_MTIME_NS == mtime_ns:
        return sorted(_APPROVED_THEME_COLUMNS_CACHE), True

    try:
        payload = json.loads(APPROVED_THEME_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], False

    raw_themes = payload.get("themes")
    if not isinstance(raw_themes, dict):
        return [], False

    columns: set[str] = set()
    for key in raw_themes.keys():
        theme_key = str(key or "").strip().lower()
        if not theme_key:
            continue
        if not theme_key.startswith("theme_"):
            theme_key = f"theme_{theme_key}"
        columns.add(theme_key)

    _APPROVED_THEME_COLUMNS_CACHE = set(columns)
    _APPROVED_THEME_COLUMNS_MTIME_NS = mtime_ns
    return sorted(columns), True


def _approved_theme_columns() -> set[str]:
    columns, valid = _approved_theme_columns_with_status()
    if not valid:
        return set()
    return set(columns)


def _theme_collection_title(theme_name: str) -> str:
    key = str(theme_name or "").strip().lower()
    if key in THEME_COLLECTION_TITLE_OVERRIDES:
        return THEME_COLLECTION_TITLE_OVERRIDES[key]
    return key.replace("_", " ").title()


def _theme_collection_map_from_reports(reports_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    receiver_collection_columns = set(COLLECTION_COLUMNS.values())
    approved_theme_columns, schema_valid = _approved_theme_columns_with_status()
    discovered_theme_columns = sorted(
        str(column)
        for column in reports_df.columns
        if str(column).startswith("theme_") and str(column) not in receiver_collection_columns
    )
    discovered_set = set(discovered_theme_columns)

    theme_columns: list[str] = []
    if approved_theme_columns:
        # Lock thematic collections to approved schema order.
        theme_columns = [column for column in approved_theme_columns if column in discovered_set]
    elif schema_valid:
        # Valid schema with no approved themes: intentionally show none.
        theme_columns = []
    else:
        # Safe fallback when schema cannot be read/parsed.
        if discovered_theme_columns:
            logger.warning(
                "approved theme schema unavailable or invalid; using discovered theme columns fallback"
            )
        theme_columns = discovered_theme_columns

    collections: dict[str, dict[str, str]] = {}

    for column in theme_columns:
        theme_name = column[len("theme_"):]
        if theme_name in EXCLUDED_THEME_KEYS:
            continue
        slug = _theme_collection_slug(theme_name)
        if not slug:
            continue
        title = _theme_collection_title(theme_name) or "Theme Collection"
        collections[slug] = {
            "slug": slug,
            "title": title,
            "description": f"Reports in the {title} thematic collection.",
            "collection_column": column,
        }
    return collections


def _collection_column_for_slug(collection_slug: str, reports_df: pd.DataFrame) -> str:
    receiver_column = COLLECTION_COLUMNS.get(collection_slug, "")
    if receiver_column:
        return receiver_column
    theme_meta = _theme_collection_map_from_reports(reports_df).get(collection_slug)
    if theme_meta:
        return str(theme_meta.get("collection_column", "")).strip()
    return ""


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

    should_refresh = bool(force_refresh)
    try:
        reports_df = load_reports(refresh=should_refresh)
    except Exception:
        if should_refresh:
            raise
        logger.warning(
            "collections.load_reports failed on cached dataset; retrying with refresh=True",
            exc_info=True,
        )
        reports_df = load_reports(refresh=True)
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
    for slug, meta in _theme_collection_map_from_reports(reports_df).items():
        column_name = str(meta.get("collection_column") or "")
        if not column_name or column_name not in reports_df.columns:
            continue
        count = int(reports_df[column_name].map(_network_truthy).sum())
        cards.append(
            {
                "slug": slug,
                "title": str(meta.get("title") or slug.replace("_", " ").title()),
                "description": str(meta.get("description") or "Thematic collection."),
                "count": count,
            }
        )
    return cards


def _normalise_cards_payload(raw_cards) -> list[dict[str, Any]]:
    if not isinstance(raw_cards, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        title = str(item.get("title") or slug.replace("_", " ").title()).strip()
        description = str(item.get("description") or "").strip()
        try:
            count = int(item.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        result.append(
            {
                "slug": slug,
                "title": title,
                "description": description,
                "count": max(0, count),
            }
        )
    return result


def get_collection_cards_snapshot(*, key: str = COLLECTION_SNAPSHOT_KEY) -> CollectionCardSnapshot | None:
    return CollectionCardSnapshot.objects.filter(key=key).first()


def get_collection_cards_for_list(*, key: str = COLLECTION_SNAPSHOT_KEY) -> tuple[list[dict[str, Any]], datetime | None]:
    snapshot = get_collection_cards_snapshot(key=key)
    if snapshot is None:
        return [], None
    return _normalise_cards_payload(snapshot.cards_json), snapshot.generated_at


@transaction.atomic
def refresh_collection_cards_snapshot(
    *,
    key: str = COLLECTION_SNAPSHOT_KEY,
    force_refresh_dataset: bool = True,
) -> CollectionCardSnapshot:
    reports_df = load_collections_dataset(force_refresh=force_refresh_dataset)
    cards = collection_cards(reports_df)
    snapshot, _ = CollectionCardSnapshot.objects.update_or_create(
        key=key,
        defaults={
            "cards_json": cards,
            "generated_at": timezone.now(),
            "source_row_count": int(len(reports_df)),
        },
    )
    return snapshot


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

    column_name = _collection_column_for_slug(collection_slug, reports_df)
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
    workbook_title: str | None,
    collection_query: str,
    selected_filters: dict[str, list[str]],
    reports_df: pd.DataFrame,
    request=None,
):
    resolved_title = _normalise_copy_title(
        workbook_title or "",
        f"{collection_title} workbook",
    )
    workspace = create_workspace_for_user(
        user=actor,
        title=resolved_title,
        slug=_next_workspace_slug_for_actor(
            actor=actor,
            title=resolved_title,
            collection_slug=collection_slug,
        ),
        description=f"Created from collection '{collection_title}'.",
        request=request,
    )

    from wb_investigations.models import InvestigationStatus
    from wb_investigations.services import create_investigation

    report_identity_allowlist: list[str] = []
    if REPORT_IDENTITY_COLUMN in reports_df.columns:
        seen_report_identities: set[str] = set()
        for value in reports_df[REPORT_IDENTITY_COLUMN].tolist():
            report_identity = str(value or "").strip()
            if not report_identity:
                continue
            if report_identity in seen_report_identities:
                continue
            seen_report_identities.add(report_identity)
            report_identity_allowlist.append(report_identity)

    scope_json = {
        "collection_slug": str(collection_slug or "").strip(),
        "collection_query": str(collection_query or "").strip(),
        "selected_filters": {
            "coroner": [str(value).strip() for value in selected_filters.get("coroner", []) if str(value).strip()],
            "area": [str(value).strip() for value in selected_filters.get("area", []) if str(value).strip()],
            "receiver": [str(value).strip() for value in selected_filters.get("receiver", []) if str(value).strip()],
        },
        "report_identity_allowlist": report_identity_allowlist,
    }
    method_json = {
        "run_filter": True,
        "run_themes": False,
        "run_extract": False,
        "pipeline_plan": [],
    }
    create_investigation(
        actor=actor,
        workspace=workspace,
        title=f"{collection_title} investigation",
        question_text=str(collection_query or "").strip(),
        scope_json=scope_json,
        method_json=method_json,
        status=InvestigationStatus.DRAFT,
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
