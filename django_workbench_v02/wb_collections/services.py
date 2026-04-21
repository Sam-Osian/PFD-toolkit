from __future__ import annotations

from datetime import datetime
import json
from collections import Counter
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


def _summarise_ranked_counts(
    items: list[tuple[str, int]],
    *,
    limit: int,
    denominator: int,
) -> list[dict[str, Any]]:
    trimmed = [(label, int(count)) for label, count in items if label and int(count) > 0][:limit]
    if not trimmed:
        return []
    max_count = max(count for _, count in trimmed) or 1
    safe_denominator = max(1, int(denominator))
    return [
        {
            "label": label,
            "count": count,
            "percent_of_top": round((count / max_count) * 100, 1),
            "percent_of_scope": round((count / safe_denominator) * 100, 1),
        }
        for label, count in trimmed
    ]


def _normalise_string_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.map(_normalise_dashboard_value).loc[lambda s: s != ""]


def build_explore_metrics(
    *,
    reports_df: pd.DataFrame,
    scoped_reports_df: pd.DataFrame,
    query: str = "",
    top_n: int = 8,
) -> dict[str, Any]:
    scope_df = scoped_reports_df.copy()
    total_reports = int(len(reports_df))
    scope_reports = int(len(scope_df))

    date_series = pd.to_datetime(
        scope_df.get("date", pd.Series(dtype="object")),
        errors="coerce",
        dayfirst=True,
    ).dropna()
    if date_series.empty:
        date_range_label = "Date range unavailable"
    else:
        start_year = int(date_series.min().year)
        end_year = int(date_series.max().year)
        date_range_label = f"{start_year}–{end_year}" if start_year != end_year else str(start_year)

    temporal_points: list[dict[str, Any]] = []
    temporal_line_path = ""
    temporal_area_path = ""
    temporal_latest_month = ""
    temporal_latest_count = 0
    temporal_start_label = ""
    temporal_end_label = ""
    temporal_axis_labels: list[str] = []
    temporal_mode = "month"
    temporal_series: dict[str, dict[str, Any]] = {
        "week": {"points": [], "line_path": "", "area_path": "", "axis_labels": [], "start_label": "", "end_label": "", "latest_label": "", "latest_count": 0},
        "month": {"points": [], "line_path": "", "area_path": "", "axis_labels": [], "start_label": "", "end_label": "", "latest_label": "", "latest_count": 0},
        "year": {"points": [], "line_path": "", "area_path": "", "axis_labels": [], "start_label": "", "end_label": "", "latest_label": "", "latest_count": 0},
    }

    def _build_temporal_series(
        counts_by_period: pd.Series,
        *,
        max_points: int | None,
        smooth_window: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "points": [],
            "line_path": "",
            "area_path": "",
            "axis_labels": [],
            "start_label": "",
            "end_label": "",
            "latest_label": "",
            "latest_count": 0,
        }
        if counts_by_period.empty:
            return result

        counts = [int(value) for value in counts_by_period.tolist()]
        labels = [str(period) for period in counts_by_period.index.tolist()]
        count_series = pd.Series(counts, dtype="float64")
        smoothed_series = count_series.rolling(window=max(1, smooth_window), min_periods=1).mean()
        smoothed_counts = [float(value) for value in smoothed_series.tolist()]

        point_pairs = list(zip(labels, counts, smoothed_counts))
        if max_points is not None and len(point_pairs) > max_points:
            sampled_indexes = sorted(
                {
                    int(round(index * (len(point_pairs) - 1) / (max_points - 1)))
                    for index in range(max_points)
                }
            )
            point_pairs = [point_pairs[index] for index in sampled_indexes]

        smoothed_only = [float(smoothed) for _, _, smoothed in point_pairs]
        clip_low = float(pd.Series(smoothed_only).quantile(0.05))
        clip_high = float(pd.Series(smoothed_only).quantile(0.95))
        if clip_high <= clip_low:
            clip_high = clip_low + 1.0
        span = clip_high - clip_low

        point_count = len(point_pairs)
        raw_points: list[tuple[float, float]] = []
        chart_width = 900.0
        chart_top = 20.0
        chart_bottom = 170.0
        chart_range = chart_bottom - chart_top
        area_bottom = 200.0

        for index, (label, count, smoothed) in enumerate(point_pairs):
            x = chart_width / 2.0 if point_count == 1 else (index * chart_width / (point_count - 1))
            clipped_count = min(max(float(smoothed), clip_low), clip_high)
            y = chart_bottom - ((clipped_count - clip_low) * chart_range / span)
            raw_points.append((x, y))
            result["points"].append(
                {"label": label, "count": count, "x": round(x, 2), "y": round(y, 2)}
            )

        line_segments = [f"M {raw_points[0][0]:.2f} {raw_points[0][1]:.2f}"]
        for x, y in raw_points[1:]:
            line_segments.append(f"L {x:.2f} {y:.2f}")
        result["line_path"] = " ".join(line_segments)
        first_x = raw_points[0][0]
        last_x = raw_points[-1][0]
        result["area_path"] = f'{result["line_path"]} L{last_x:.2f} {area_bottom:.2f} L{first_x:.2f} {area_bottom:.2f} Z'
        result["latest_label"] = point_pairs[-1][0]
        result["latest_count"] = point_pairs[-1][1]
        result["start_label"] = point_pairs[0][0][:4]
        result["end_label"] = point_pairs[-1][0][:4]

        start_year = int(result["start_label"])
        end_year = int(result["end_label"])
        if end_year - start_year >= 2:
            axis_labels = [str(year) for year in range(start_year, end_year, 2)]
            if str(end_year - 1) not in axis_labels:
                axis_labels.append(str(end_year - 1))
            axis_labels.append("Now")
            result["axis_labels"] = axis_labels
        else:
            result["axis_labels"] = [result["start_label"], "Now"]
        return result

    if not date_series.empty:
        temporal_series["week"] = _build_temporal_series(
            date_series.dt.to_period("W-SUN").value_counts().sort_index(),
            max_points=None,
            smooth_window=2,
        )
        temporal_series["month"] = _build_temporal_series(
            date_series.dt.to_period("M").value_counts().sort_index(),
            max_points=None,
            smooth_window=3,
        )
        temporal_series["year"] = _build_temporal_series(
            date_series.dt.to_period("Y").value_counts().sort_index(),
            max_points=20,
            smooth_window=1,
        )

        chosen = temporal_series[temporal_mode]
        temporal_points = chosen["points"]
        temporal_line_path = chosen["line_path"]
        temporal_area_path = chosen["area_path"]
        temporal_latest_month = chosen["latest_label"]
        temporal_latest_count = chosen["latest_count"]
        temporal_start_label = chosen["start_label"]
        temporal_end_label = chosen["end_label"]
        temporal_axis_labels = chosen["axis_labels"]

    area_counts = _normalise_string_series(scope_df.get("area")).value_counts()
    top_areas = _summarise_ranked_counts(
        [(str(label), int(count)) for label, count in area_counts.items()],
        limit=top_n,
        denominator=scope_reports,
    )

    receiver_counter: Counter[str] = Counter()
    for raw in scope_df.get("receiver", pd.Series(dtype="object")).tolist():
        for receiver in _split_receivers(raw):
            receiver_counter[receiver] += 1
    top_receivers = _summarise_ranked_counts(
        receiver_counter.most_common(top_n),
        limit=top_n,
        denominator=scope_reports,
    )

    theme_rows: list[tuple[str, int]] = []
    for _, meta in _theme_collection_map_from_reports(reports_df).items():
        column_name = str(meta.get("collection_column") or "").strip()
        if not column_name or column_name not in scope_df.columns:
            continue
        count = int(scope_df[column_name].map(_network_truthy).sum())
        if count <= 0:
            continue
        title = str(meta.get("title") or column_name.replace("_", " ").title())
        theme_rows.append((title, count))
    theme_rows.sort(key=lambda item: (-item[1], item[0]))
    top_themes = _summarise_ranked_counts(
        theme_rows,
        limit=top_n,
        denominator=scope_reports,
    )

    unique_coroner_count = int(
        _normalise_string_series(scope_df.get("coroner")).str.casefold().nunique()
    )
    unique_receivers: set[str] = set()
    for raw in scope_df.get("receiver", pd.Series(dtype="object")).tolist():
        for receiver in _split_receivers(raw):
            cleaned = _normalise_dashboard_value(receiver)
            if cleaned:
                unique_receivers.add(cleaned.casefold())
    unique_receiver_count = len(unique_receivers)
    most_common_theme = top_themes[0]["label"] if top_themes else "No theme signals"

    return {
        "query": query,
        "has_query": bool(query),
        "total_reports": total_reports,
        "scope_reports": scope_reports,
        "date_range_label": date_range_label,
        "scope_label": (
            f'Custom search for "{query}"' if query else "All reports"
        ),
        "temporal_points": temporal_points,
        "temporal_line_path": temporal_line_path,
        "temporal_area_path": temporal_area_path,
        "temporal_latest_month": temporal_latest_month,
        "temporal_latest_count": temporal_latest_count,
        "temporal_start_label": temporal_start_label,
        "temporal_end_label": temporal_end_label,
        "temporal_axis_labels": temporal_axis_labels,
        "temporal_mode": temporal_mode,
        "temporal_series": temporal_series,
        "unique_coroner_count": unique_coroner_count,
        "unique_receiver_count": unique_receiver_count,
        "most_common_theme": most_common_theme,
        "top_areas": top_areas,
        "top_receivers": top_receivers,
        "top_themes": top_themes,
    }


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
