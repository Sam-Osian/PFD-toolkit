from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import zip_longest

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.db.models import Count, Max, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from wb_auditlog.models import AuditEvent
from wb_auditlog.services import log_audit_event
from wb_investigations.models import Investigation
from wb_runs.artifact_storage import ArtifactStorageError, open_artifact_for_download
from wb_runs.models import (
    ArtifactStatus,
    ArtifactType,
    InvestigationRun,
    RunArtifact,
    RunStatus,
    RunType,
    RunWorkerHeartbeat,
)
from wb_runs.services import (
    approve_run_for_execution,
    reject_run_for_execution,
)
from wb_sharing.models import WorkspaceShareLink
from wb_workspaces.models import Workspace, WorkspaceMembership, WorkspaceReportExclusion
from wb_workspaces.permissions import can_edit_workspace
from wb_workspaces.report_identity import REPORT_IDENTITY_COLUMN, with_report_identities
from wb_workspaces.services import (
    WorkspaceReportExclusionError,
    restore_workspace_report_exclusion,
    upsert_workspace_report_exclusion,
)


User = get_user_model()

ACTIVE_RUN_STATUSES = {
    RunStatus.STARTING,
    RunStatus.RUNNING,
    RunStatus.CANCELLING,
}
VIEW_EVENT_ACTIONS = {
    "investigation.viewed",
    "run.viewed",
    "run.artifact_downloaded",
    "sharing.link_viewed",
}
LOOKBACK_DAYS_DEFAULT = 7
LOOKBACK_DAYS_MIN = 1
LOOKBACK_DAYS_MAX = 90
RECENT_RUN_LIMIT = 40
WORKSPACE_RUN_LIMIT = 30
USER_RUN_LIMIT = 60
DATASET_PREVIEW_LIMIT = 120
BUSY_WORKER_TARGET = 3
PENDING_APPROVAL_LIMIT = 120
FAILED_RUN_LIMIT = 180
ALLOWED_MODELS = {
    "gemma4:26b",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4o",
}
SCOPE_CHOICES = {
    "all_reports",
    "last_3_years",
    "last_year",
    "last_6_months",
    "most_recent_100",
    "custom_range",
}
RUN_TYPE_ORDER = [RunType.FILTER, RunType.THEMES, RunType.EXTRACT]


@dataclass
class TypedRunReviewPayload:
    title: str
    question_text: str
    scope_option: str
    custom_start_date: str
    custom_end_date: str
    run_filter: bool
    run_themes: bool
    run_extract: bool
    search_query: str
    filter_df: bool
    include_supporting_quotes: bool
    seed_topics: str
    min_themes: int | None
    max_themes: int | None
    extra_theme_instructions: str
    provider: str
    model_name: str
    max_parallel_workers: int
    request_completion_email: bool
    feature_fields: list[dict]
    allow_multiple: bool
    force_assign: bool
    skip_if_present: bool
    extract_include_supporting_quotes: bool


def _coerce_lookback_days(raw_value) -> int:
    try:
        parsed = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        parsed = LOOKBACK_DAYS_DEFAULT
    return min(LOOKBACK_DAYS_MAX, max(LOOKBACK_DAYS_MIN, parsed))


def _safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "nat", "none", "null"}:
        return ""
    return text


def _to_bool(value) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_int(value, *, default: int | None = None, minimum: int | None = None, maximum: int | None = None):
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _parse_decimal(value) -> Decimal | None:
    compact = str(value or "").strip()
    if not compact:
        return None
    try:
        return Decimal(compact)
    except (InvalidOperation, ValueError):
        return None


def _format_relative_delta(ts, *, now):
    if not ts:
        return "-"
    delta = max(0, int((now - ts).total_seconds()))
    if delta < 60:
        return f"{delta}s"
    minutes, seconds = divmod(delta, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _duration_between(started_at, finished_at, *, now):
    if not started_at:
        return "-"
    effective_end = finished_at or now
    return _format_relative_delta(started_at, now=effective_end)


def _derive_scope_option(*, query_start_date, query_end_date, report_limit):
    if report_limit == 100:
        return "most_recent_100"
    if query_start_date and query_end_date:
        return "custom_range"
    return "all_reports"


def _typed_payload_from_run(run: InvestigationRun) -> TypedRunReviewPayload:
    investigation = run.investigation
    config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
    scope = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    plan = config.get("pipeline_plan") if isinstance(config.get("pipeline_plan"), list) else [run.run_type]
    plan = [str(item).strip().lower() for item in plan if str(item).strip().lower() in RUN_TYPE_ORDER]
    if not plan:
        plan = [RunType.FILTER]

    query_start_date = run.query_start_date or scope.get("query_start_date") or ""
    query_end_date = run.query_end_date or scope.get("query_end_date") or ""
    report_limit = config.get("report_limit", scope.get("report_limit"))
    scope_option = str(scope.get("temporal_scope_option") or "").strip().lower() or _derive_scope_option(
        query_start_date=query_start_date,
        query_end_date=query_end_date,
        report_limit=report_limit,
    )
    if scope_option not in SCOPE_CHOICES:
        scope_option = "all_reports"

    provider = str(config.get("provider") or "local_ollama").strip().lower()
    if provider not in {"local_ollama", "openai"}:
        provider = "openai"

    model_name = str(config.get("model_name") or "gemma4:26b").strip()
    if not model_name:
        model_name = "gemma4:26b"
    feature_rows = config.get("feature_fields") if isinstance(config.get("feature_fields"), list) else []
    feature_fields: list[dict] = []
    for row in feature_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("field_name") or "").strip()
        description = str(row.get("description") or "").strip()
        field_type = str(row.get("type") or "text").strip().lower() or "text"
        if field_type == "number":
            field_type = "decimal"
        if field_type not in {"text", "decimal", "boolean"}:
            field_type = "text"
        if not name:
            continue
        feature_fields.append(
            {
                "name": name,
                "description": description,
                "type": field_type,
            }
        )
    if not feature_fields:
        feature_fields = [{"name": "", "description": "", "type": "text"}]

    return TypedRunReviewPayload(
        title=str(investigation.title or "").strip(),
        question_text=str(investigation.question_text or "").strip(),
        scope_option=scope_option,
        custom_start_date=str(query_start_date or ""),
        custom_end_date=str(query_end_date or ""),
        run_filter=RunType.FILTER in plan,
        run_themes=RunType.THEMES in plan,
        run_extract=RunType.EXTRACT in plan,
        search_query=str(config.get("search_query") or investigation.question_text or "").strip(),
        filter_df=bool(config.get("filter_df", True)),
        include_supporting_quotes=bool(config.get("produce_spans", False)),
        seed_topics=str(config.get("seed_topics") or "").strip(),
        min_themes=_parse_int(config.get("min_themes"), default=None, minimum=1, maximum=100),
        max_themes=_parse_int(config.get("max_themes"), default=None, minimum=1, maximum=100),
        extra_theme_instructions=str(config.get("extra_theme_instructions") or "").strip(),
        provider=provider,
        model_name=model_name,
        max_parallel_workers=_parse_int(config.get("max_parallel_workers"), default=1, minimum=1, maximum=32) or 1,
        request_completion_email=True,
        feature_fields=feature_fields,
        allow_multiple=bool(config.get("allow_multiple", False)),
        force_assign=bool(config.get("force_assign", False)),
        skip_if_present=bool(config.get("skip_if_present", True)),
        extract_include_supporting_quotes=bool(config.get("produce_spans", False)),
    )


def _feature_fields_from_post(request) -> list[dict]:
    names = request.POST.getlist("feature_field_name")
    descriptions = request.POST.getlist("feature_field_description")
    types = request.POST.getlist("feature_field_type")
    rows: list[dict] = []
    for raw_name, raw_description, raw_type in zip_longest(names, descriptions, types, fillvalue=""):
        name = str(raw_name or "").strip()
        description = str(raw_description or "").strip()
        field_type = str(raw_type or "text").strip().lower() or "text"
        if field_type == "number":
            field_type = "decimal"
        if field_type not in {"text", "decimal", "boolean"}:
            field_type = "text"
        if not name and not description:
            continue
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "description": description,
                "type": field_type,
            }
        )
    return rows


def _typed_payload_from_post(request) -> TypedRunReviewPayload:
    scope_option = str(request.POST.get("scope_option") or "all_reports").strip().lower()
    if scope_option not in SCOPE_CHOICES:
        scope_option = "all_reports"

    provider = str(request.POST.get("provider") or "local_ollama").strip().lower()
    if provider not in {"local_ollama", "openai"}:
        provider = "local_ollama"
    model_name = str(request.POST.get("model_name") or "gemma4:26b").strip()
    if model_name not in ALLOWED_MODELS:
        model_name = "gemma4:26b" if provider == "local_ollama" else "gpt-5-mini"

    min_themes = _parse_int(request.POST.get("min_themes"), default=None, minimum=1, maximum=100)
    max_themes = _parse_int(request.POST.get("max_themes"), default=None, minimum=1, maximum=100)
    if min_themes and max_themes and min_themes > max_themes:
        min_themes, max_themes = max_themes, min_themes
    include_supporting_quotes = _to_bool(request.POST.get("include_supporting_quotes"))
    extract_include_supporting_quotes = _to_bool(request.POST.get("extract_include_supporting_quotes"))

    return TypedRunReviewPayload(
        title=str(request.POST.get("title") or "").strip(),
        question_text=str(request.POST.get("question_text") or "").strip(),
        scope_option=scope_option,
        custom_start_date=str(request.POST.get("custom_start_date") or "").strip(),
        custom_end_date=str(request.POST.get("custom_end_date") or "").strip(),
        run_filter=_to_bool(request.POST.get("run_filter")),
        run_themes=_to_bool(request.POST.get("run_themes")),
        run_extract=_to_bool(request.POST.get("run_extract")),
        search_query=str(request.POST.get("search_query") or "").strip(),
        filter_df=_to_bool(request.POST.get("filter_df")),
        include_supporting_quotes=include_supporting_quotes,
        seed_topics=str(request.POST.get("seed_topics") or "").strip(),
        min_themes=min_themes,
        max_themes=max_themes,
        extra_theme_instructions=str(request.POST.get("extra_theme_instructions") or "").strip(),
        provider=provider,
        model_name=model_name,
        max_parallel_workers=_parse_int(request.POST.get("max_parallel_workers"), default=1, minimum=1, maximum=32) or 1,
        request_completion_email=_to_bool(request.POST.get("request_completion_email")),
        feature_fields=_feature_fields_from_post(request),
        allow_multiple=_to_bool(request.POST.get("allow_multiple")),
        force_assign=_to_bool(request.POST.get("force_assign")),
        skip_if_present=_to_bool(request.POST.get("skip_if_present")),
        extract_include_supporting_quotes=extract_include_supporting_quotes,
    )


def _build_pipeline_plan(payload: TypedRunReviewPayload, *, fallback_run_type: str) -> list[str]:
    plan = []
    if payload.run_filter:
        plan.append(RunType.FILTER)
    if payload.run_themes:
        plan.append(RunType.THEMES)
    if payload.run_extract:
        plan.append(RunType.EXTRACT)
    if not plan:
        plan = [str(fallback_run_type or RunType.FILTER).strip().lower()]
    return plan


def _resolve_scope_dates(payload: TypedRunReviewPayload):
    now = timezone.localdate()
    if payload.scope_option == "custom_range":
        return payload.custom_start_date or "", payload.custom_end_date or "", None
    if payload.scope_option == "most_recent_100":
        return "", "", 100
    if payload.scope_option == "last_6_months":
        return (now - timedelta(days=183)).isoformat(), now.isoformat(), None
    if payload.scope_option == "last_year":
        return (now - timedelta(days=365)).isoformat(), now.isoformat(), None
    if payload.scope_option == "last_3_years":
        return (now - timedelta(days=365 * 3)).isoformat(), now.isoformat(), None
    return "", "", None


def _parse_iso_date(value: str):
    compact = str(value or "").strip()
    if not compact:
        return None
    try:
        return timezone.datetime.fromisoformat(compact).date()
    except ValueError:
        return None


def _scheduled_for_from_post(request):
    raw = str(request.POST.get("scheduled_for") or "").strip()
    if not raw:
        return None
    try:
        parsed = timezone.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def _sanitize_error_code(value) -> str:
    code = str(value or "").strip()
    return code if code else "UNKNOWN"


def _worker_snapshot(*, now):
    stale_seconds = max(1, int(getattr(settings, "WORKER_HEARTBEAT_STALE_SECONDS", 120)))
    stale_threshold = now - timedelta(seconds=stale_seconds)
    heartbeats = list(
        RunWorkerHeartbeat.objects.select_related("last_run", "last_run__workspace").order_by("-last_seen_at")
    )
    rows = []
    online_count = 0
    for heartbeat in heartbeats:
        is_online = heartbeat.last_seen_at >= stale_threshold
        if is_online:
            online_count += 1
        rows.append(
            {
                "heartbeat": heartbeat,
                "is_online": is_online,
                "seconds_since_seen": int((now - heartbeat.last_seen_at).total_seconds()),
            }
        )

    busy_worker_ids = set(
        InvestigationRun.objects.filter(status__in=ACTIVE_RUN_STATUSES)
        .exclude(worker_id="")
        .values_list("worker_id", flat=True)
    )
    queued_runs = InvestigationRun.objects.filter(status=RunStatus.QUEUED).count()
    active_runs = InvestigationRun.objects.filter(status__in=ACTIVE_RUN_STATUSES).count()
    return {
        "rows": rows,
        "stale_seconds": stale_seconds,
        "worker_count": len(heartbeats),
        "online_count": online_count,
        "busy_count": len(busy_worker_ids),
        "queued_runs": queued_runs,
        "active_runs": active_runs,
    }


def _concurrency_window_metrics(*, start, end):
    if end <= start:
        return {
            "window_seconds": 0,
            "all_three_busy_seconds": 0.0,
            "all_three_busy_percent": 0.0,
            "average_concurrency": 0.0,
            "peak_concurrency": 0,
        }

    run_rows = InvestigationRun.objects.filter(
        started_at__isnull=False,
        started_at__lt=end,
    ).filter(Q(finished_at__isnull=True) | Q(finished_at__gt=start))

    events: list[tuple] = []
    for run in run_rows.only("started_at", "finished_at"):
        interval_start = run.started_at or start
        interval_end = run.finished_at or end
        if interval_start < start:
            interval_start = start
        if interval_end > end:
            interval_end = end
        if interval_end <= interval_start:
            continue
        events.append((interval_start, 1))
        events.append((interval_end, -1))

    total_window_seconds = float((end - start).total_seconds())
    if not events:
        return {
            "window_seconds": total_window_seconds,
            "all_three_busy_seconds": 0.0,
            "all_three_busy_percent": 0.0,
            "average_concurrency": 0.0,
            "peak_concurrency": 0,
        }

    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    prev_ts = start
    all_three_busy_seconds = 0.0
    integrated_concurrency = 0.0

    for ts, delta in events:
        if ts > prev_ts:
            span = float((ts - prev_ts).total_seconds())
            if current >= BUSY_WORKER_TARGET:
                all_three_busy_seconds += span
            integrated_concurrency += span * float(current)
            prev_ts = ts
        current += int(delta)
        if current > peak:
            peak = current

    if prev_ts < end:
        span = float((end - prev_ts).total_seconds())
        if current >= BUSY_WORKER_TARGET:
            all_three_busy_seconds += span
        integrated_concurrency += span * float(current)

    return {
        "window_seconds": total_window_seconds,
        "all_three_busy_seconds": all_three_busy_seconds,
        "all_three_busy_percent": (
            (all_three_busy_seconds / total_window_seconds) * 100.0 if total_window_seconds > 0 else 0.0
        ),
        "average_concurrency": (
            integrated_concurrency / total_window_seconds if total_window_seconds > 0 else 0.0
        ),
        "peak_concurrency": peak,
    }


def _human_visitor_metrics(*, start, end):
    events = AuditEvent.objects.filter(
        created_at__gte=start,
        created_at__lt=end,
        action_type__in=VIEW_EVENT_ACTIONS,
    ).order_by("created_at")

    unique_ip_hashes: set[str] = set()
    unique_users: set[int] = set()
    anonymous_views = 0
    human_views = 0
    by_day: dict = defaultdict(int)

    for event in events:
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        if not bool(payload.get("is_human_view")):
            continue
        human_views += 1
        day_key = timezone.localtime(event.created_at).date()
        by_day[day_key] += 1
        if event.user_id:
            unique_users.add(int(event.user_id))
        else:
            anonymous_views += 1
        if event.ip_hash:
            unique_ip_hashes.add(str(event.ip_hash))

    day_rows = []
    cursor = timezone.localdate(start)
    final_day = timezone.localdate(end)
    while cursor <= final_day:
        day_rows.append({"date": cursor, "views": int(by_day.get(cursor, 0))})
        cursor += timedelta(days=1)
    day_rows.reverse()

    return {
        "human_views": human_views,
        "unique_ip_visitors": len(unique_ip_hashes),
        "unique_authenticated_visitors": len(unique_users),
        "anonymous_views": anonymous_views,
        "daily_rows": day_rows,
    }


def _public_share_map(*, workspace_ids, now):
    share_map: dict = {}
    if not workspace_ids:
        return share_map
    shares = (
        WorkspaceShareLink.objects.filter(
            workspace_id__in=workspace_ids,
            is_public=True,
            is_active=True,
        )
        .filter(Q(expires_at__isnull=True) | Q(expires_at__gt=now))
        .order_by("workspace_id", "-created_at")
    )
    for share in shares:
        if share.workspace_id not in share_map:
            share_map[share.workspace_id] = share
    return share_map


def _latest_workspace_dataset_artifact(*, workspace):
    for artifact_type in (
        ArtifactType.FILTERED_DATASET,
        ArtifactType.EXTRACTION_TABLE,
        ArtifactType.THEME_ASSIGNMENTS,
    ):
        artifact = (
            RunArtifact.objects.filter(
                workspace=workspace,
                status=ArtifactStatus.READY,
                artifact_type=artifact_type,
            )
            .order_by("-created_at")
            .first()
        )
        if artifact is not None:
            return artifact
    return None


def _workspace_dataset_preview(*, workspace, max_rows: int = DATASET_PREVIEW_LIMIT):
    artifact = _latest_workspace_dataset_artifact(workspace=workspace)
    empty = {
        "artifact": artifact,
        "rows": [],
        "total_rows": 0,
        "error": "",
    }
    if artifact is None:
        return empty

    try:
        file_obj, _ = open_artifact_for_download(artifact)
    except ArtifactStorageError as exc:
        return {**empty, "error": str(exc)}

    try:
        reports_df = pd.read_csv(file_obj)
    except Exception as exc:  # pragma: no cover - defensive parser guard
        return {**empty, "error": f"Could not read artifact CSV: {exc}"}
    finally:
        try:
            file_obj.close()
        except Exception:
            pass

    if reports_df.empty:
        return empty

    reports_df = with_report_identities(reports_df)
    if "date" in reports_df.columns:
        parsed_dates = pd.to_datetime(reports_df["date"], errors="coerce", utc=True)
        reports_df = (
            reports_df.assign(_sort_date=parsed_dates)
            .sort_values(by=["_sort_date"], ascending=False, na_position="last")
            .drop(columns=["_sort_date"], errors="ignore")
        )

    excluded_identities = set(
        WorkspaceReportExclusion.objects.filter(workspace=workspace).values_list("report_identity", flat=True)
    )
    rows = []
    for _, row in reports_df.head(max_rows).iterrows():
        report_identity = _safe_text(row.get(REPORT_IDENTITY_COLUMN))
        if not report_identity:
            continue
        rows.append(
            {
                "report_identity": report_identity,
                "title": _safe_text(row.get("title")) or _safe_text(row.get("investigation")),
                "date": _safe_text(row.get("date")),
                "area": _safe_text(row.get("area")),
                "receiver": _safe_text(row.get("receiver")),
                "url": _safe_text(row.get("report_url")) or _safe_text(row.get("url")),
                "is_excluded": report_identity in excluded_identities,
            }
        )

    return {
        "artifact": artifact,
        "rows": rows,
        "total_rows": int(len(reports_df.index)),
        "error": "",
    }


def _can_moderate_workspace_rows(*, user, workspace) -> bool:
    return bool(user and (user.is_superuser or can_edit_workspace(user, workspace)))


def _require_workspace_moderation_permission(*, user, workspace) -> None:
    if not _can_moderate_workspace_rows(user=user, workspace=workspace):
        raise PermissionDenied(
            "Only superusers or workspace editors can moderate workspace dataset rows."
        )


@transaction.atomic
def _apply_typed_review_edits(*, actor, run: InvestigationRun, payload: TypedRunReviewPayload):
    investigation = run.investigation
    config = dict(run.input_config_json or {})
    scope_json = dict(investigation.scope_json or {})

    plan = _build_pipeline_plan(payload, fallback_run_type=run.run_type)
    start_date, end_date, report_limit = _resolve_scope_dates(payload)

    config["pipeline_plan"] = plan
    config["pipeline_index"] = 0
    config["search_query"] = payload.search_query
    config["filter_df"] = bool(payload.filter_df)
    config["produce_spans"] = bool(payload.include_supporting_quotes or payload.extract_include_supporting_quotes)
    config["seed_topics"] = payload.seed_topics
    config["min_themes"] = payload.min_themes
    config["max_themes"] = payload.max_themes
    config["extra_theme_instructions"] = payload.extra_theme_instructions
    config["provider"] = payload.provider
    config["model_name"] = payload.model_name
    config["max_parallel_workers"] = payload.max_parallel_workers
    if RunType.EXTRACT in plan:
        config["feature_fields"] = payload.feature_fields
        config["allow_multiple"] = bool(payload.allow_multiple)
        config["force_assign"] = bool(payload.force_assign)
        config["skip_if_present"] = bool(payload.skip_if_present)
    else:
        config.pop("feature_fields", None)
        config.pop("allow_multiple", None)
        config.pop("force_assign", None)
        config.pop("skip_if_present", None)
    if report_limit is None:
        config.pop("report_limit", None)
    else:
        config["report_limit"] = report_limit

    scope_json["temporal_scope_option"] = payload.scope_option
    scope_json["query_start_date"] = start_date
    scope_json["query_end_date"] = end_date
    scope_json["report_limit"] = report_limit

    run.query_start_date = _parse_iso_date(start_date)
    run.query_end_date = _parse_iso_date(end_date)
    run.input_config_json = config
    run.save(update_fields=["query_start_date", "query_end_date", "input_config_json", "updated_at"])

    investigation.title = payload.title or investigation.title
    investigation.question_text = payload.question_text
    investigation.scope_json = scope_json
    investigation.method_json = {
        **(investigation.method_json if isinstance(investigation.method_json, dict) else {}),
        "run_filter": RunType.FILTER in plan,
        "run_themes": RunType.THEMES in plan,
        "run_extract": RunType.EXTRACT in plan,
        "pipeline_plan": plan,
    }
    investigation.save(update_fields=["title", "question_text", "scope_json", "method_json", "updated_at"])

    log_audit_event(
        action_type="ops.run_review.edited",
        target_type="investigation_run",
        target_id=str(run.id),
        workspace=run.workspace,
        user=actor,
        payload={
            "investigation_id": str(investigation.id),
            "pipeline_plan": plan,
            "provider": payload.provider,
            "model_name": payload.model_name,
            "scope_option": payload.scope_option,
        },
    )


def _approval_queryset(*, request):
    queryset = (
        InvestigationRun.objects.select_related("workspace", "requested_by", "investigation")
        .filter(
            approval_status="pending",
            status=RunStatus.QUEUED,
        )
        .order_by("-queued_at")
    )
    run_type = str(request.GET.get("type") or "").strip().lower()
    if run_type in {RunType.FILTER, RunType.THEMES, RunType.EXTRACT}:
        queryset = queryset.filter(run_type=run_type)
    provider = str(request.GET.get("provider") or "").strip().lower()
    if provider in {"local_ollama", "openai"}:
        queryset = queryset.filter(input_config_json__provider=provider)
    return queryset


@staff_member_required(login_url="admin:login")
@require_GET
def approvals(request):
    now = timezone.now()
    pending_runs = list(_approval_queryset(request=request)[:PENDING_APPROVAL_LIMIT])
    selected_run_id = str(request.GET.get("run") or "").strip()
    selected_run = None
    if selected_run_id:
        for candidate in pending_runs:
            if str(candidate.id) == selected_run_id:
                selected_run = candidate
                break

    rows = []
    for run in pending_runs:
        config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
        payload = _typed_payload_from_run(run)
        rows.append(
            {
                "run": run,
                "provider": str(config.get("provider") or "local_ollama"),
                "model_name": str(config.get("model_name") or ""),
                "queued_ago": _format_relative_delta(run.queued_at, now=now),
                "is_selected": selected_run is not None and run.id == selected_run.id,
                "payload": payload,
            }
        )
    context = {
        "ops_section": "approvals",
        "pending_rows": rows,
        "selected_run": selected_run,
        "filter_type": str(request.GET.get("type") or "").strip().lower(),
        "filter_provider": str(request.GET.get("provider") or "").strip().lower(),
    }
    return render(request, "wb_ops/approvals.html", context)


@staff_member_required(login_url="admin:login")
@require_POST
def approval_action(request, run_id):
    run = get_object_or_404(
        InvestigationRun.objects.select_related("workspace", "investigation"),
        id=run_id,
    )
    action = str(request.POST.get("action") or "").strip().lower()
    redirect_url = reverse("ops-approvals") + f"?run={run.id}"

    if action not in {"approve", "reject"}:
        messages.error(request, "Invalid run review action.")
        return redirect(redirect_url)

    if run.approval_status != "pending":
        messages.error(request, "This run is no longer pending approval.")
        return redirect(reverse("ops-approvals"))

    if action == "reject":
        reason = str(request.POST.get("approval_note") or "").strip()
        try:
            reject_run_for_execution(
                actor=request.user,
                run=run,
                reason=reason,
                request=request,
            )
            messages.success(request, "Run rejected.")
        except (PermissionDenied, ValidationError) as exc:
            messages.error(request, str(exc))
        return redirect(reverse("ops-approvals"))

    payload = _typed_payload_from_post(request)
    scheduled_for = _scheduled_for_from_post(request)
    note = str(request.POST.get("approval_note") or "").strip()
    try:
        _apply_typed_review_edits(actor=request.user, run=run, payload=payload)
        approve_run_for_execution(
            actor=request.user,
            run=run,
            note=note,
            scheduled_for=scheduled_for,
            request=request,
        )
        messages.success(request, "Run approved and queued.")
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
        return redirect(redirect_url)
    return redirect(reverse("ops-approvals"))


@staff_member_required(login_url="admin:login")
@require_GET
def failures(request):
    now = timezone.now()
    days = _coerce_lookback_days(request.GET.get("days"))
    start = now - timedelta(days=days)
    failed_runs = list(
        InvestigationRun.objects.select_related("workspace", "requested_by", "investigation")
        .filter(status=RunStatus.FAILED, created_at__gte=start)
        .order_by("-created_at")[:FAILED_RUN_LIMIT]
    )
    error_groups = (
        InvestigationRun.objects.filter(status=RunStatus.FAILED, created_at__gte=start)
        .values("error_code")
        .annotate(total=Count("id"))
        .order_by("-total")
    )
    selected_run_id = str(request.GET.get("run") or "").strip()
    selected_run = None
    for run in failed_runs:
        if str(run.id) == selected_run_id:
            selected_run = run
            break
    if selected_run is None and failed_runs:
        selected_run = failed_runs[0]

    timeline = []
    if selected_run is not None:
        for event in selected_run.events.order_by("created_at"):
            timeline.append(
                {
                    "event": event,
                    "payload_pretty": json.dumps(event.payload_json or {}, indent=2, sort_keys=True),
                }
            )

    context = {
        "ops_section": "failures",
        "lookback_days": days,
        "failed_runs": failed_runs,
        "error_groups": error_groups,
        "selected_run": selected_run,
        "timeline": timeline,
    }
    return render(request, "wb_ops/failures.html", context)


@staff_member_required(login_url="admin:login")
@require_GET
def workers(request):
    now = timezone.now()
    worker_snapshot = _worker_snapshot(now=now)
    worker_rows = []
    active_runs = list(
        InvestigationRun.objects.select_related("workspace", "requested_by", "investigation")
        .filter(status__in=ACTIVE_RUN_STATUSES)
        .order_by("-started_at")
    )
    run_by_worker = {}
    for run in active_runs:
        worker_id = str(run.worker_id or "").strip()
        if worker_id and worker_id not in run_by_worker:
            run_by_worker[worker_id] = run

    for row in worker_snapshot["rows"]:
        heartbeat = row["heartbeat"]
        assigned_run = run_by_worker.get(str(heartbeat.worker_id))
        worker_rows.append(
            {
                "heartbeat": heartbeat,
                "is_online": row["is_online"],
                "seconds_since_seen": row["seconds_since_seen"],
                "active_run": assigned_run,
            }
        )

    queue_rows = list(
        InvestigationRun.objects.select_related("workspace", "requested_by", "investigation")
        .filter(status__in={RunStatus.RUNNING, RunStatus.STARTING, RunStatus.CANCELLING, RunStatus.QUEUED})
        .order_by("-created_at")[:80]
    )
    context = {
        "ops_section": "workers",
        "worker_snapshot": worker_snapshot,
        "worker_rows": worker_rows,
        "queue_rows": queue_rows,
        "now": now,
    }
    return render(request, "wb_ops/workers.html", context)


@staff_member_required(login_url="admin:login")
@require_GET
def workspace_list(request):
    query = str(request.GET.get("q") or "").strip()
    workspaces = Workspace.objects.select_related("created_by").annotate(
        member_count=Count("memberships", distinct=True),
        run_count=Count("runs", distinct=True),
    )
    if query:
        workspaces = workspaces.filter(
            Q(title__icontains=query)
            | Q(slug__icontains=query)
            | Q(created_by__email__icontains=query)
        )
    workspaces = workspaces.order_by("-updated_at")[:200]
    context = {
        "ops_section": "workspaces",
        "workspace_rows": workspaces,
        "query": query,
    }
    return render(request, "wb_ops/workspaces.html", context)


@staff_member_required(login_url="admin:login")
@require_GET
def dashboard(request):
    now = timezone.now()
    lookback_days = _coerce_lookback_days(request.GET.get("days"))
    start = now - timedelta(days=lookback_days)
    worker_snapshot = _worker_snapshot(now=now)
    concurrency = _concurrency_window_metrics(start=start, end=now)
    visitors = _human_visitor_metrics(start=start, end=now)

    recent_runs = list(
        InvestigationRun.objects.select_related("workspace", "requested_by", "investigation").order_by("-created_at")[
            :RECENT_RUN_LIMIT
        ]
    )
    workspaces = list(
        Workspace.objects.select_related("created_by")
        .annotate(
            member_count=Count("memberships", distinct=True),
            run_count=Count("runs", distinct=True),
        )
        .order_by("-updated_at")[:20]
    )
    workspace_ids = [workspace.id for workspace in workspaces]
    latest_run_map: dict = {}
    if workspace_ids:
        for run in (
            InvestigationRun.objects.filter(workspace_id__in=workspace_ids)
            .select_related("requested_by")
            .order_by("workspace_id", "-created_at")
        ):
            if run.workspace_id not in latest_run_map:
                latest_run_map[run.workspace_id] = run

    share_map = _public_share_map(workspace_ids=workspace_ids, now=now)
    workspace_rows = []
    for workspace in workspaces:
        share_link = share_map.get(workspace.id)
        workspace_rows.append(
            {
                "workspace": workspace,
                "latest_run": latest_run_map.get(workspace.id),
                "public_share_link": share_link,
                "public_share_url": (
                    request.build_absolute_uri(
                        reverse("share-link-detail", kwargs={"share_id": share_link.id})
                    )
                    if share_link is not None
                    else ""
                ),
            }
        )

    top_users = list(
        User.objects.annotate(
            runs_in_window=Count(
                "requested_runs",
                filter=Q(requested_runs__created_at__gte=start),
                distinct=True,
            )
        )
        .filter(runs_in_window__gt=0)
        .order_by("-runs_in_window", "email")[:8]
    )

    context = {
        "lookback_days": lookback_days,
        "users_total": User.objects.count(),
        "users_staff": User.objects.filter(is_staff=True).count(),
        "workspaces_total": Workspace.objects.count(),
        "workspaces_public": Workspace.objects.filter(visibility="public", is_listed=True).count(),
        "runs_total": InvestigationRun.objects.count(),
        "runs_last_24h": InvestigationRun.objects.filter(created_at__gte=now - timedelta(hours=24)).count(),
        "recent_runs": recent_runs,
        "workspace_rows": workspace_rows,
        "top_users": top_users,
        "worker_snapshot": worker_snapshot,
        "concurrency": concurrency,
        "visitors": visitors,
    }
    return render(request, "wb_ops/dashboard.html", context)


@staff_member_required(login_url="admin:login")
@require_GET
def user_list(request):
    users = (
        User.objects.annotate(
            workspace_count=Count("workspace_memberships__workspace", distinct=True),
            owned_workspace_count=Count("created_workspaces", distinct=True),
            run_count=Count("requested_runs", distinct=True),
            active_run_count=Count(
                "requested_runs",
                filter=Q(requested_runs__status__in=ACTIVE_RUN_STATUSES),
                distinct=True,
            ),
            last_activity=Max("audit_events__created_at"),
        )
        .order_by("-last_activity", "email")
    )
    return render(
        request,
        "wb_ops/users.html",
        {
            "ops_section": "users",
            "users": users,
        },
    )


@staff_member_required(login_url="admin:login")
@require_GET
def user_detail(request, user_id):
    target_user = get_object_or_404(User, id=user_id)
    memberships = list(
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=target_user)
        .order_by("-workspace__updated_at")
    )
    runs = list(
        InvestigationRun.objects.select_related("workspace", "investigation")
        .filter(requested_by=target_user)
        .order_by("-created_at")[:USER_RUN_LIMIT]
    )
    run_rows = [
        {
            "run": run,
            "input_config_json": json.dumps(run.input_config_json or {}, indent=2, sort_keys=True),
        }
        for run in runs
    ]
    run_status_counts = (
        InvestigationRun.objects.filter(requested_by=target_user)
        .values("status")
        .annotate(total=Count("id"))
        .order_by("status")
    )
    recent_activity = list(
        AuditEvent.objects.filter(user=target_user)
        .order_by("-created_at")[:20]
    )

    return render(
        request,
        "wb_ops/user_detail.html",
        {
            "ops_section": "users",
            "target_user": target_user,
            "memberships": memberships,
            "run_rows": run_rows,
            "run_total": InvestigationRun.objects.filter(requested_by=target_user).count(),
            "run_status_counts": run_status_counts,
            "recent_activity": recent_activity,
        },
    )


@staff_member_required(login_url="admin:login")
@require_GET
def workspace_detail(request, workspace_id):
    workspace = get_object_or_404(Workspace.objects.select_related("created_by"), id=workspace_id)
    investigation = Investigation.objects.filter(workspace=workspace).first()
    runs = list(
        InvestigationRun.objects.select_related("requested_by", "investigation")
        .filter(workspace=workspace)
        .order_by("-created_at")[:WORKSPACE_RUN_LIMIT]
    )
    run_rows = [
        {
            "run": run,
            "input_config_json": json.dumps(run.input_config_json or {}, indent=2, sort_keys=True),
        }
        for run in runs
    ]
    exclusions = list(
        WorkspaceReportExclusion.objects.select_related("excluded_by")
        .filter(workspace=workspace)
        .order_by("-created_at")
    )
    dataset_preview = _workspace_dataset_preview(workspace=workspace)
    public_shares = list(
        WorkspaceShareLink.objects.filter(
            workspace=workspace,
            is_public=True,
            is_active=True,
        )
        .filter(Q(expires_at__isnull=True) | Q(expires_at__gt=timezone.now()))
        .order_by("-created_at")[:5]
    )
    public_share_rows = [
        {
            "share": share,
            "url": request.build_absolute_uri(
                reverse("share-link-detail", kwargs={"share_id": share.id})
            ),
        }
        for share in public_shares
    ]
    investigation_payload = {
        "scope_json": investigation.scope_json if investigation else {},
        "method_json": investigation.method_json if investigation else {},
    }
    recent_edits = list(
        AuditEvent.objects.filter(workspace=workspace)
        .order_by("-created_at")[:20]
    )
    return render(
        request,
        "wb_ops/workspace_detail.html",
        {
            "ops_section": "workspaces",
            "workspace": workspace,
            "investigation": investigation,
            "investigation_payload_json": json.dumps(
                investigation_payload,
                indent=2,
                sort_keys=True,
            ),
            "run_rows": run_rows,
            "exclusions": exclusions,
            "dataset_preview": dataset_preview,
            "public_share_rows": public_share_rows,
            "user_can_moderate": _can_moderate_workspace_rows(user=request.user, workspace=workspace),
            "workspace_dashboard_url": reverse("workbook-open", kwargs={"workbook_id": workspace.id}),
            "recent_edits": recent_edits,
        },
    )


@staff_member_required(login_url="admin:login")
@require_POST
def exclude_workspace_row(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    _require_workspace_moderation_permission(user=request.user, workspace=workspace)

    report_identity = str(request.POST.get("report_identity") or "").strip()
    report_title = str(request.POST.get("report_title") or "").strip()
    report_date = str(request.POST.get("report_date") or "").strip()
    report_url = str(request.POST.get("report_url") or "").strip()
    reason = str(request.POST.get("reason") or "").strip() or "Removed from ops moderation interface."
    try:
        upsert_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            report_identity=report_identity,
            reason=reason,
            report_title=report_title,
            report_date=report_date,
            report_url=report_url,
            request=request,
        )
    except (WorkspaceReportExclusionError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Dataset row excluded from this workspace.")
    return redirect("ops-workspace-detail", workspace_id=workspace.id)


@staff_member_required(login_url="admin:login")
@require_POST
def restore_workspace_exclusion(request, workspace_id, exclusion_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    _require_workspace_moderation_permission(user=request.user, workspace=workspace)
    exclusion = get_object_or_404(
        WorkspaceReportExclusion,
        id=exclusion_id,
        workspace=workspace,
    )
    try:
        restore_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            exclusion=exclusion,
            request=request,
        )
    except (WorkspaceReportExclusionError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Excluded report restored.")
    return redirect("ops-workspace-detail", workspace_id=workspace.id)
