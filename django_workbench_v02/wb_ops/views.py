from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from wb_auditlog.models import AuditEvent
from wb_investigations.models import Investigation
from wb_runs.artifact_storage import ArtifactStorageError, open_artifact_for_download
from wb_runs.models import (
    ArtifactStatus,
    ArtifactType,
    InvestigationRun,
    RunArtifact,
    RunStatus,
    RunWorkerHeartbeat,
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
        )
        .order_by("-is_superuser", "-is_staff", "email")
    )
    return render(request, "wb_ops/users.html", {"users": users})


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

    return render(
        request,
        "wb_ops/user_detail.html",
        {
            "target_user": target_user,
            "memberships": memberships,
            "run_rows": run_rows,
            "run_total": InvestigationRun.objects.filter(requested_by=target_user).count(),
            "run_status_counts": run_status_counts,
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
    return render(
        request,
        "wb_ops/workspace_detail.html",
        {
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
