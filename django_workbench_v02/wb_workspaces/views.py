from datetime import timedelta
import csv
import json
from io import StringIO, TextIOWrapper

import pandas as pd
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Q
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from accounts.views import (
    EXPLORE_REPORTS_PAGE_SIZE,
    _coerce_positive_int,
    _explore_column_label,
    _explore_ordered_columns,
    _explore_report_rows,
    _explore_shared_querydict,
)
from wb_investigations.models import Investigation
from wb_collections.services import (
    _network_truthy,
    _parse_report_dates,
    _split_receivers,
    _theme_collection_map_from_reports,
    build_explore_metrics,
    reports_for_collection,
)
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
from wb_sharing.forms import ShareLinkCreateForm
from wb_sharing.models import ShareMode, WorkspaceShareLink

from .activity import is_human_view_request, should_update_last_viewed
from .forms import (
    ActiveLLMCredentialDeleteForm,
    ActiveLLMConfigForm,
    WorkspaceCreateForm,
    WorkspaceCredentialDeleteForm,
    WorkspaceCredentialUpsertForm,
    WorkspaceMemberAddForm,
    WorkspaceReportExclusionCreateForm,
    WorkspaceMemberUpdateForm,
)
from .models import (
    MembershipAccessMode,
    MembershipRole,
    Workspace,
    WorkspaceCredential,
    WorkspaceMembership,
    WorkspaceReportExclusion,
    WorkspaceVisibility,
)
from .permissions import can_edit_workspace, can_manage_members, can_manage_shares, can_run_workflows, can_view_workspace
from .services import (
    WorkspaceCredentialValidationError,
    WorkspaceLifecycleError,
    WorkspaceMembershipError,
    WorkspaceReportExclusionError,
    add_workspace_member,
    create_workspace_for_user,
    delete_user_llm_credential,
    delete_workspace_credential,
    delete_workspace_immediately,
    get_active_workspace_for_user,
    remove_workspace_member,
    archive_workspace,
    restore_workspace,
    set_active_workspace_for_user,
    restore_workspace_report_exclusion,
    upsert_workspace_report_exclusion,
    upsert_workspace_credential,
    upsert_workspace_llm_setting,
    upsert_user_llm_credential,
    upsert_user_llm_setting,
    update_workspace_member,
)
from .revisions import (
    WorkspaceRevisionError,
    redo_workspace_revision,
    revert_workspace_reports,
    start_over_workspace_state,
    undo_workspace_revision,
)


User = get_user_model()


PIPELINE_PENDING_STATUSES = {
    RunStatus.QUEUED,
    RunStatus.STARTING,
    RunStatus.RUNNING,
    RunStatus.CANCELLING,
}
PIPELINE_FAILED_STATUSES = {
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
    RunStatus.CANCELLED,
}
RUN_TYPE_LABELS = {
    "filter": "Filtering",
    "themes": "Themes",
    "extract": "Extracting",
    "export": "Exporting",
}
RUN_STATUS_CARD_LABELS = {
    RunStatus.QUEUED: "Queued",
    RunStatus.STARTING: "Starting",
    RunStatus.RUNNING: "Running",
    RunStatus.CANCELLING: "Cancelling",
    RunStatus.CANCELLED: "Cancelled",
    RunStatus.SUCCEEDED: "Complete",
    RunStatus.FAILED: "Failed",
    RunStatus.TIMED_OUT: "Timed out",
}


def _scope_option_from_dates(*, start: str, end: str, report_limit) -> str:
    if report_limit == 100:
        return "most_recent_100"
    start_value = str(start or "").strip()
    end_value = str(end or "").strip()
    if start_value and end_value:
        return "custom_range"
    return "all_reports"


def _wizard_copy_prefill(
    *,
    workspace: Workspace,
    investigation: Investigation | None,
    latest_run: InvestigationRun | None,
) -> dict:
    config = latest_run.input_config_json if latest_run and isinstance(latest_run.input_config_json, dict) else {}
    scope_json = (
        investigation.scope_json
        if investigation is not None and isinstance(investigation.scope_json, dict)
        else {}
    )
    method_json = (
        investigation.method_json
        if investigation is not None and isinstance(investigation.method_json, dict)
        else {}
    )

    pipeline_plan = (
        config.get("pipeline_plan")
        if isinstance(config.get("pipeline_plan"), list)
        else (method_json.get("pipeline_plan") if isinstance(method_json.get("pipeline_plan"), list) else [])
    )
    has_filter = bool(method_json.get("run_filter", RunType.FILTER in pipeline_plan))
    has_themes = bool(method_json.get("run_themes", RunType.THEMES in pipeline_plan))
    has_extract = bool(method_json.get("run_extract", RunType.EXTRACT in pipeline_plan))
    run_filter = has_filter or RunType.FILTER in pipeline_plan
    run_themes = has_themes or RunType.THEMES in pipeline_plan
    run_extract = has_extract or RunType.EXTRACT in pipeline_plan

    scope_start = str(scope_json.get("query_start_date") or "")
    scope_end = str(scope_json.get("query_end_date") or "")
    scope_option = str(scope_json.get("temporal_scope_option") or "").strip() or _scope_option_from_dates(
        start=scope_start,
        end=scope_end,
        report_limit=config.get("report_limit") or scope_json.get("report_limit"),
    )

    selected_filters = config.get("selected_filters") if isinstance(config.get("selected_filters"), dict) else {}
    feature_fields = config.get("feature_fields") if isinstance(config.get("feature_fields"), list) else []
    sanitised_features: list[dict[str, str]] = []
    for row in feature_fields:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("field_name") or "").strip()
        description = str(row.get("description") or "").strip()
        field_type = str(row.get("type") or "text").strip().lower() or "text"
        if field_type == "integer":
            field_type = "decimal"
        if field_type not in {"text", "decimal", "boolean"}:
            field_type = "text"
        sanitised_features.append({"name": name, "description": description, "type": field_type})

    try:
        max_parallel_workers = int(config.get("max_parallel_workers") or 1)
    except (TypeError, ValueError):
        max_parallel_workers = 1

    title = (
        str(investigation.title).strip()
        if investigation is not None and str(investigation.title or "").strip()
        else str(workspace.title or "").strip()
    )
    question_text = (
        str(investigation.question_text).strip()
        if investigation is not None and str(investigation.question_text or "").strip()
        else str(workspace.description or "").strip()
    )

    return {
        "title": title,
        "question_text": question_text,
        "scope_option": scope_option,
        "custom_start_date": scope_start,
        "custom_end_date": scope_end,
        "run_filter": bool(run_filter),
        "run_themes": bool(run_themes),
        "run_extract": bool(run_extract),
        "search_query": str(config.get("search_query") or question_text).strip(),
        "filter_df": bool(config.get("filter_df", True)),
        "include_supporting_quotes": bool(config.get("produce_spans", False)),
        "coroner_filters": ", ".join(selected_filters.get("coroner", []) or []),
        "area_filters": ", ".join(selected_filters.get("area", []) or []),
        "receiver_filters": ", ".join(selected_filters.get("receiver", []) or []),
        "seed_topics": str(config.get("seed_topics") or "").strip(),
        "min_themes": config.get("min_themes"),
        "max_themes": config.get("max_themes"),
        "extra_theme_instructions": str(config.get("extra_theme_instructions") or "").strip(),
        "feature_fields": json.dumps(sanitised_features),
        "allow_multiple": bool(config.get("allow_multiple", False)),
        "force_assign": bool(config.get("force_assign", False)),
        "skip_if_present": bool(config.get("skip_if_present", True)),
        "extract_include_supporting_quotes": bool(config.get("produce_spans", False)),
        "provider": str(config.get("provider") or "openai").strip().lower(),
        "model_name": str(config.get("model_name") or "gpt-4.1-mini").strip(),
        "max_parallel_workers": max_parallel_workers,
        "request_completion_email": True,
    }


def _pipeline_status_for_run(run: InvestigationRun | None) -> str:
    if run is None:
        return ""
    if run.status in PIPELINE_PENDING_STATUSES:
        return "pending"
    if run.status in PIPELINE_FAILED_STATUSES:
        return "failed-warning"
    if run.status == RunStatus.SUCCEEDED:
        return "complete"
    return ""


def _reports_found_from_artifact(artifact: RunArtifact | None) -> int:
    if artifact is None:
        return 0
    if isinstance(artifact.metadata_json, dict):
        metadata = artifact.metadata_json
        for key in ("matched_reports", "output_reports", "total_reports"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _active_public_share_link_for_workspace(*, workspace: Workspace) -> WorkspaceShareLink | None:
    now = timezone.now()
    return (
        WorkspaceShareLink.objects.filter(
            workspace=workspace,
            is_active=True,
            is_public=True,
        )
        .filter(Q(expires_at__isnull=True) | Q(expires_at__gt=now))
        .order_by("-created_at")
        .first()
    )


def _worker_health(*, workspace_ids: list | None = None) -> tuple[bool, str]:
    stale_seconds = max(1, int(getattr(settings, "WORKER_HEARTBEAT_STALE_SECONDS", 120)))
    threshold = timezone.now() - timedelta(seconds=stale_seconds)
    scope_ids = [workspace_id for workspace_id in (workspace_ids or []) if workspace_id]

    pending_qs = InvestigationRun.objects.filter(status__in=PIPELINE_PENDING_STATUSES)
    active_qs = InvestigationRun.objects.filter(
        status__in=[RunStatus.STARTING, RunStatus.RUNNING, RunStatus.CANCELLING]
    )
    if scope_ids:
        pending_qs = pending_qs.filter(workspace_id__in=scope_ids)
        active_qs = active_qs.filter(workspace_id__in=scope_ids)

    # If there are no in-flight runs for this dashboard scope, do not show worker warnings.
    if not pending_qs.exists():
        return (True, "")

    # If any active run has recent updates, treat worker as effectively online.
    # Use a global active-run check (not workspace-scoped), because a single worker can
    # be healthy while currently busy on a different workspace.
    if InvestigationRun.objects.filter(
        status__in=[RunStatus.STARTING, RunStatus.RUNNING, RunStatus.CANCELLING],
        updated_at__gte=threshold,
    ).exists():
        return (True, "")

    latest = RunWorkerHeartbeat.objects.order_by("-last_seen_at").first()
    if latest is None:
        return (
            False,
            "Worker offline: runs will stay queued until `run_runs_worker` is running.",
        )
    if latest.last_seen_at < threshold:
        return (
            False,
            "Worker heartbeat is stale: restart `run_runs_worker` to process queued runs.",
        )
    return (True, f"Worker online ({latest.worker_id}).")


def _artifact_preview(artifact: RunArtifact | None, *, max_rows: int = 10) -> dict:
    empty = {"columns": [], "rows": [], "error": "", "row_count": 0}
    if artifact is None:
        return empty
    try:
        file_obj, _ = open_artifact_for_download(artifact)
    except ArtifactStorageError as exc:
        return {**empty, "error": str(exc)}

    columns: list[str] = []
    rows: list[list[str]] = []
    row_count = 0
    try:
        if hasattr(file_obj, "readable") and not isinstance(file_obj, TextIOWrapper):
            text_stream = TextIOWrapper(file_obj, encoding="utf-8", errors="replace")
        else:
            text_stream = file_obj
        reader = csv.reader(text_stream)
        for idx, row in enumerate(reader):
            if idx == 0:
                columns = [str(cell or "").strip() for cell in row]
                continue
            row_count += 1
            if len(rows) < max_rows:
                rows.append([str(cell or "").strip() for cell in row])
    except Exception as exc:
        return {**empty, "error": f"Could not read artifact preview: {exc}"}
    finally:
        try:
            file_obj.close()
        except Exception:
            pass
    return {"columns": columns, "rows": rows, "error": "", "row_count": row_count}


def _artifact_dataframe(artifact: RunArtifact | None) -> pd.DataFrame:
    if artifact is None:
        return pd.DataFrame()
    try:
        file_obj, _ = open_artifact_for_download(artifact)
    except ArtifactStorageError:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_obj)
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            file_obj.close()
        except Exception:
            pass


def _pipeline_dashboard_context(*, workspace: Workspace, request) -> dict:
    investigation = Investigation.objects.filter(workspace=workspace).first()
    runs = list(
        InvestigationRun.objects.filter(workspace=workspace)
        .select_related("investigation")
        .order_by("-created_at")
    )
    latest_run = runs[0] if runs else None
    latest_success = next((run for run in runs if run.status == RunStatus.SUCCEEDED), None)

    def latest_artifact(artifact_type: str) -> RunArtifact | None:
        return (
            RunArtifact.objects.filter(
                workspace=workspace,
                status=ArtifactStatus.READY,
                artifact_type=artifact_type,
            )
            .order_by("-created_at")
            .first()
        )

    filtered_artifact = latest_artifact(ArtifactType.FILTERED_DATASET)
    theme_summary_artifact = latest_artifact(ArtifactType.THEME_SUMMARY)
    extraction_artifact = latest_artifact(ArtifactType.EXTRACTION_TABLE)

    filtered_meta = filtered_artifact.metadata_json if filtered_artifact and isinstance(filtered_artifact.metadata_json, dict) else {}
    theme_meta = theme_summary_artifact.metadata_json if theme_summary_artifact and isinstance(theme_summary_artifact.metadata_json, dict) else {}
    extract_meta = extraction_artifact.metadata_json if extraction_artifact and isinstance(extraction_artifact.metadata_json, dict) else {}

    filtered_preview = _artifact_preview(filtered_artifact)
    theme_preview = _artifact_preview(theme_summary_artifact)
    extract_preview = _artifact_preview(extraction_artifact)
    filtered_df = _artifact_dataframe(filtered_artifact)

    reports_count = int(filtered_meta.get("matched_reports") or filtered_preview["row_count"] or 0)
    themes_count = int(theme_meta.get("discovered_themes") or theme_preview["row_count"] or 0)
    extract_rows = int(extract_meta.get("output_reports") or extract_preview["row_count"] or 0)
    if not filtered_df.empty:
        reports_count = int(len(filtered_df))

    latest_filter_run = next((run for run in runs if run.run_type == "filter"), None)
    search_query = ""
    if latest_filter_run and isinstance(latest_filter_run.input_config_json, dict):
        search_query = str(latest_filter_run.input_config_json.get("search_query") or "").strip()

    if filtered_df.empty:
        explore = {
            "query": search_query,
            "has_query": bool(search_query),
            "total_reports": 0,
            "scope_reports": 0,
            "date_range_label": "Date range unavailable",
            "scope_label": "Filtered pipeline result",
            "top_areas": [],
            "top_receivers": [],
            "top_themes": [],
            "temporal_line_path": "",
            "temporal_area_path": "",
            "temporal_axis_labels": [],
            "temporal_start_label": "",
            "temporal_end_label": "",
            "temporal_series": {"month": {"points": []}},
            "unique_receiver_count": 0,
            "most_common_theme": "No theme signals",
        }
    else:
        explore = build_explore_metrics(
            reports_df=filtered_df,
            scoped_reports_df=filtered_df,
            query=search_query,
        )
        explore["scope_label"] = "Filtered pipeline result"

    requested_page = _coerce_positive_int(request.GET.get("page"), default=1)
    reports_columns: list[str] = []
    reports_rows: list[dict[str, object]] = []
    reports_total = int(len(filtered_df))
    reports_total_pages = 1
    reports_page_from = 0
    reports_page_to = 0
    reports_prev_page = None
    reports_next_page = None
    reports_page = 1
    if reports_total > 0:
        reports_columns = _explore_ordered_columns(filtered_df)
        ordered_column_labels = [_explore_column_label(column_name) for column_name in reports_columns]
        display_df = filtered_df.copy()
        if "date" in display_df.columns:
            display_df["_sort_date"] = _parse_report_dates(
                display_df.get("date", pd.Series(dtype="object"))
            )
            display_df = display_df.sort_values(
                by=["_sort_date"],
                ascending=False,
                na_position="last",
            ).drop(columns=["_sort_date"], errors="ignore")

        reports_total_pages = max(1, int((reports_total + EXPLORE_REPORTS_PAGE_SIZE - 1) / EXPLORE_REPORTS_PAGE_SIZE))
        reports_page = max(1, min(requested_page, reports_total_pages))
        start_index = (reports_page - 1) * EXPLORE_REPORTS_PAGE_SIZE
        end_index = min(start_index + EXPLORE_REPORTS_PAGE_SIZE, reports_total)
        page_df = display_df.iloc[start_index:end_index].copy()
        reports_rows = _explore_report_rows(reports_df=page_df, ordered_columns=reports_columns)
        reports_columns = ordered_column_labels
        reports_page_from = start_index + 1
        reports_page_to = end_index
        reports_prev_page = reports_page - 1 if reports_page > 1 else None
        reports_next_page = reports_page + 1 if reports_page < reports_total_pages else None

    run_detail_items = []
    for run in runs[:12]:
        run_detail_items.append(
            {
                "id": str(run.id),
                "status": run.status,
                "type": run.run_type,
                "queued_at": run.queued_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "worker_id": run.worker_id,
                "error_code": run.error_code,
                "error_message": run.error_message,
                "input_config_json": json.dumps(run.input_config_json or {}, indent=2, sort_keys=True),
            }
        )

    return {
        "workspace": workspace,
        "investigation": investigation,
        "latest_run": latest_run,
        "latest_success": latest_success,
        "runs": runs[:8],
        "reports_count": reports_count,
        "themes_count": themes_count,
        "extract_rows": extract_rows,
        "filtered_preview": filtered_preview,
        "themes_preview": theme_preview,
        "extract_preview": extract_preview,
        "has_results": bool(filtered_artifact or theme_summary_artifact or extraction_artifact),
        "explore": explore,
        "reports_columns": reports_columns,
        "reports_rows": reports_rows,
        "reports_count": reports_total,
        "reports_page": reports_page,
        "reports_total_pages": reports_total_pages,
        "reports_page_from": reports_page_from,
        "reports_page_to": reports_page_to,
        "reports_prev_page": reports_prev_page,
        "reports_next_page": reports_next_page,
        "debug_runs": run_detail_items,
        "search_query": search_query,
    }


@require_http_methods(["GET", "POST"])
def dashboard(request):
    if not request.user.is_authenticated:
        public_workbooks = Workspace.objects.filter(
            visibility=WorkspaceVisibility.PUBLIC,
            is_listed=True,
            archived_at__isnull=True,
        ).order_by("-updated_at")[:6]
        return render(
            request,
            "wb_workspaces/workbooks_guest.html",
            {"public_workbooks": public_workbooks},
        )

    memberships = (
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=request.user)
        .order_by("-workspace__updated_at")
    )
    memberships_list = list(memberships)
    dashboard_rows: list[dict] = []
    for membership in memberships_list:
        workspace = membership.workspace
        can_manage_lifecycle = bool(request.user.is_superuser) or (
            membership.role == MembershipRole.OWNER
            and membership.access_mode == MembershipAccessMode.EDIT
            and membership.can_manage_members
        )
        dashboard_rows.append(
            {
                "membership": membership,
                "workspace": workspace,
                "is_archived": workspace.archived_at is not None,
                "can_restore": can_manage_lifecycle,
                "can_archive": can_manage_lifecycle and workspace.archived_at is None,
                "can_hard_delete": can_manage_lifecycle and workspace.archived_at is not None,
                "can_cancel_running_run": False,
                "running_run": None,
                "pending_stage_label": "",
                "pipeline_state": "",
                "pipeline_stage_label": "",
                "pipeline_run_status": "",
                "pipeline_status_label": "Not started",
                "pipeline_updated_at": None,
                "investigation_title": workspace.title,
                "investigation_description": "",
                "reports_found_count": 0,
                "show_reports_found_count": False,
                "last_edited_at": workspace.updated_at,
            }
        )
    workspace_ids = [row["workspace"].id for row in dashboard_rows]
    latest_run_by_workspace_id: dict = {}
    investigations_by_workspace_id: dict = {}
    reports_found_by_workspace_id: dict = {}
    if workspace_ids:
        investigations = Investigation.objects.filter(workspace_id__in=workspace_ids).only(
            "workspace_id",
            "title",
            "question_text",
            "updated_at",
        )
        for investigation in investigations:
            investigations_by_workspace_id[investigation.workspace_id] = investigation

        runs = (
            InvestigationRun.objects.select_related("investigation")
            .filter(workspace_id__in=workspace_ids)
            .order_by("workspace_id", "-created_at")
        )
        for run in runs:
            key = run.workspace_id
            if key not in latest_run_by_workspace_id:
                latest_run_by_workspace_id[key] = run

        filtered_artifacts = (
            RunArtifact.objects.filter(
                workspace_id__in=workspace_ids,
                status=ArtifactStatus.READY,
                artifact_type=ArtifactType.FILTERED_DATASET,
            )
            .only("workspace_id", "metadata_json")
            .order_by("workspace_id", "-created_at")
        )
        for artifact in filtered_artifacts:
            key = artifact.workspace_id
            if key not in reports_found_by_workspace_id:
                reports_found_by_workspace_id[key] = _reports_found_from_artifact(artifact)

    for row in dashboard_rows:
        workspace_id = row["workspace"].id
        investigation = investigations_by_workspace_id.get(workspace_id)
        if investigation is not None:
            row["investigation_title"] = str(investigation.title or row["workspace"].title).strip() or row["workspace"].title
            row["investigation_description"] = str(investigation.question_text or "").strip()
            row["last_edited_at"] = investigation.updated_at

        run = latest_run_by_workspace_id.get(workspace_id)
        row["latest_run"] = run
        row["latest_investigation_id"] = str(run.investigation_id) if run else ""
        if run:
            stage_label = RUN_TYPE_LABELS.get(str(run.run_type), str(run.run_type).title())
            row["pipeline_state"] = _pipeline_status_for_run(run)
            row["pipeline_stage_label"] = stage_label
            row["pipeline_run_status"] = str(run.status)
            row["pipeline_status_label"] = RUN_STATUS_CARD_LABELS.get(run.status, str(run.status).replace("_", " ").title())
            row["pipeline_updated_at"] = run.updated_at
            if run.updated_at and run.updated_at > row["last_edited_at"]:
                row["last_edited_at"] = run.updated_at
        if run and run.status == RunStatus.RUNNING:
            row["running_run"] = run
            row["pending_stage_label"] = RUN_TYPE_LABELS.get(str(run.run_type), str(run.run_type).title())
            row["can_cancel_running_run"] = bool(can_run_workflows(request.user, row["workspace"]))

        reports_found_count = int(reports_found_by_workspace_id.get(workspace_id, 0))
        row["reports_found_count"] = reports_found_count
        row["show_reports_found_count"] = row["pipeline_state"] == "complete"

        if row["pipeline_updated_at"] and row["pipeline_updated_at"] > row["last_edited_at"]:
            row["last_edited_at"] = row["pipeline_updated_at"]

        copy_prefill = _wizard_copy_prefill(
            workspace=row["workspace"],
            investigation=investigation,
            latest_run=run,
        )
        row["copy_prefill"] = copy_prefill
        row["copy_prefill_id"] = f"workspace-copy-prefill-{workspace_id}"
    active_workspace = get_active_workspace_for_user(user=request.user)
    if active_workspace is None and memberships_list:
        for membership in memberships_list:
            if membership.workspace.archived_at is None:
                active_workspace = membership.workspace
                set_active_workspace_for_user(
                    user=request.user,
                    workspace=active_workspace,
                    request=request,
                )
                break
    active_workspace_id = str(active_workspace.id) if active_workspace is not None else ""

    worker_ready, worker_status_message = _worker_health(workspace_ids=workspace_ids)

    return render(
        request,
        "wb_workspaces/dashboard.html",
        {
            "memberships": memberships_list,
            "active_rows": [row for row in dashboard_rows if not row["is_archived"]],
            "archived_rows": [row for row in dashboard_rows if row["is_archived"]],
            "active_workspace_id": active_workspace_id,
            "worker_ready": worker_ready,
            "worker_status_message": worker_status_message,
        },
    )


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


def _workspace_explore_params(request) -> dict[str, object]:
    return {
        "query": str(request.GET.get("q") or "").strip(),
        "selected_receivers": _dedupe_values(request.GET.getlist("receiver")),
        "selected_areas": _dedupe_values(request.GET.getlist("area")),
        "date_start_raw": str(request.GET.get("date_start") or "").strip(),
        "date_end_raw": str(request.GET.get("date_end") or "").strip(),
    }


def _latest_workspace_artifact(*, workspace: Workspace, artifact_type: str) -> RunArtifact | None:
    return (
        RunArtifact.objects.filter(
            workspace=workspace,
            status=ArtifactStatus.READY,
            artifact_type=artifact_type,
        )
        .order_by("-created_at")
        .first()
    )


def _workspace_dataset_artifact(*, workspace: Workspace) -> RunArtifact | None:
    return (
        RunArtifact.objects.filter(
            workspace=workspace,
            status=ArtifactStatus.READY,
            artifact_type__in=(
                ArtifactType.EXTRACTION_TABLE,
                ArtifactType.THEME_ASSIGNMENTS,
                ArtifactType.FILTERED_DATASET,
            ),
        )
        .order_by("-created_at")
        .first()
    )


def _pipeline_theme_column_allowlist(*, workspace: Workspace) -> set[str]:
    theme_summary_artifact = _latest_workspace_artifact(
        workspace=workspace,
        artifact_type=ArtifactType.THEME_SUMMARY,
    )
    summary_df = _artifact_dataframe(theme_summary_artifact)
    if summary_df.empty or "theme" not in summary_df.columns:
        return set()
    allowlist: set[str] = set()
    for raw in summary_df["theme"].tolist():
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        allowlist.add(cleaned)
        allowlist.add(cleaned.casefold())
        if not cleaned.startswith("theme_"):
            allowlist.add(f"theme_{cleaned}")
            allowlist.add(f"theme_{cleaned}".casefold())
    return allowlist


def _pipeline_theme_label_map(*, workspace: Workspace) -> dict[str, str]:
    theme_summary_artifact = _latest_workspace_artifact(
        workspace=workspace,
        artifact_type=ArtifactType.THEME_SUMMARY,
    )
    summary_df = _artifact_dataframe(theme_summary_artifact)
    if summary_df.empty or "theme" not in summary_df.columns:
        return {}
    label_map: dict[str, str] = {}
    for raw in summary_df["theme"].tolist():
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        normalized = cleaned.casefold()
        prefixed = cleaned if cleaned.startswith("theme_") else f"theme_{cleaned}"
        label = cleaned
        label_map[cleaned] = label
        label_map[normalized] = label
        label_map[prefixed] = label
        label_map[prefixed.casefold()] = label
    return label_map


def _pipeline_theme_metadata_map(*, workspace: Workspace) -> dict[str, dict[str, str]]:
    theme_summary_artifact = _latest_workspace_artifact(
        workspace=workspace,
        artifact_type=ArtifactType.THEME_SUMMARY,
    )
    summary_df = _artifact_dataframe(theme_summary_artifact)
    if summary_df.empty or "theme" not in summary_df.columns:
        return {}
    metadata_map: dict[str, dict[str, str]] = {}
    for _, row in summary_df.iterrows():
        cleaned = str(row.get("theme") or "").strip()
        if not cleaned:
            continue
        description = str(row.get("description") or "").strip()
        normalized = cleaned.casefold()
        prefixed = cleaned if cleaned.startswith("theme_") else f"theme_{cleaned}"
        payload = {
            "label": cleaned,
            "description": description,
        }
        metadata_map[cleaned] = payload
        metadata_map[normalized] = payload
        metadata_map[prefixed] = payload
        metadata_map[prefixed.casefold()] = payload
    return metadata_map


def _prune_non_pipeline_theme_columns(*, workspace: Workspace, reports_df: pd.DataFrame) -> pd.DataFrame:
    if reports_df.empty:
        return reports_df
    theme_columns = [str(column) for column in reports_df.columns if str(column).startswith("theme_")]
    if not theme_columns:
        return reports_df
    allowlist = _pipeline_theme_column_allowlist(workspace=workspace)
    if not allowlist:
        return reports_df.drop(columns=theme_columns, errors="ignore")

    drop_columns = [
        column
        for column in theme_columns
        if column not in allowlist and column.casefold() not in allowlist
    ]
    if not drop_columns:
        return reports_df
    return reports_df.drop(columns=drop_columns, errors="ignore")


def _theme_rows_from_pipeline_scope(*, workspace: Workspace, scoped_reports: pd.DataFrame) -> list[dict[str, object]]:
    if scoped_reports.empty:
        return []
    label_map = _pipeline_theme_label_map(workspace=workspace)
    metadata_map = _pipeline_theme_metadata_map(workspace=workspace)
    rows: list[dict[str, object]] = []
    for column in scoped_reports.columns:
        column_name = str(column or "").strip()
        if not column_name.startswith("theme_"):
            continue
        count = int(scoped_reports[column_name].map(_network_truthy).sum())
        if count <= 0:
            continue
        default_label = column_name.replace("theme_", "").replace("_", " ").strip().title()
        label = (
            label_map.get(column_name)
            or label_map.get(column_name.casefold())
            or label_map.get(column_name.replace("theme_", ""))
            or label_map.get(column_name.replace("theme_", "").casefold())
            or default_label
        )
        metadata = (
            metadata_map.get(column_name)
            or metadata_map.get(column_name.casefold())
            or metadata_map.get(column_name.replace("theme_", ""))
            or metadata_map.get(column_name.replace("theme_", "").casefold())
            or {}
        )
        rows.append(
            {
                "label": str(label),
                "count": count,
                "description": str(metadata.get("description") or "").strip(),
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["label"]).casefold()))
    return rows


def _theme_rows_from_generic_scope(*, reports_df: pd.DataFrame, scoped_reports: pd.DataFrame) -> list[dict[str, object]]:
    if reports_df.empty or scoped_reports.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, meta in _theme_collection_map_from_reports(reports_df).items():
        column_name = str(meta.get("collection_column") or "").strip()
        if not column_name or column_name not in scoped_reports.columns:
            continue
        count = int(scoped_reports[column_name].map(_network_truthy).sum())
        if count <= 0:
            continue
        title = str(meta.get("title") or column_name.replace("_", " ").title())
        rows.append(
            {
                "label": title,
                "count": count,
                "description": str(meta.get("description") or "").strip(),
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["label"]).casefold()))
    return rows


def _workspace_ordered_columns(filtered_reports: pd.DataFrame) -> list[str]:
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
    internal_exact = {"_score", "__report_identity"}
    ordered_columns: list[str] = []
    for column in preferred_columns:
        if column in filtered_reports.columns and column not in ordered_columns:
            ordered_columns.append(column)
    for column in filtered_reports.columns:
        name = str(column)
        if name in internal_exact or name.startswith("collection_"):
            continue
        if name not in ordered_columns:
            ordered_columns.append(name)
    return ordered_columns


def _workspace_build_explore_payload(*, workspace: Workspace, request):
    params = _workspace_explore_params(request)
    base_artifact = _workspace_dataset_artifact(workspace=workspace)
    base_df_raw = _artifact_dataframe(base_artifact)
    base_df = _prune_non_pipeline_theme_columns(workspace=workspace, reports_df=base_df_raw)
    dataset_available = base_artifact is not None
    if base_df.empty:
        explore = {
            "query": str(params["query"]),
            "has_query": bool(params["query"]),
            "total_reports": 0,
            "scope_reports": 0,
            "date_range_label": "Date range unavailable",
            "scope_label": "Workspace pipeline results",
            "top_areas": [],
            "top_receivers": [],
            "top_themes": [],
            "selected_ai_filter": "custom",
            "selected_ai_filter_title": workspace.title,
            "selected_receivers": list(params["selected_receivers"]),
            "selected_areas": list(params["selected_areas"]),
            "date_start": str(params["date_start_raw"]),
            "date_end": str(params["date_end_raw"]),
            "ai_filter_options": [],
            "receiver_options": [],
            "area_options": [],
            "all_areas": [],
            "all_receivers": [],
            "all_themes": [],
            "temporal_line_path": "",
            "temporal_area_path": "",
            "temporal_axis_labels": [],
            "temporal_start_label": "",
            "temporal_end_label": "",
            "temporal_series": {"month": {"points": []}},
            "unique_receiver_count": 0,
            "themes_from_investigation": False,
            "dataset_min_date": "",
            "dataset_max_date": timezone.now().date().isoformat(),
            "date_start_effective": "",
            "date_end_effective": timezone.now().date().isoformat(),
        }
        return {"dataset_available": dataset_available, "explore": explore, "scoped_reports": base_df}

    selected_filters = {
        "coroner": [],
        "area": list(params["selected_areas"]),
        "receiver": list(params["selected_receivers"]),
    }
    scoped_reports = reports_for_collection(
        reports_df=base_df,
        collection_slug="custom",
        query=str(params["query"]),
        selected_filters=selected_filters,
    )
    scoped_reports_raw = reports_for_collection(
        reports_df=base_df_raw,
        collection_slug="custom",
        query=str(params["query"]),
        selected_filters=selected_filters,
    )

    date_start_raw = str(params["date_start_raw"])
    date_end_raw = str(params["date_end_raw"])
    if date_start_raw or date_end_raw:
        def _apply_date_mask(df: pd.DataFrame) -> pd.DataFrame:
            date_series = _parse_report_dates(df.get("date", pd.Series(dtype="object")))
            mask = pd.Series(True, index=df.index)
            if date_start_raw:
                start_ts = pd.to_datetime(date_start_raw, errors="coerce")
                if pd.notna(start_ts):
                    mask = mask & (date_series >= start_ts)
            if date_end_raw:
                end_ts = pd.to_datetime(date_end_raw, errors="coerce")
                if pd.notna(end_ts):
                    mask = mask & (date_series <= end_ts)
            return df.loc[mask].reset_index(drop=True)

        scoped_reports = _apply_date_mask(scoped_reports)
        scoped_reports_raw = _apply_date_mask(scoped_reports_raw)

    explore = build_explore_metrics(reports_df=base_df, scoped_reports_df=scoped_reports, query=str(params["query"]))
    scope_area_counts = (
        scoped_reports.get("area", pd.Series(dtype="object"))
        .map(lambda value: str(value or "").strip())
        .loc[lambda series: series != ""]
        .value_counts()
    )
    scope_receiver_counter: dict[str, int] = {}
    for raw in scoped_reports.get("receiver", pd.Series(dtype="object")).tolist():
        for receiver in _split_receivers(raw):
            if receiver:
                scope_receiver_counter[receiver] = scope_receiver_counter.get(receiver, 0) + 1

    latest_filter_run = (
        InvestigationRun.objects.filter(
            workspace=workspace,
            run_type=RunType.FILTER,
        )
        .order_by("-created_at")
        .first()
    )
    latest_filter_config = (
        latest_filter_run.input_config_json
        if latest_filter_run and isinstance(latest_filter_run.input_config_json, dict)
        else {}
    )
    pipeline_themes_selected = bool(latest_filter_config.get("run_themes", False))
    pipeline_theme_metadata_available = bool(_pipeline_theme_metadata_map(workspace=workspace))
    use_pipeline_themes = bool(pipeline_themes_selected and pipeline_theme_metadata_available)

    if use_pipeline_themes:
        scope_theme_rows = _theme_rows_from_pipeline_scope(
            workspace=workspace,
            scoped_reports=scoped_reports,
        )
    else:
        scope_theme_rows = _theme_rows_from_generic_scope(
            reports_df=base_df_raw,
            scoped_reports=scoped_reports_raw,
        )

    top_theme_rows = scope_theme_rows[:8]
    top_theme_denominator = max(1, int(top_theme_rows[0]["count"])) if top_theme_rows else 1
    top_themes = [
        {
            "label": str(item.get("label") or ""),
            "count": int(item.get("count") or 0),
            "description": str(item.get("description") or "").strip(),
            "percent_of_top": int(
                round((int(item.get("count") or 0) / float(top_theme_denominator)) * 100)
            ),
        }
        for item in top_theme_rows
    ]
    all_themes = [
        {
            "label": str(item.get("label") or ""),
            "count": int(item.get("count") or 0),
            "description": str(item.get("description") or "").strip(),
        }
        for item in scope_theme_rows
    ]
    receiver_counter: dict[str, int] = {}
    for raw in base_df.get("receiver", []):
        for receiver in _split_receivers(raw):
            receiver_counter[receiver] = receiver_counter.get(receiver, 0) + 1
    area_values = {
        str(value or "").strip()
        for value in base_df.get("area", [])
        if str(value or "").strip()
    }
    dataset_date_series = _parse_report_dates(
        base_df.get("date", pd.Series(dtype="object"))
    ).dropna()
    default_start = dataset_date_series.min().date().isoformat() if not dataset_date_series.empty else ""
    default_end = timezone.now().date().isoformat()
    explore.update(
        {
            "selected_ai_filter": "custom",
            "selected_ai_filter_title": workspace.title,
            "selected_receivers": list(params["selected_receivers"]),
            "selected_areas": list(params["selected_areas"]),
            "date_start": date_start_raw,
            "date_end": date_end_raw,
            "ai_filter_options": [],
            "receiver_options": sorted(receiver_counter.keys(), key=lambda value: value.casefold()),
            "area_options": sorted(area_values, key=lambda value: value.casefold()),
            "dataset_min_date": default_start,
            "dataset_max_date": default_end,
            "date_start_effective": date_start_raw or default_start,
            "date_end_effective": date_end_raw or default_end,
            "scope_label": "Workspace pipeline results",
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
            "all_themes": all_themes,
            "top_themes": top_themes,
            "themes_from_investigation": use_pipeline_themes,
        }
    )
    return {"dataset_available": dataset_available, "explore": explore, "scoped_reports": scoped_reports}


def _workspace_reports_panel_context(*, workspace: Workspace, request) -> dict:
    payload = _workspace_build_explore_payload(workspace=workspace, request=request)
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
        ordered_columns = _workspace_ordered_columns(scoped_reports)
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
        total_pages = max(1, int((reports_count + EXPLORE_REPORTS_PAGE_SIZE - 1) / EXPLORE_REPORTS_PAGE_SIZE))
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
        query=str(explore_data.get("query") or ""),
        ai_filter="",
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
    ).urlencode()

    return {
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
        "dataset_panel_base": reverse("workbook-reports-panel", kwargs={"workbook_id": workspace.id}),
        "dataset_browser_base": reverse("workbook-open", kwargs={"workbook_id": workspace.id}),
        "dataset_shared_query": shared_query,
        "scope_label": "workspace pipeline",
    }


@login_required
@require_POST
def activate_workspace(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    try:
        set_active_workspace_for_user(user=request.user, workspace=workspace, request=request)
    except WorkspaceLifecycleError as exc:
        messages.error(request, str(exc))
        return redirect("workbook-dashboard")
    messages.success(request, f"Active workbook set to '{workspace.title}'.")
    next_url = str(request.POST.get("next_url") or "").strip()
    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-dashboard")


@login_required
@require_GET
def open_workspace(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    try:
        set_active_workspace_for_user(user=request.user, workspace=workspace, request=request)
    except WorkspaceLifecycleError as exc:
        messages.error(request, str(exc))
        return redirect("workbook-dashboard")
    payload = _workspace_build_explore_payload(workspace=workspace, request=request)
    explore_data = payload["explore"]
    dataset_available = bool(payload["dataset_available"])

    show_reports_requested = True
    panel_query = _explore_shared_querydict(
        query=str(explore_data.get("query") or ""),
        ai_filter="",
        date_start=str(explore_data.get("date_start") or ""),
        date_end=str(explore_data.get("date_end") or ""),
        selected_receivers=list(explore_data.get("selected_receivers") or []),
        selected_areas=list(explore_data.get("selected_areas") or []),
        page=_coerce_positive_int(request.GET.get("page"), default=1),
    )
    panel_base = reverse("workbook-reports-panel", kwargs={"workbook_id": workspace.id})
    panel_url = f"{panel_base}?{panel_query.urlencode()}" if panel_query else panel_base

    runs = list(
        InvestigationRun.objects.filter(workspace=workspace)
        .select_related("investigation")
        .order_by("-created_at")
    )
    latest_run = runs[0] if runs else None
    investigation = Investigation.objects.filter(workspace=workspace).first()
    can_share_workspace_investigation = bool(
        investigation is not None and can_edit_workspace(request.user, workspace)
    )
    public_share_url = ""
    active_public_share = _active_public_share_link_for_workspace(workspace=workspace)
    if active_public_share is not None:
        public_share_url = request.build_absolute_uri(
            reverse("share-link-detail", kwargs={"share_id": active_public_share.id})
        )
    run_detail_items = []
    for run in runs[:15]:
        run_detail_items.append(
            {
                "id": str(run.id),
                "status": run.status,
                "type": run.run_type,
                "queued_at": run.queued_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "worker_id": run.worker_id,
                "error_code": run.error_code,
                "error_message": run.error_message,
                "input_config_json": json.dumps(run.input_config_json or {}, indent=2, sort_keys=True),
            }
        )

    return render(
        request,
        "accounts/explore.html",
        {
            "explore": explore_data,
            "dataset_available": dataset_available,
            "explore_reports_panel_url": panel_url,
            "show_reports_requested": show_reports_requested and dataset_available,
            "is_collection_view": True,
            "is_workspace_view": True,
            "collection": {"title": workspace.title, "description": workspace.description},
            "collection_slug": f"workspace-{workspace.id}",
            "collection_title": workspace.title,
            "collection_description": str(workspace.description or "").strip(),
            "filter_action_url": reverse("workbook-open", kwargs={"workbook_id": workspace.id}),
            "filter_reset_url": reverse("workbook-open", kwargs={"workbook_id": workspace.id}),
            "explore_export_url": reverse("workbook-export-csv", kwargs={"workbook_id": workspace.id}),
            "workspace_id": str(workspace.id),
            "workspace_investigation": investigation,
            "can_share_workspace_investigation": can_share_workspace_investigation,
            "workspace_public_share_url": public_share_url,
            "workspace_latest_run": latest_run,
            "workspace_debug_runs": run_detail_items,
        },
    )


@login_required
@require_GET
def workspace_reports_panel(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    context = _workspace_reports_panel_context(workspace=workspace, request=request)
    return render(request, "accounts/_explore_reports_panel.html", context)


@login_required
@require_GET
def workspace_export_csv(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")

    payload = _workspace_build_explore_payload(workspace=workspace, request=request)
    scoped_reports: pd.DataFrame = payload["scoped_reports"]
    if scoped_reports.empty:
        return HttpResponseBadRequest("No workspace reports available for export.")

    ordered_columns = _workspace_ordered_columns(scoped_reports)
    export_df = scoped_reports[ordered_columns].copy() if ordered_columns else scoped_reports.copy()
    buffer = StringIO()
    export_df.to_csv(buffer, index=False)
    response = HttpResponse(buffer.getvalue(), content_type="text/csv; charset=utf-8")
    slug = str(workspace.slug or workspace.title or "workspace").strip().replace(" ", "-").lower()
    response["Content-Disposition"] = f'attachment; filename="{slug}-pipeline-reports.csv"'
    return response


@login_required
@require_POST
def archive_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        archive_workspace(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceLifecycleError) as exc:
        messages.error(request, str(exc))
    return redirect("workbook-dashboard")


@login_required
@require_POST
def restore_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    next_url = str(request.POST.get("next") or "").strip()
    if next_url and not url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        next_url = ""
    try:
        restore_workspace(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceLifecycleError) as exc:
        messages.error(request, str(exc))
    if next_url:
        return redirect(next_url)
    return redirect("workbook-open", workbook_id=workspace.id)


@login_required
@require_POST
def delete_workspace_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    reason = str(request.POST.get("reason") or "").strip()
    try:
        delete_workspace_immediately(
            actor=request.user,
            workspace=workspace,
            reason=reason,
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
        return redirect("workbook-open", workbook_id=workspace.id)
    return redirect("workbook-dashboard")


@require_GET
def workspace_detail(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")
    # Retired user-facing workspace detail page: canonical user view is workbook-open.
    return redirect("workbook-open", workbook_id=workspace.id)
    if request.user.is_authenticated and workspace.archived_at is None:
        set_active_workspace_for_user(user=request.user, workspace=workspace, request=request)

    now = timezone.now()
    if is_human_view_request(request=request) and should_update_last_viewed(
        existing_last_viewed_at=workspace.last_viewed_at,
        now=now,
    ):
        workspace.last_viewed_at = now
        workspace.save(update_fields=["last_viewed_at"])

    membership = None
    if request.user.is_authenticated:
        membership = WorkspaceMembership.objects.filter(
            workspace=workspace, user=request.user
        ).first()

    manage_members_allowed = bool(
        request.user.is_authenticated and can_manage_members(request.user, workspace)
    )
    can_edit_state = bool(
        request.user.is_authenticated and can_edit_workspace(request.user, workspace)
    )
    manage_shares_allowed = bool(
        request.user.is_authenticated and can_manage_shares(request.user, workspace)
    )
    memberships = WorkspaceMembership.objects.filter(workspace=workspace).select_related("user")
    user_workbook_memberships = WorkspaceMembership.objects.none()
    if request.user.is_authenticated:
        user_workbook_memberships = (
            WorkspaceMembership.objects.select_related("workspace")
            .filter(user=request.user)
            .order_by("-workspace__updated_at")
        )
    active_workspace = get_active_workspace_for_user(user=request.user) if request.user.is_authenticated else None
    active_workspace_id = str(active_workspace.id) if active_workspace is not None else ""
    add_form = WorkspaceMemberAddForm(initial={"can_run_workflows": True})
    credential_form = WorkspaceCredentialUpsertForm()
    credential_delete_form = WorkspaceCredentialDeleteForm()
    user_credentials = WorkspaceCredential.objects.none()
    can_manage_credentials = bool(
        request.user.is_authenticated and membership and membership.can_run_workflows
    ) or bool(request.user.is_authenticated and request.user.is_superuser)
    if request.user.is_authenticated:
        user_credentials = WorkspaceCredential.objects.filter(
            workspace=workspace,
            user=request.user,
        ).order_by("provider")
    share_links = WorkspaceShareLink.objects.filter(workspace=workspace).order_by("-created_at")
    investigations = Investigation.objects.filter(workspace=workspace).order_by("-updated_at")
    latest_workspace_run = (
        InvestigationRun.objects.filter(workspace=workspace).order_by("-created_at").first()
    )
    pipeline_status = _pipeline_status_for_run(latest_workspace_run)
    pipeline_stage_label = ""
    pipeline_run_status = ""
    pipeline_last_update = None
    if latest_workspace_run is not None:
        pipeline_stage_label = RUN_TYPE_LABELS.get(
            str(latest_workspace_run.run_type), str(latest_workspace_run.run_type).title()
        )
        pipeline_run_status = str(latest_workspace_run.status)
        pipeline_last_update = latest_workspace_run.updated_at
    share_create_form = ShareLinkCreateForm(
        initial={"mode": ShareMode.SNAPSHOT, "is_public": True}
    )
    report_exclusions = WorkspaceReportExclusion.objects.filter(workspace=workspace)

    return render(
        request,
        "wb_workspaces/workspace_detail.html",
        {
            "workspace": workspace,
            "membership": membership,
            "memberships": memberships,
            "user_workbook_memberships": user_workbook_memberships,
            "active_workspace_id": active_workspace_id,
            "manage_members_allowed": manage_members_allowed,
            "can_edit_state": can_edit_state,
            "manage_shares_allowed": manage_shares_allowed,
            "add_form": add_form,
            "credential_form": credential_form,
            "credential_delete_form": credential_delete_form,
            "user_credentials": user_credentials,
            "can_manage_credentials": can_manage_credentials,
            "share_links": share_links,
            "investigations": investigations,
            "pipeline_status": pipeline_status,
            "pipeline_stage_label": pipeline_stage_label,
            "pipeline_run_status": pipeline_run_status,
            "pipeline_last_update": pipeline_last_update,
            "share_create_form": share_create_form,
            "report_exclusions": report_exclusions,
            "role_choices": MembershipRole.choices,
            "access_mode_choices": MembershipAccessMode.choices,
            "share_mode_choices": ShareMode.choices,
            "current_revision": workspace.current_revision,
        },
    )


@login_required
@require_POST
def undo_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        undo_workspace_revision(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Undid workbook state to previous revision.")
    return redirect("workbook-open", workbook_id=workspace.id)


@login_required
@require_POST
def redo_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        redo_workspace_revision(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Redid workbook state to next revision.")
    return redirect("workbook-open", workbook_id=workspace.id)


@login_required
@require_POST
def start_over_workspace_state_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        start_over_workspace_state(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook reset to baseline revision.")
    return redirect("workbook-open", workbook_id=workspace.id)


@login_required
@require_POST
def revert_workspace_reports_view(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    try:
        revert_workspace_reports(actor=request.user, workspace=workspace, request=request)
    except (PermissionDenied, ValidationError, WorkspaceRevisionError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Excluded reports have been reverted.")
    return redirect("workbook-open", workbook_id=workspace.id)


@require_GET
def public_workspace_list(request):
    workspaces = Workspace.objects.filter(
        visibility=WorkspaceVisibility.PUBLIC, is_listed=True, archived_at__isnull=True
    ).order_by("-updated_at")
    return render(
        request,
        "wb_workspaces/public_workspace_list.html",
        {"workspaces": workspaces},
    )


@login_required
@require_POST
def add_member(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    form = WorkspaceMemberAddForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member form submission.")
        return redirect("workbook-open", workbook_id=workbook_id)

    target_email = form.cleaned_data["email"].strip().lower()
    target_user = User.objects.filter(email__iexact=target_email).first()
    if target_user is None:
        messages.error(request, f"No account exists for {target_email}.")
        return redirect("workbook-open", workbook_id=workbook_id)

    try:
        add_workspace_member(
            actor=request.user,
            workspace=workspace,
            target_user=target_user,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{target_user.email} added to workbook.")
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def update_member(request, workbook_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    form = WorkspaceMemberUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid member update form submission.")
        return redirect("workbook-open", workbook_id=workbook_id)

    try:
        update_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            role=form.cleaned_data["role"],
            access_mode=form.cleaned_data["access_mode"],
            can_run_workflows=form.cleaned_data["can_run_workflows"],
            can_manage_members_flag=form.cleaned_data["can_manage_members"],
            can_manage_shares_flag=form.cleaned_data["can_manage_shares"],
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{membership.user.email} membership updated.")
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def remove_member(request, workbook_id, membership_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    membership = get_object_or_404(
        WorkspaceMembership,
        id=membership_id,
        workspace=workspace,
    )
    try:
        remove_workspace_member(
            actor=request.user,
            workspace=workspace,
            membership=membership,
            request=request,
        )
    except (WorkspaceMembershipError, ValidationError, PermissionDenied) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Workbook membership removed.")
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def save_active_llm_config(request):
    form = ActiveLLMConfigForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid LLM configuration submission.")
        return redirect("llm-config")

    provider = str(form.cleaned_data.get("provider") or "openai").strip().lower()
    model_name = str(form.cleaned_data.get("model_name") or "gpt-4.1-mini").strip()
    max_parallel_workers = int(form.cleaned_data.get("max_parallel_workers") or 1)
    api_key = str(form.cleaned_data.get("api_key") or "").strip()
    base_url = str(form.cleaned_data.get("base_url") or "").strip()
    next_url = str(form.cleaned_data.get("next_url") or "").strip()

    try:
        upsert_user_llm_setting(
            actor=request.user,
            provider=provider,
            model_name=model_name,
            max_parallel_workers=max_parallel_workers,
            request=request,
        )
        if api_key:
            upsert_user_llm_credential(
                actor=request.user,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                request=request,
            )
    except (WorkspaceCredentialValidationError, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        if api_key:
            messages.success(request, "LLM config and credential saved.")
        else:
            messages.success(request, "LLM config saved.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("llm-config")


@login_required
@require_POST
def clear_active_llm_credential(request):
    form = ActiveLLMCredentialDeleteForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid credential delete submission.")
        return redirect("llm-config")

    provider = str(form.cleaned_data.get("provider") or "openai").strip().lower()
    next_url = str(form.cleaned_data.get("next_url") or "").strip()
    deleted = delete_user_llm_credential(
        actor=request.user,
        provider=provider,
        request=request,
    )
    if deleted:
        messages.success(request, f"Cleared {provider} credential from your account defaults.")
    else:
        messages.warning(request, f"No saved {provider} credential was found.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("llm-config")


@login_required
@require_POST
def save_credential(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    membership = WorkspaceMembership.objects.filter(
        workspace=workspace,
        user=request.user,
    ).first()
    if not request.user.is_superuser and (membership is None or not membership.can_run_workflows):
        raise PermissionDenied("You do not have permission to manage credentials in this workbook.")

    form = WorkspaceCredentialUpsertForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid credential submission.")
        return redirect("workbook-open", workbook_id=workbook_id)

    try:
        credential = upsert_workspace_credential(
            actor=request.user,
            workspace=workspace,
            provider=form.cleaned_data["provider"],
            api_key=form.cleaned_data["api_key"],
            base_url=form.cleaned_data["base_url"],
            request=request,
        )
    except (WorkspaceCredentialValidationError, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(
            request,
            f"Saved {credential.provider} credential ending in {credential.key_last4}.",
        )
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def remove_credential(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        raise PermissionDenied("You do not have access to this workbook.")
    membership = WorkspaceMembership.objects.filter(
        workspace=workspace,
        user=request.user,
    ).first()
    if not request.user.is_superuser and (membership is None or not membership.can_run_workflows):
        raise PermissionDenied("You do not have permission to manage credentials in this workbook.")

    form = WorkspaceCredentialDeleteForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid credential delete submission.")
        return redirect("workbook-open", workbook_id=workbook_id)

    deleted = delete_workspace_credential(
        actor=request.user,
        workspace=workspace,
        provider=form.cleaned_data["provider"],
        request=request,
    )
    if deleted:
        messages.success(request, f"Deleted {form.cleaned_data['provider']} credential.")
    else:
        messages.warning(request, f"No {form.cleaned_data['provider']} credential found.")
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def exclude_report(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    form = WorkspaceReportExclusionCreateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid excluded-report submission.")
        return redirect("workbook-open", workbook_id=workbook_id)
    next_url = str(form.cleaned_data.get("next_url") or "").strip()

    try:
        upsert_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            report_identity=form.cleaned_data["report_identity"],
            reason=form.cleaned_data["reason"],
            report_title=form.cleaned_data["report_title"],
            report_date=form.cleaned_data["report_date"],
            report_url=form.cleaned_data["report_url"],
            request=request,
        )
    except (WorkspaceReportExclusionError, PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Report excluded from this workbook.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-open", workbook_id=workbook_id)


@login_required
@require_POST
def restore_excluded_report(request, workbook_id, exclusion_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    exclusion = get_object_or_404(
        WorkspaceReportExclusion,
        id=exclusion_id,
        workspace=workspace,
    )
    next_url = str(request.POST.get("next_url") or "").strip()
    try:
        restore_workspace_report_exclusion(
            actor=request.user,
            workspace=workspace,
            exclusion=exclusion,
            request=request,
        )
    except (WorkspaceReportExclusionError, PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Excluded report restored.")

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect("workbook-open", workbook_id=workbook_id)
