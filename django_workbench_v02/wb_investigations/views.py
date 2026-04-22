import json
from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from wb_runs.models import InvestigationRun, RunStatus, RunType
from wb_runs.services import queue_run
from wb_notifications.services import NotificationRequestError, create_notification_request
from wb_workspaces.models import Workspace
from wb_workspaces.permissions import can_edit_workspace, can_run_workflows, can_view_workspace
from wb_workspaces.services import create_workspace_for_user

from .forms import (
    InvestigationExportForm,
    InvestigationCreateForm,
    InvestigationUpdateForm,
    InvestigationWizardExtractConfigForm,
    InvestigationWizardFilterConfigForm,
    InvestigationWizardMethodForm,
    InvestigationWizardQuestionForm,
    InvestigationWizardReviewForm,
    InvestigationWizardScopeForm,
    InvestigationWizardState,
    InvestigationWizardThemesConfigForm,
)
from .models import Investigation
from .services import (
    InvestigationServiceError,
    create_investigation,
    launch_investigation_wizard_pipeline,
    record_investigation_view,
    update_investigation,
)


PIPELINE_STAGE_LABELS = {
    RunType.FILTER: "Screen and filter",
    RunType.THEMES: "Discover themes",
    RunType.EXTRACT: "Extract structured data",
    RunType.EXPORT: "Export bundle",
}

TERMINAL_STATUSES = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
    RunStatus.CANCELLED,
}

FAILED_STATUSES = {RunStatus.FAILED, RunStatus.TIMED_OUT}


def _next_workspace_slug_for_user(*, user, title: str) -> str:
    base = slugify(str(title or "").strip())[:80]
    if not base:
        base = "workspace"
    candidate = base
    counter = 2
    while Workspace.objects.filter(created_by=user, slug=candidate).exists():
        suffix = f"-{counter}"
        candidate = f"{base[: max(1, 100 - len(suffix))]}{suffix}"
        counter += 1
    return candidate


def _investigation_detail_with_open_wizard_url(
    *,
    workbook_id,
    investigation_id,
    wizard_step: str = "",
    retry_run_id: str = "",
) -> str:
    base_url = reverse(
        "investigation-detail",
        kwargs={"workbook_id": workbook_id, "investigation_id": investigation_id},
    )
    params = {"open_wizard": "1"}
    if str(wizard_step or "").strip():
        params["wizard_step"] = str(wizard_step).strip()
    if str(retry_run_id or "").strip():
        params["retry_run_id"] = str(retry_run_id).strip()
    return f"{base_url}?{urlencode(params)}"


def _workspace_dashboard_with_open_wizard_url() -> str:
    base_url = reverse("workspace-dashboard")
    return f"{base_url}?{urlencode({'open_wizard': '1'})}"


def _normalise_pipeline_plan(raw_plan) -> list[str]:
    if not isinstance(raw_plan, list):
        return []
    allowed = {RunType.FILTER, RunType.THEMES, RunType.EXTRACT, RunType.EXPORT}
    result: list[str] = []
    for raw in raw_plan:
        value = str(raw or "").strip().lower()
        if value in allowed:
            result.append(value)
    return result


def _pipeline_index_for_run(run: InvestigationRun, *, plan: list[str]) -> int:
    config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
    raw_index = config.get("pipeline_index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(plan):
        return index
    try:
        return plan.index(str(run.run_type or "").strip().lower())
    except ValueError:
        return -1


def _previous_pipeline_run_id(run: InvestigationRun) -> str:
    for event in run.events.all():
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        previous_id = str(payload.get("pipeline_previous_run_id") or "").strip()
        if previous_id:
            return previous_id
    return ""


def _continued_after_failed_upstream(run: InvestigationRun) -> bool:
    for event in run.events.all():
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        if bool(payload.get("continued_after_failed_upstream")):
            return True
    return False


def _build_pipeline_timeline(*, investigation: Investigation) -> dict | None:
    runs = list(
        InvestigationRun.objects.filter(investigation=investigation)
        .prefetch_related("events")
        .order_by("created_at")
    )
    if not runs:
        return None

    root_candidates: list[InvestigationRun] = []
    for run in runs:
        config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
        plan = _normalise_pipeline_plan(config.get("pipeline_plan"))
        if not plan:
            continue
        if _pipeline_index_for_run(run, plan=plan) != 0:
            continue
        if str(run.run_type).strip().lower() != plan[0]:
            continue
        root_candidates.append(run)
    if not root_candidates:
        return None

    root = root_candidates[-1]
    root_config = root.input_config_json if isinstance(root.input_config_json, dict) else {}
    plan = _normalise_pipeline_plan(root_config.get("pipeline_plan"))
    if not plan:
        return None

    continue_on_fail = bool(root_config.get("pipeline_continue_on_fail", False))
    forward_links: dict[str, list[InvestigationRun]] = {}
    for run in runs:
        previous_id = _previous_pipeline_run_id(run)
        if not previous_id:
            continue
        forward_links.setdefault(previous_id, []).append(run)

    for linked_runs in forward_links.values():
        linked_runs.sort(key=lambda item: item.created_at)

    timeline_entries: list[dict] = []
    previous_stage_run: InvestigationRun | None = None
    for index, stage_type in enumerate(plan):
        stage_run: InvestigationRun | None = None
        if index == 0:
            stage_run = root
        elif previous_stage_run is not None:
            candidates = [
                run
                for run in forward_links.get(str(previous_stage_run.id), [])
                if _pipeline_index_for_run(run, plan=plan) == index
                and str(run.run_type).strip().lower() == stage_type
            ]
            if candidates:
                stage_run = candidates[-1]

        status = stage_run.status if stage_run is not None else "pending"
        note = ""
        if stage_run is not None and previous_stage_run is not None:
            if _continued_after_failed_upstream(stage_run) or (
                previous_stage_run.status in FAILED_STATUSES and continue_on_fail
            ):
                note = "Continued after upstream failure."
        elif stage_run is None and previous_stage_run is not None:
            if previous_stage_run.status in FAILED_STATUSES and not continue_on_fail:
                note = "Pipeline halted after previous stage failure."
            elif previous_stage_run.status == RunStatus.CANCELLED:
                note = "Pipeline halted after cancellation."
            elif previous_stage_run.status not in TERMINAL_STATUSES:
                note = "Waiting for previous stage to finish."
            elif previous_stage_run.status in FAILED_STATUSES and continue_on_fail:
                note = "Continue-on-failure is enabled; next stage has not been queued yet."

        timeline_entries.append(
            {
                "index": index,
                "stage_type": stage_type,
                "stage_label": PIPELINE_STAGE_LABELS.get(stage_type, stage_type.title()),
                "run": stage_run,
                "status": status,
                "note": note,
            }
        )
        previous_stage_run = stage_run

    active_stage_index = -1
    for idx, entry in enumerate(timeline_entries):
        if entry["status"] not in TERMINAL_STATUSES and entry["status"] != "pending":
            active_stage_index = idx
            break
    if active_stage_index < 0:
        for idx, entry in enumerate(timeline_entries):
            if entry["status"] == "pending":
                active_stage_index = idx
                break
    if active_stage_index < 0 and timeline_entries:
        active_stage_index = len(timeline_entries) - 1

    return {
        "root_run": root,
        "plan": plan,
        "continue_on_fail": continue_on_fail,
        "entries": timeline_entries,
        "active_stage_index": active_stage_index,
    }


def _wizard_initial_state(*, investigation: Investigation) -> InvestigationWizardState:
    scope_json = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    method_json = investigation.method_json if isinstance(investigation.method_json, dict) else {}
    return InvestigationWizardState(
        stage="question",
        title=investigation.title or "",
        question_text=investigation.question_text or "",
        scope_option=str(scope_json.get("temporal_scope_option") or "all_reports"),
        scope_start_date=str(scope_json.get("query_start_date") or ""),
        scope_end_date=str(scope_json.get("query_end_date") or ""),
        run_filter=bool(method_json.get("run_filter", True)),
        run_themes=bool(method_json.get("run_themes", False)),
        run_extract=bool(method_json.get("run_extract", False)),
        filter_config={},
        themes_config={},
        extract_config={},
        review_config={},
    )


def _scope_option_from_dates(*, start: str, end: str, report_limit) -> str:
    if report_limit == 100:
        return "most_recent_100"
    start_value = str(start or "").strip()
    end_value = str(end or "").strip()
    if start_value and end_value:
        return "custom_range"
    return "all_reports"


def _wizard_retry_prefill(*, investigation: Investigation, run: InvestigationRun) -> dict:
    config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
    scope_json = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    method_json = investigation.method_json if isinstance(investigation.method_json, dict) else {}

    pipeline_plan = config.get("pipeline_plan") if isinstance(config.get("pipeline_plan"), list) else []
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
        report_limit=config.get("report_limit"),
    )

    selected_filters = config.get("selected_filters") if isinstance(config.get("selected_filters"), dict) else {}
    feature_fields = config.get("feature_fields") if isinstance(config.get("feature_fields"), list) else []
    sanitised_features = []
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

    return {
        "title": str(investigation.title or "").strip(),
        "question_text": str(investigation.question_text or "").strip(),
        "scope_option": scope_option,
        "custom_start_date": scope_start,
        "custom_end_date": scope_end,
        "run_filter": bool(run_filter),
        "run_themes": bool(run_themes),
        "run_extract": bool(run_extract),
        "search_query": str(config.get("search_query") or investigation.question_text or "").strip(),
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


@login_required
@require_http_methods(["GET", "POST"])
def investigation_start(request):
    if request.method == "POST":
        form = InvestigationWizardQuestionForm(request.POST)
        if form.is_valid():
            workspace_title = str(form.cleaned_data["title"] or "").strip()
            question_text = str(form.cleaned_data["question_text"] or "").strip()
            is_modal_submit = str(request.POST.get("wizard_modal_submit") or "").strip() == "1"

            modal_scope_option = "all_reports"
            modal_scope_start_date = ""
            modal_scope_end_date = ""
            modal_run_filter = True
            modal_run_themes = False
            modal_run_extract = False
            modal_themes_config: dict = {}
            modal_extract_config: dict = {}
            modal_review_config: dict = {}
            modal_credential_input: dict = {}
            modal_filter_config: dict = {}
            if is_modal_submit:
                scope_form = InvestigationWizardScopeForm(request.POST)
                method_form = InvestigationWizardMethodForm(request.POST)
                if not scope_form.is_valid() or not method_form.is_valid():
                    messages.error(request, "Invalid wizard configuration. Please review scope and method.")
                    return redirect(request.META.get("HTTP_REFERER") or "landing")
                modal_scope_option = str(scope_form.cleaned_data.get("scope_option") or "all_reports").strip()
                modal_scope_start_date = str(scope_form.cleaned_data.get("custom_start_date") or "")
                modal_scope_end_date = str(scope_form.cleaned_data.get("custom_end_date") or "")
                modal_run_filter = bool(method_form.cleaned_data.get("run_filter", False))
                modal_run_themes = bool(method_form.cleaned_data.get("run_themes", False))
                modal_run_extract = bool(method_form.cleaned_data.get("run_extract", False))

                if modal_run_filter:
                    filter_payload = request.POST.copy()
                    filter_payload["enabled"] = "on"
                    filter_form = InvestigationWizardFilterConfigForm(filter_payload)
                    if not filter_form.is_valid():
                        messages.error(request, "Filter configuration is invalid.")
                        return redirect(request.META.get("HTTP_REFERER") or "landing")
                    modal_filter_config = {
                        "search_query": str(filter_form.cleaned_data.get("search_query") or "").strip(),
                        "filter_df": bool(filter_form.cleaned_data.get("filter_df", True)),
                        "include_supporting_quotes": bool(
                            filter_form.cleaned_data.get("include_supporting_quotes", False)
                        ),
                        "selected_filters": filter_form.cleaned_data.get("selected_filters") or {},
                    }

                if modal_run_themes:
                    themes_payload = request.POST.copy()
                    themes_payload["enabled"] = "on"
                    themes_form = InvestigationWizardThemesConfigForm(themes_payload)
                    if not themes_form.is_valid():
                        messages.error(request, "Themes configuration is invalid.")
                        return redirect(request.META.get("HTTP_REFERER") or "landing")
                    modal_themes_config = {
                        "seed_topics": str(themes_form.cleaned_data.get("seed_topics") or "").strip(),
                        "min_themes": themes_form.cleaned_data.get("min_themes"),
                        "max_themes": themes_form.cleaned_data.get("max_themes"),
                        "extra_theme_instructions": str(
                            themes_form.cleaned_data.get("extra_theme_instructions") or ""
                        ).strip(),
                    }

                if modal_run_extract:
                    extract_payload = request.POST.copy()
                    extract_payload["enabled"] = "on"
                    extract_form = InvestigationWizardExtractConfigForm(extract_payload)
                    if not extract_form.is_valid():
                        for err in extract_form.non_field_errors():
                            messages.error(request, err)
                        for _, errors in extract_form.errors.items():
                            for err in errors:
                                messages.error(request, err)
                        return redirect(request.META.get("HTTP_REFERER") or "landing")
                    modal_extract_config = {
                        "feature_fields": extract_form.cleaned_data.get("feature_fields") or [],
                        "allow_multiple": bool(extract_form.cleaned_data.get("allow_multiple", False)),
                        "force_assign": bool(extract_form.cleaned_data.get("force_assign", False)),
                        "skip_if_present": bool(extract_form.cleaned_data.get("skip_if_present", True)),
                        "include_supporting_quotes": bool(
                            extract_form.cleaned_data.get("extract_include_supporting_quotes", False)
                        ),
                    }

                review_form = InvestigationWizardReviewForm(request.POST)
                if review_form.is_valid():
                    modal_review_config = {
                        "execution_mode": "real",
                        "provider": review_form.cleaned_data.get("provider") or "openai",
                        "model_name": review_form.cleaned_data.get("model_name") or "gpt-4.1-mini",
                        "max_parallel_workers": int(
                            review_form.cleaned_data.get("max_parallel_workers") or 1
                        ),
                        "request_completion_email": bool(
                            review_form.cleaned_data.get("request_completion_email", False)
                        ),
                        "notify_on": review_form.cleaned_data.get("notify_on") or "any",
                        "base_url": str(review_form.cleaned_data.get("base_url") or "").strip(),
                    }
                    modal_credential_input = {
                        "api_key": str(review_form.cleaned_data.get("api_key") or "").strip(),
                        "base_url": str(review_form.cleaned_data.get("base_url") or "").strip(),
                        "save_api_key": True,
                    }
                else:
                    messages.error(request, "Review configuration is invalid.")
                    return redirect(request.META.get("HTTP_REFERER") or "landing")
            try:
                workspace = create_workspace_for_user(
                    user=request.user,
                    title=workspace_title,
                    slug=_next_workspace_slug_for_user(user=request.user, title=workspace_title),
                    description=question_text,
                    request=request,
                )
                investigation = create_investigation(
                    actor=request.user,
                    workspace=workspace,
                    title=workspace_title,
                    question_text=question_text,
                    scope_json={},
                    method_json={},
                    status="draft",
                    request=request,
                )
            except (PermissionDenied, ValidationError, InvestigationServiceError) as exc:
                messages.error(request, str(exc))
            else:
                state = _wizard_initial_state(investigation=investigation)
                state.title = workspace_title
                state.question_text = question_text
                if is_modal_submit:
                    state.scope_option = modal_scope_option
                    state.scope_start_date = modal_scope_start_date
                    state.scope_end_date = modal_scope_end_date
                    state.run_filter = modal_run_filter
                    state.run_themes = modal_run_themes
                    state.run_extract = modal_run_extract
                    state.filter_config = modal_filter_config
                    state.themes_config = modal_themes_config
                    state.extract_config = modal_extract_config
                    state.review_config = modal_review_config
                    state.stage = "review"
                else:
                    state.stage = "scope"
                if is_modal_submit:
                    try:
                        run, review = launch_investigation_wizard_pipeline(
                            actor=request.user,
                            investigation=investigation,
                            wizard_state=state,
                            credential_input=modal_credential_input,
                            request=request,
                        )
                    except (PermissionDenied, ValidationError, InvestigationServiceError) as exc:
                        messages.error(request, str(exc))
                        return redirect(
                            _investigation_detail_with_open_wizard_url(
                                workbook_id=workspace.id,
                                investigation_id=investigation.id,
                            )
                        )

                    if review.get("request_completion_email"):
                        try:
                            create_notification_request(
                                run=run,
                                user=request.user,
                                notify_on=review.get("notify_on") or "any",
                                request=request,
                            )
                        except (NotificationRequestError, ValidationError) as exc:
                            messages.warning(
                                request,
                                f"Pipeline launched, but completion notification was not created: {exc}",
                            )
                    messages.success(request, "Investigation launched. Workspace is now pending.")
                    return redirect("workspace-dashboard")

                messages.success(request, "Workspace created. Open the investigation wizard to continue.")
                return redirect(
                    _investigation_detail_with_open_wizard_url(
                        workbook_id=workspace.id,
                        investigation_id=investigation.id,
                    )
                )
    return redirect(_workspace_dashboard_with_open_wizard_url())


@login_required
@require_http_methods(["GET", "POST"])
def investigation_wizard(request, workbook_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workbook_id,
    )
    if not can_edit_workspace(request.user, investigation.workspace):
        raise PermissionDenied("You do not have permission to run this investigation wizard.")
    return redirect(
        _investigation_detail_with_open_wizard_url(
            workbook_id=workbook_id,
            investigation_id=investigation_id,
        )
    )


@require_http_methods(["GET", "POST"])
def investigation_list(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")

    can_edit = bool(request.user.is_authenticated and can_edit_workspace(request.user, workspace))
    existing_investigation = Investigation.objects.filter(workspace=workspace).first()
    if request.method == "POST":
        if not request.user.is_authenticated:
            return redirect("accounts-login")
        form = InvestigationCreateForm(request.POST)
        if not can_edit:
            messages.error(request, "You do not have permission to create investigations.")
            return redirect("investigation-list", workbook_id=workspace.id)
        if existing_investigation is not None:
            messages.info(
                request,
                "This workspace already has an investigation. Opening it now.",
            )
            return redirect(
                "investigation-detail",
                workbook_id=workspace.id,
                investigation_id=existing_investigation.id,
            )
        if form.is_valid():
            try:
                investigation = create_investigation(
                    actor=request.user,
                    workspace=workspace,
                    title=form.cleaned_data["title"],
                    question_text=form.cleaned_data["question_text"],
                    scope_json=form.cleaned_data["scope_json"],
                    method_json=form.cleaned_data["method_json"],
                    status=form.cleaned_data["status"],
                    request=request,
                )
            except (PermissionDenied, ValidationError) as exc:
                messages.error(request, str(exc))
            else:
                messages.success(request, "Investigation created.")
                return redirect(
                    "investigation-detail",
                    workbook_id=workspace.id,
                    investigation_id=investigation.id,
                )
            return redirect("investigation-list", workbook_id=workspace.id)
    else:
        form = InvestigationCreateForm()

    investigations = Investigation.objects.filter(workspace=workspace).order_by("-updated_at")
    return render(
        request,
        "wb_investigations/investigation_list.html",
        {
            "workspace": workspace,
            "investigations": investigations,
            "has_investigation": investigations.exists(),
            "can_edit": can_edit,
            "create_form": form,
        },
    )


@require_GET
def investigation_entry(request, workbook_id):
    workspace = get_object_or_404(Workspace, id=workbook_id)
    if not can_view_workspace(request.user, workspace):
        return redirect("accounts-login")

    can_edit = bool(
        request.user.is_authenticated and can_edit_workspace(request.user, workspace)
    )
    investigation = Investigation.objects.filter(workspace=workspace).first()

    if investigation is None:
        if not can_edit:
            messages.info(request, "No investigation exists yet for this workspace.")
            return redirect("workspace-detail", workbook_id=workspace.id)
        try:
            investigation = create_investigation(
                actor=request.user,
                workspace=workspace,
                title=f"{workspace.title} investigation",
                question_text="",
                scope_json={},
                method_json={},
                status="draft",
                request=request,
            )
        except (PermissionDenied, ValidationError, InvestigationServiceError) as exc:
            messages.error(request, str(exc))
            return redirect("workspace-detail", workbook_id=workspace.id)

    if can_edit:
        return redirect(
            _investigation_detail_with_open_wizard_url(
                workbook_id=workspace.id,
                investigation_id=investigation.id,
            )
        )
    return redirect(
        "investigation-detail",
        workbook_id=workspace.id,
        investigation_id=investigation.id,
    )


@require_GET
def investigation_detail(request, workbook_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace", "created_by"),
        id=investigation_id,
        workspace_id=workbook_id,
    )
    if not can_view_workspace(request.user, investigation.workspace):
        return redirect("accounts-login")

    record_investigation_view(investigation=investigation, user=request.user, request=request)

    can_edit = bool(
        request.user.is_authenticated
        and can_edit_workspace(request.user, investigation.workspace)
    )
    can_run = bool(
        request.user.is_authenticated
        and can_run_workflows(request.user, investigation.workspace)
    )
    runs = InvestigationRun.objects.filter(investigation=investigation).order_by("-created_at")
    retry_run_id = str(request.GET.get("retry_run_id") or "").strip()
    retry_wizard_prefill_json = ""
    if retry_run_id and can_edit:
        retry_run = runs.filter(id=retry_run_id).first()
        if retry_run is not None:
            retry_wizard_prefill_json = json.dumps(
                _wizard_retry_prefill(investigation=investigation, run=retry_run)
            )
    pipeline_timeline = _build_pipeline_timeline(investigation=investigation)
    return render(
        request,
        "wb_investigations/investigation_detail.html",
        {
            "investigation": investigation,
            "workspace": investigation.workspace,
            "can_edit": can_edit,
            "can_run": can_run,
            "runs": runs,
            "pipeline_timeline": pipeline_timeline,
            "retry_wizard_prefill_json": retry_wizard_prefill_json,
            "export_form": InvestigationExportForm(),
        },
    )


@login_required
@require_POST
def queue_export_bundle(request, workbook_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workbook_id,
    )
    form = InvestigationExportForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid export configuration.")
        return redirect(
            "investigation-detail",
            workbook_id=workbook_id,
            investigation_id=investigation_id,
        )
    try:
        config = {
            "bundle_name": form.cleaned_data.get("bundle_name") or "",
            "download_include_dataset": bool(form.cleaned_data.get("download_include_dataset", True)),
            "download_include_excluded": bool(form.cleaned_data.get("download_include_excluded", True)),
            "download_include_theme": bool(form.cleaned_data.get("download_include_theme", True)),
            "download_include_feature_grid": bool(
                form.cleaned_data.get("download_include_feature_grid", True)
            ),
            "download_include_script": bool(form.cleaned_data.get("download_include_script", False)),
            "latest_per_artifact_type": bool(form.cleaned_data.get("latest_per_artifact_type", True)),
        }
        if form.cleaned_data.get("max_artifacts") is not None:
            config["max_artifacts"] = int(form.cleaned_data.get("max_artifacts"))
        run = queue_run(
            actor=request.user,
            investigation=investigation,
            run_type=RunType.EXPORT,
            input_config_json=config,
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        request_completion_email = bool(form.cleaned_data.get("request_completion_email"))
        if request_completion_email:
            notify_on = form.cleaned_data.get("notify_on") or "any"
            try:
                create_notification_request(
                    run=run,
                    user=request.user,
                    notify_on=notify_on,
                    request=request,
                )
            except (NotificationRequestError, ValidationError) as exc:
                messages.warning(
                    request,
                    f"Export queued, but completion notification could not be created: {exc}",
                )
            else:
                messages.success(request, "Export queued. Completion email notification requested.")
        else:
            messages.success(request, "Export queued.")
    return redirect(
        "investigation-detail",
        workbook_id=workbook_id,
        investigation_id=investigation_id,
    )


@login_required
@require_POST
def investigation_update(request, workbook_id, investigation_id):
    investigation = get_object_or_404(
        Investigation.objects.select_related("workspace"),
        id=investigation_id,
        workspace_id=workbook_id,
    )
    form = InvestigationUpdateForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid investigation update form.")
        return redirect(
            "investigation-detail",
            workbook_id=workbook_id,
            investigation_id=investigation_id,
        )

    try:
        update_investigation(
            actor=request.user,
            investigation=investigation,
            title=form.cleaned_data["title"],
            question_text=form.cleaned_data["question_text"],
            scope_json=form.cleaned_data["scope_json"],
            method_json=form.cleaned_data["method_json"],
            status=form.cleaned_data["status"],
            request=request,
        )
    except (PermissionDenied, ValidationError) as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Investigation updated.")

    return redirect(
        "investigation-detail",
        workbook_id=workbook_id,
        investigation_id=investigation_id,
    )
