from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from wb_runs.models import InvestigationRun, RunStatus, RunType
from wb_runs.services import queue_run
from wb_notifications.services import NotificationRequestError, create_notification_request
from wb_workspaces.models import Workspace
from wb_workspaces.permissions import can_edit_workspace, can_run_workflows, can_view_workspace

from .forms import (
    InvestigationExportForm,
    InvestigationCreateForm,
    InvestigationUpdateForm,
    InvestigationWizardExtractConfigForm,
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
            if previous_stage_run.status in FAILED_STATUSES and continue_on_fail:
                note = "Continued after previous stage failed/timed out."
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


def _wizard_session_key(*, investigation_id) -> str:
    return f"investigation_wizard_state:{investigation_id}"


def _wizard_stage_sequence(state: InvestigationWizardState) -> list[str]:
    sequence = ["question", "scope", "method"]
    if state.run_themes:
        sequence.append("themes")
    if state.run_extract:
        sequence.append("extract")
    sequence.append("review")
    return sequence


def _wizard_initial_state(*, investigation: Investigation) -> InvestigationWizardState:
    scope_json = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    method_json = investigation.method_json if isinstance(investigation.method_json, dict) else {}
    return InvestigationWizardState(
        stage="question",
        title=investigation.title or "",
        question_text=investigation.question_text or "",
        scope_option=str(scope_json.get("temporal_scope_option") or "all_reports"),
        run_filter=bool(method_json.get("run_filter", True)),
        run_themes=bool(method_json.get("run_themes", False)),
        run_extract=bool(method_json.get("run_extract", False)),
        themes_config={},
        extract_config={},
        review_config={},
    )


def _wizard_form_for_stage(stage: str, *, state: InvestigationWizardState, post_data=None):
    if stage == "question":
        return InvestigationWizardQuestionForm(
            post_data,
            initial={"title": state.title, "question_text": state.question_text},
        )
    if stage == "scope":
        return InvestigationWizardScopeForm(
            post_data,
            initial={"scope_option": state.scope_option},
        )
    if stage == "method":
        return InvestigationWizardMethodForm(
            post_data,
            initial={
                "run_filter": state.run_filter,
                "run_themes": state.run_themes,
                "run_extract": state.run_extract,
            },
        )
    if stage == "themes":
        initial = {"enabled": True, **(state.themes_config or {})}
        return InvestigationWizardThemesConfigForm(post_data, initial=initial)
    if stage == "extract":
        extract_config = state.extract_config or {}
        feature_rows = (
            extract_config.get("feature_fields")
            if isinstance(extract_config.get("feature_fields"), list)
            else []
        )
        feature_lines = []
        for row in feature_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("field_name") or "").strip()
            description = str(row.get("description") or "").strip()
            field_type = str(row.get("type") or "").strip()
            if not name:
                continue
            feature_lines.append(f"{name} | {description} | {field_type}")
        initial = {
            "enabled": True,
            **extract_config,
            "feature_fields": "\n".join(feature_lines),
        }
        return InvestigationWizardExtractConfigForm(post_data, initial=initial)
    if stage == "review":
        return InvestigationWizardReviewForm(post_data, initial=state.review_config or {})
    raise ValueError(f"Unsupported wizard stage '{stage}'.")


def _apply_stage_to_state(*, stage: str, state: InvestigationWizardState, cleaned_data: dict) -> None:
    if stage == "question":
        state.title = str(cleaned_data.get("title") or "").strip()
        state.question_text = str(cleaned_data.get("question_text") or "").strip()
        return
    if stage == "scope":
        state.scope_option = str(cleaned_data.get("scope_option") or state.scope_option).strip()
        return
    if stage == "method":
        state.run_filter = bool(cleaned_data.get("run_filter", False))
        state.run_themes = bool(cleaned_data.get("run_themes"))
        state.run_extract = bool(cleaned_data.get("run_extract"))
        if not state.run_themes:
            state.themes_config = {}
        if not state.run_extract:
            state.extract_config = {}
        return
    if stage == "themes":
        state.themes_config = {
            "seed_topics": cleaned_data.get("seed_topics") or "",
            "min_themes": cleaned_data.get("min_themes"),
            "max_themes": cleaned_data.get("max_themes"),
            "extra_theme_instructions": cleaned_data.get("extra_theme_instructions") or "",
        }
        return
    if stage == "extract":
        state.extract_config = {
            "feature_fields": cleaned_data.get("feature_fields") or [],
            "allow_multiple": bool(cleaned_data.get("allow_multiple", False)),
            "force_assign": bool(cleaned_data.get("force_assign", False)),
            "skip_if_present": bool(cleaned_data.get("skip_if_present", True)),
        }
        return
    if stage == "review":
        state.review_config = {
            "execution_mode": cleaned_data.get("execution_mode") or "real",
            "provider": cleaned_data.get("provider") or "openai",
            "model_name": cleaned_data.get("model_name") or "gpt-4.1-mini",
            "request_completion_email": bool(cleaned_data.get("request_completion_email", False)),
            "notify_on": cleaned_data.get("notify_on") or "any",
        }


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

    session_key = _wizard_session_key(investigation_id=investigation.id)
    state = InvestigationWizardState.from_json(request.session.get(session_key))
    if not state.title and not state.question_text:
        state = _wizard_initial_state(investigation=investigation)

    stage_sequence = _wizard_stage_sequence(state)
    if state.stage not in stage_sequence:
        state.stage = stage_sequence[0]

    if request.method == "POST":
        action = str(request.POST.get("wizard_action") or "next").strip().lower()
        current_stage = state.stage
        stage_sequence = _wizard_stage_sequence(state)
        if current_stage not in stage_sequence:
            current_stage = stage_sequence[0]
            state.stage = current_stage
        current_index = stage_sequence.index(current_stage)

        if action == "reset":
            if session_key in request.session:
                del request.session[session_key]
            messages.info(request, "Wizard reset.")
            return redirect(
                "workbook-investigation-wizard",
                workbook_id=workbook_id,
                investigation_id=investigation_id,
            )

        if action == "back":
            previous_index = max(0, current_index - 1)
            state.stage = stage_sequence[previous_index]
            request.session[session_key] = state.to_json()
            request.session.modified = True
            return redirect(
                "workbook-investigation-wizard",
                workbook_id=workbook_id,
                investigation_id=investigation_id,
            )

        form = _wizard_form_for_stage(current_stage, state=state, post_data=request.POST)
        if form.is_valid():
            _apply_stage_to_state(stage=current_stage, state=state, cleaned_data=form.cleaned_data)
            stage_sequence = _wizard_stage_sequence(state)
            next_index = min(len(stage_sequence) - 1, stage_sequence.index(current_stage) + 1)

            if action == "launch":
                if current_stage != "review":
                    state.stage = stage_sequence[next_index]
                else:
                    try:
                        run, review = launch_investigation_wizard_pipeline(
                            actor=request.user,
                            investigation=investigation,
                            wizard_state=state,
                            request=request,
                        )
                    except (PermissionDenied, ValidationError, InvestigationServiceError) as exc:
                        messages.error(request, str(exc))
                    else:
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
                        if session_key in request.session:
                            del request.session[session_key]
                        messages.success(request, "Investigation pipeline launched.")
                        return redirect(
                            "workbook-investigation-detail",
                            workbook_id=workbook_id,
                            investigation_id=investigation_id,
                        )
            else:
                state.stage = stage_sequence[next_index]

            request.session[session_key] = state.to_json()
            request.session.modified = True
            return redirect(
                "workbook-investigation-wizard",
                workbook_id=workbook_id,
                investigation_id=investigation_id,
            )
    else:
        form = _wizard_form_for_stage(state.stage, state=state, post_data=None)

    stage_sequence = _wizard_stage_sequence(state)
    if state.stage not in stage_sequence:
        state.stage = stage_sequence[0]
    current_index = stage_sequence.index(state.stage)
    request.session[session_key] = state.to_json()
    request.session.modified = True
    return render(
        request,
        "wb_investigations/investigation_wizard.html",
        {
            "investigation": investigation,
            "workspace": investigation.workspace,
            "wizard_state": state,
            "wizard_stage": state.stage,
            "wizard_stage_sequence": stage_sequence,
            "wizard_stage_index": current_index,
            "wizard_form": form,
            "is_last_stage": current_index == len(stage_sequence) - 1,
            "can_go_back": current_index > 0,
        },
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
            return redirect("workbook-investigation-list", workbook_id=workspace.id)
        if existing_investigation is not None:
            messages.info(
                request,
                "This workbook already has an investigation. Opening it now.",
            )
            return redirect(
                "workbook-investigation-detail",
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
                    "workbook-investigation-detail",
                    workbook_id=workspace.id,
                    investigation_id=investigation.id,
                )
            return redirect("workbook-investigation-list", workbook_id=workspace.id)
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
            messages.info(request, "No investigation exists yet for this workbook.")
            return redirect("workbook-detail", workbook_id=workspace.id)
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
            return redirect("workbook-detail", workbook_id=workspace.id)

    if can_edit:
        return redirect(
            "workbook-investigation-wizard",
            workbook_id=workspace.id,
            investigation_id=investigation.id,
        )
    return redirect(
        "workbook-investigation-detail",
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
            "workbook-investigation-detail",
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
        "workbook-investigation-detail",
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
            "workbook-investigation-detail",
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
        "workbook-investigation-detail",
        workbook_id=workbook_id,
        investigation_id=investigation_id,
    )
