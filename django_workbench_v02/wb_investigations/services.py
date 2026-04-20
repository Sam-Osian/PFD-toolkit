from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_action_cache_event, log_audit_event
from wb_notifications.models import NotificationTrigger
from wb_runs.models import RunType
from wb_runs.services import queue_run
from wb_workspaces.activity import is_human_view_request, should_update_last_viewed
from wb_workspaces.models import RevisionChangeType
from wb_workspaces.permissions import can_edit_workspace
from wb_workspaces.revisions import capture_workspace_state, write_workspace_revision
from wb_workspaces.services import has_workspace_credential

from .models import Investigation


class InvestigationServiceError(ValidationError):
    pass


def _normalise_review_config(review_config: dict | None) -> dict:
    raw = review_config if isinstance(review_config, dict) else {}
    execution_mode = str(raw.get("execution_mode") or "real").strip().lower()
    if execution_mode not in {"real", "simulate"}:
        execution_mode = "real"
    provider = str(raw.get("provider") or "openai").strip().lower()
    if provider not in {"openai", "openrouter"}:
        provider = "openai"
    model_name = str(raw.get("model_name") or "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    notify_on = str(raw.get("notify_on") or NotificationTrigger.ANY).strip().lower()
    if notify_on not in {
        NotificationTrigger.SUCCESS,
        NotificationTrigger.FAILURE,
        NotificationTrigger.ANY,
    }:
        notify_on = NotificationTrigger.ANY
    return {
        "execution_mode": execution_mode,
        "provider": provider,
        "model_name": model_name,
        "request_completion_email": bool(raw.get("request_completion_email", False)),
        "notify_on": notify_on,
    }


@transaction.atomic
def launch_investigation_wizard_pipeline(
    *,
    actor,
    investigation: Investigation,
    wizard_state,
    request=None,
):
    if not can_edit_workspace(actor, investigation.workspace):
        raise PermissionDenied("You do not have permission to launch this investigation pipeline.")

    from .forms import temporal_scope_parameters

    scope_params = temporal_scope_parameters(scope_option=wizard_state.scope_option)
    pipeline_plan = wizard_state.pipeline_plan()
    if not pipeline_plan or pipeline_plan[0] != RunType.FILTER:
        raise InvestigationServiceError("Wizard pipeline must start with filter stage.")

    review = _normalise_review_config(wizard_state.review_config)
    if review["execution_mode"] == "real":
        if not has_workspace_credential(
            user=actor,
            workspace=investigation.workspace,
            provider=review["provider"],
        ):
            raise InvestigationServiceError(
                f"No saved {review['provider']} API key for this workbook. Save a key before launching."
            )

    scope_json = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    scope_json = {
        **scope_json,
        "temporal_scope_option": wizard_state.scope_option,
    }
    if scope_params.get("report_limit"):
        scope_json["report_limit"] = scope_params.get("report_limit")
    else:
        scope_json.pop("report_limit", None)

    method_json = investigation.method_json if isinstance(investigation.method_json, dict) else {}
    method_json = {
        **method_json,
        "pipeline_plan": pipeline_plan,
        "run_themes": bool(wizard_state.run_themes),
        "run_extract": bool(wizard_state.run_extract),
    }

    update_investigation(
        actor=actor,
        investigation=investigation,
        title=wizard_state.title or investigation.title,
        question_text=wizard_state.question_text or investigation.question_text,
        scope_json=scope_json,
        method_json=method_json,
        status=investigation.status,
        request=request,
    )

    run_config = {
        "execution_mode": review["execution_mode"],
        "provider": review["provider"],
        "model_name": review["model_name"],
        "search_query": wizard_state.question_text or investigation.question_text,
        "pipeline_plan": pipeline_plan,
        "pipeline_index": 0,
        "pipeline_continue_on_fail": True,
        "pipeline_require_upstream_artifact": False,
    }
    if scope_params.get("report_limit") is not None:
        run_config["report_limit"] = scope_params["report_limit"]

    if bool(wizard_state.run_themes):
        themes_config = wizard_state.themes_config if isinstance(wizard_state.themes_config, dict) else {}
        if themes_config.get("seed_topics"):
            run_config["seed_topics"] = themes_config.get("seed_topics")
        if themes_config.get("min_themes") is not None:
            run_config["min_themes"] = int(themes_config.get("min_themes"))
        if themes_config.get("max_themes") is not None:
            run_config["max_themes"] = int(themes_config.get("max_themes"))
        if themes_config.get("extra_theme_instructions"):
            run_config["extra_theme_instructions"] = themes_config.get("extra_theme_instructions")

    if bool(wizard_state.run_extract):
        extract_config = wizard_state.extract_config if isinstance(wizard_state.extract_config, dict) else {}
        run_config["feature_fields"] = extract_config.get("feature_fields") or []
        run_config["allow_multiple"] = bool(extract_config.get("allow_multiple", False))
        run_config["force_assign"] = bool(extract_config.get("force_assign", False))
        run_config["skip_if_present"] = bool(extract_config.get("skip_if_present", True))

    run = queue_run(
        actor=actor,
        investigation=investigation,
        run_type=RunType.FILTER,
        input_config_json=run_config,
        query_start_date=scope_params.get("query_start_date"),
        query_end_date=scope_params.get("query_end_date"),
        request=request,
    )
    return run, review


@transaction.atomic
def create_investigation(
    *,
    actor,
    workspace,
    title: str,
    question_text: str,
    scope_json: dict | None,
    method_json: dict | None,
    status: str,
    request=None,
) -> Investigation:
    if not can_edit_workspace(actor, workspace):
        raise PermissionDenied("You do not have permission to create investigations in this workbook.")
    if Investigation.objects.filter(workspace=workspace).exists():
        raise InvestigationServiceError("This workbook already has an investigation.")

    investigation = Investigation.objects.create(
        workspace=workspace,
        created_by=actor,
        title=title,
        question_text=question_text or "",
        scope_json=scope_json or {},
        method_json=method_json or {},
        status=status,
    )
    log_audit_event(
        action_type="investigation.created",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=workspace,
        user=actor,
        payload={"title": investigation.title, "status": investigation.status},
        request=request,
    )
    write_workspace_revision(
        workspace=workspace,
        actor=actor,
        change_type=RevisionChangeType.EDIT,
        state_json=capture_workspace_state(workspace=workspace),
        request=request,
        payload={
            "action": "investigation_created",
            "investigation_id": str(investigation.id),
        },
    )
    log_action_cache_event(
        workspace=workspace,
        user=actor,
        action_key="investigation.create",
        entity_type="investigation",
        entity_id=str(investigation.id),
        options={
            "status": investigation.status,
        },
        state_before={},
        state_after={
            "title": investigation.title,
            "question_text": investigation.question_text,
            "scope_json": investigation.scope_json,
            "method_json": investigation.method_json,
            "status": investigation.status,
        },
        context={"source": "service"},
    )
    return investigation


@transaction.atomic
def update_investigation(
    *,
    actor,
    investigation: Investigation,
    title: str,
    question_text: str,
    scope_json: dict | None,
    method_json: dict | None,
    status: str,
    request=None,
) -> Investigation:
    if not can_edit_workspace(actor, investigation.workspace):
        raise PermissionDenied("You do not have permission to update this investigation.")

    before = {
        "title": investigation.title,
        "question_text": investigation.question_text,
        "scope_json": investigation.scope_json,
        "method_json": investigation.method_json,
        "status": investigation.status,
    }

    investigation.title = title
    investigation.question_text = question_text or ""
    investigation.scope_json = scope_json or {}
    investigation.method_json = method_json or {}
    investigation.status = status
    investigation.save(
        update_fields=[
            "title",
            "question_text",
            "scope_json",
            "method_json",
            "status",
            "updated_at",
        ]
    )

    log_audit_event(
        action_type="investigation.updated",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=investigation.workspace,
        user=actor,
        payload={
            "before": before,
            "after": {
                "title": investigation.title,
                "question_text": investigation.question_text,
                "scope_json": investigation.scope_json,
                "method_json": investigation.method_json,
                "status": investigation.status,
            },
        },
        request=request,
    )
    write_workspace_revision(
        workspace=investigation.workspace,
        actor=actor,
        change_type=RevisionChangeType.EDIT,
        state_json=capture_workspace_state(workspace=investigation.workspace),
        request=request,
        payload={
            "action": "investigation_updated",
            "investigation_id": str(investigation.id),
        },
    )
    log_action_cache_event(
        workspace=investigation.workspace,
        user=actor,
        action_key="investigation.update",
        entity_type="investigation",
        entity_id=str(investigation.id),
        options={"status": investigation.status},
        state_before=before,
        state_after={
            "title": investigation.title,
            "question_text": investigation.question_text,
            "scope_json": investigation.scope_json,
            "method_json": investigation.method_json,
            "status": investigation.status,
        },
        context={"source": "service"},
    )
    return investigation


def record_investigation_view(*, investigation: Investigation, user=None, request=None) -> None:
    now = timezone.now()
    is_human_view = is_human_view_request(request=request)
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=investigation.last_viewed_at,
        now=now,
    ):
        investigation.last_viewed_at = now
        investigation.save(update_fields=["last_viewed_at"])

    workspace = investigation.workspace
    if is_human_view and should_update_last_viewed(
        existing_last_viewed_at=workspace.last_viewed_at,
        now=now,
    ):
        workspace.last_viewed_at = now
        workspace.save(update_fields=["last_viewed_at"])

    log_audit_event(
        action_type="investigation.viewed",
        target_type="investigation",
        target_id=str(investigation.id),
        workspace=investigation.workspace,
        user=user if user and user.is_authenticated else None,
        payload={
            "status": investigation.status,
            "is_human_view": is_human_view,
        },
        request=request,
    )
