from __future__ import annotations

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.utils import timezone

from wb_auditlog.services import log_action_cache_event, log_audit_event
from wb_notifications.models import NotificationTrigger
from wb_runs.models import RunType
from wb_runs.services import queue_run
from wb_workspaces.activity import is_human_view_request, should_update_last_viewed
from wb_workspaces.models import RevisionChangeType, WorkspaceLLMProvider
from wb_workspaces.permissions import can_edit_workspace, can_run_workflows
from wb_workspaces.revisions import capture_workspace_state, write_workspace_revision
from wb_workspaces.services import has_workspace_credential, upsert_workspace_credential

from .models import Investigation


class InvestigationServiceError(ValidationError):
    pass


ALLOWED_REVIEW_PROVIDERS = {
    WorkspaceLLMProvider.OPENAI,
    WorkspaceLLMProvider.OPENROUTER,
}

PROVIDER_MODEL_ALLOWLIST = {
    WorkspaceLLMProvider.OPENAI: {
        "gpt-4.1-mini",
        "gpt-4.1",
    },
    WorkspaceLLMProvider.OPENROUTER: {
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1",
    },
}


def _normalise_review_config(review_config: dict | None) -> dict:
    raw = review_config if isinstance(review_config, dict) else {}
    execution_mode = "real"
    raw_provider = raw.get("provider")
    if raw_provider is None:
        provider = WorkspaceLLMProvider.OPENAI
    else:
        provider = str(raw_provider).strip().lower()

    raw_model_name = raw.get("model_name")
    if raw_model_name is None:
        model_name = "gpt-4.1-mini"
    else:
        model_name = str(raw_model_name).strip()
    try:
        max_parallel_workers = int(raw.get("max_parallel_workers") or 1)
    except (TypeError, ValueError):
        max_parallel_workers = 1
    max_parallel_workers = min(32, max(1, max_parallel_workers))
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
        "max_parallel_workers": max_parallel_workers,
        "request_completion_email": bool(raw.get("request_completion_email", True)),
        "notify_on": notify_on,
    }


def _normalise_credential_input(credential_input: dict | None) -> dict:
    raw = credential_input if isinstance(credential_input, dict) else {}
    return {
        "api_key": str(raw.get("api_key") or "").strip(),
        "base_url": str(raw.get("base_url") or "").strip(),
        "save_api_key": bool(raw.get("save_api_key", True)),
    }


def evaluate_investigation_launch_readiness(
    *,
    actor,
    investigation: Investigation,
    wizard_state,
    credential_input: dict | None = None,
) -> dict:
    review = _normalise_review_config(wizard_state.review_config)
    credential = _normalise_credential_input(credential_input)
    pipeline_plan = wizard_state.pipeline_plan()

    permission_ready = bool(
        can_edit_workspace(actor, investigation.workspace)
        and can_run_workflows(actor, investigation.workspace)
    )
    pipeline_ready = bool(pipeline_plan)
    provider_model_ready = True
    provider_model_message = "Provider/model combination is valid."
    if review["provider"] not in ALLOWED_REVIEW_PROVIDERS:
        provider_model_ready = False
        provider_model_message = (
            f"Provider '{review['provider'] or 'unknown'}' is not supported for wizard runs."
        )
    elif not review["model_name"]:
        provider_model_ready = False
        provider_model_message = "Model name is required before launch."
    else:
        allowed_models = PROVIDER_MODEL_ALLOWLIST.get(review["provider"], set())
        if allowed_models and review["model_name"] not in allowed_models:
            allowed_models_text = ", ".join(sorted(allowed_models))
            provider_model_ready = False
            provider_model_message = (
                f"Model '{review['model_name']}' is not supported for {review['provider']}. "
                f"Choose one of: {allowed_models_text}."
            )

    credential_ready = True
    credential_block_reason = ""
    if review["execution_mode"] == "real":
        if not provider_model_ready:
            credential_ready = False
            credential_block_reason = "Resolve provider/model readiness before checking credentials."
        else:
            saved = has_workspace_credential(
                user=actor,
                workspace=investigation.workspace,
                provider=review["provider"],
            )
            provided = bool(credential.get("api_key"))
            save_requested = bool(credential.get("save_api_key", True))
            if provided and not save_requested and not saved:
                credential_ready = False
                credential_block_reason = (
                    "Server-side runs require a saved API key. Enable key saving or use an existing saved key."
                )
            elif not saved and not provided:
                credential_ready = False
                credential_block_reason = (
                    f"No saved {review['provider']} API key for this workbook. Add a key before launch."
                )

    checks = [
        {
            "key": "permissions",
            "label": "Workflow permission",
            "ready": permission_ready,
            "message": (
                "You can launch runs in this workspace."
                if permission_ready
                else "You do not have permission to launch workflows in this workspace."
            ),
        },
        {
            "key": "pipeline",
            "label": "Pipeline",
            "ready": pipeline_ready,
            "message": (
                f"{len(pipeline_plan)} stage(s) selected."
                if pipeline_ready
                else "Select at least one pipeline stage before launching."
            ),
        },
        {
            "key": "provider_model",
            "label": "Provider/model",
            "ready": provider_model_ready,
            "message": provider_model_message,
        },
        {
            "key": "credential",
            "label": "Credential",
            "ready": credential_ready,
            "message": (
                f"{review['provider'].title()} key ready for this workspace."
                if credential_ready
                else credential_block_reason
            ),
        },
    ]

    blocking_errors = [check["message"] for check in checks if not check["ready"]]
    return {
        "can_launch": not blocking_errors,
        "checks": checks,
        "blocking_errors": blocking_errors,
        "review": review,
        "credential": credential,
        "pipeline_plan": pipeline_plan,
    }


@transaction.atomic
def launch_investigation_wizard_pipeline(
    *,
    actor,
    investigation: Investigation,
    wizard_state,
    credential_input: dict | None = None,
    request=None,
):
    readiness = evaluate_investigation_launch_readiness(
        actor=actor,
        investigation=investigation,
        wizard_state=wizard_state,
        credential_input=credential_input,
    )
    if not readiness["can_launch"]:
        raise InvestigationServiceError(" ".join(readiness["blocking_errors"]))

    review = readiness["review"]
    credential = readiness["credential"]
    pipeline_plan = readiness["pipeline_plan"]
    if not (
        can_edit_workspace(actor, investigation.workspace)
        and can_run_workflows(actor, investigation.workspace)
    ):
        raise PermissionDenied("You do not have permission to launch this investigation pipeline.")

    from .forms import temporal_scope_parameters

    scope_start_date = (
        timezone.datetime.fromisoformat(wizard_state.scope_start_date).date()
        if getattr(wizard_state, "scope_start_date", "")
        else None
    )
    scope_end_date = (
        timezone.datetime.fromisoformat(wizard_state.scope_end_date).date()
        if getattr(wizard_state, "scope_end_date", "")
        else None
    )
    scope_params = temporal_scope_parameters(
        scope_option=wizard_state.scope_option,
        custom_start_date=scope_start_date,
        custom_end_date=scope_end_date,
    )

    if review["execution_mode"] == "real" and credential.get("api_key"):
        upsert_workspace_credential(
            actor=actor,
            workspace=investigation.workspace,
            provider=review["provider"],
            api_key=credential["api_key"],
            base_url=credential.get("base_url") or "",
            request=request,
        )

    scope_json = investigation.scope_json if isinstance(investigation.scope_json, dict) else {}
    scope_json = {
        **scope_json,
        "temporal_scope_option": wizard_state.scope_option,
    }
    if scope_params.get("query_start_date") is not None:
        scope_json["query_start_date"] = str(scope_params["query_start_date"])
    else:
        scope_json.pop("query_start_date", None)
    if scope_params.get("query_end_date") is not None:
        scope_json["query_end_date"] = str(scope_params["query_end_date"])
    else:
        scope_json.pop("query_end_date", None)
    if scope_params.get("report_limit"):
        scope_json["report_limit"] = scope_params.get("report_limit")
    else:
        scope_json.pop("report_limit", None)

    method_json = investigation.method_json if isinstance(investigation.method_json, dict) else {}
    method_json = {
        **method_json,
        "pipeline_plan": pipeline_plan,
        "run_filter": bool(wizard_state.run_filter),
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
        "max_parallel_workers": int(review.get("max_parallel_workers") or 1),
        "search_query": wizard_state.question_text or investigation.question_text,
        "pipeline_plan": pipeline_plan,
        "pipeline_index": 0,
        "pipeline_continue_on_fail": True,
        "pipeline_require_upstream_artifact": False,
    }
    if bool(wizard_state.run_filter):
        filter_config = wizard_state.filter_config if isinstance(wizard_state.filter_config, dict) else {}
        search_query = str(
            filter_config.get("search_query") or wizard_state.question_text or investigation.question_text
        ).strip()
        if search_query:
            run_config["search_query"] = search_query
        run_config["filter_df"] = bool(filter_config.get("filter_df", True))
        run_config["produce_spans"] = bool(filter_config.get("include_supporting_quotes", False))
        run_config["drop_spans"] = False
        selected_filters = filter_config.get("selected_filters")
        if isinstance(selected_filters, dict):
            run_config["selected_filters"] = {
                "coroner": list(selected_filters.get("coroner", []) or []),
                "area": list(selected_filters.get("area", []) or []),
                "receiver": list(selected_filters.get("receiver", []) or []),
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
        run_config["produce_spans"] = bool(extract_config.get("include_supporting_quotes", False))
        run_config["drop_spans"] = False

    run = queue_run(
        actor=actor,
        investigation=investigation,
        run_type=pipeline_plan[0],
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
