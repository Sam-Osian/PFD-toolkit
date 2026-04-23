from __future__ import annotations

import logging
import random
import time
import uuid
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from .artifact_storage import ArtifactStorageError, store_artifact_file
from .models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunStatus,
    RunType,
    RunArtifact,
    RunEvent,
    RunEventType,
    RunWorkerHeartbeat,
)
from .services import is_terminal_status, set_run_status
from .pfd_toolkit_adapter import (
    AdapterCancelledError,
    AdapterConfigurationError,
    execute_export_workflow,
    execute_extract_workflow,
    execute_filter_workflow,
    execute_themes_workflow,
)


logger = logging.getLogger(__name__)


RUN_STAGE_MAP = {
    RunType.FILTER: [
        ("Preparing filter inputs", 10),
        ("Applying report filters", 45),
        ("Compiling filtered dataset", 80),
    ],
    RunType.THEMES: [
        ("Preparing theme discovery", 10),
        ("Discovering themes", 50),
        ("Summarizing theme outputs", 85),
    ],
    RunType.EXTRACT: [
        ("Preparing extraction prompts", 10),
        ("Running extraction", 55),
        ("Compiling extraction table", 85),
    ],
    RunType.EXPORT: [
        ("Preparing export bundle", 20),
        ("Generating package", 60),
        ("Finalizing export output", 90),
    ],
}

ARTIFACT_TYPE_BY_RUN_TYPE = {
    RunType.FILTER: ArtifactType.FILTERED_DATASET,
    RunType.THEMES: ArtifactType.THEME_SUMMARY,
    RunType.EXTRACT: ArtifactType.EXTRACTION_TABLE,
    RunType.EXPORT: ArtifactType.BUNDLE_EXPORT,
}

REAL_ADAPTER_RUN_TYPES = {RunType.FILTER, RunType.THEMES, RunType.EXTRACT, RunType.EXPORT}

RUN_LABEL_BY_TYPE = {
    RunType.FILTER: "filter",
    RunType.THEMES: "themes",
    RunType.EXTRACT: "extract",
    RunType.EXPORT: "export",
}

EXECUTION_ERROR_CODE_BY_RUN_TYPE = {
    RunType.FILTER: "FILTER_EXECUTION_ERROR",
    RunType.THEMES: "THEMES_EXECUTION_ERROR",
    RunType.EXTRACT: "EXTRACT_EXECUTION_ERROR",
    RunType.EXPORT: "EXPORT_EXECUTION_ERROR",
}

PIPELINE_TERMINAL_CONTINUE_STATUSES = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}
PIPELINE_FAILED_UPSTREAM_STATUSES = {
    RunStatus.FAILED,
    RunStatus.TIMED_OUT,
}

TRANSIENT_ERROR_SNIPPETS = (
    "timeout",
    "timed out",
    "temporary",
    "temporarily",
    "rate limit",
    "too many requests",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "network is unreachable",
)

NON_TRANSIENT_ERROR_SNIPPETS = (
    "insufficient_quota",
    "insufficient quota",
)


def _reload_run(run_id):
    return InvestigationRun.objects.select_related("workspace", "investigation").get(id=run_id)


def _retry_enabled() -> bool:
    return bool(getattr(settings, "RUN_RETRY_ENABLED", True))


def _retry_max_attempts() -> int:
    return max(1, int(getattr(settings, "RUN_RETRY_MAX_ATTEMPTS", 3)))


def _retry_backoff_seconds() -> list[int]:
    raw = getattr(settings, "RUN_RETRY_BACKOFF_SECONDS", (30, 120, 600))
    if isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            try:
                values.append(max(0, int(item)))
            except (TypeError, ValueError):
                continue
        if values:
            return values
    return [30, 120, 600]


def _retry_jitter_pct() -> int:
    return max(0, int(getattr(settings, "RUN_RETRY_JITTER_PCT", 20)))


def _artifact_retention_days() -> int:
    return max(1, int(getattr(settings, "ARTIFACT_RETENTION_DAYS", 365)))


def _stage_timeout_seconds() -> int:
    return max(0, int(getattr(settings, "RUN_STAGE_TIMEOUT_SECONDS", 1800)))


def _total_timeout_seconds() -> int:
    return max(0, int(getattr(settings, "RUN_TOTAL_TIMEOUT_SECONDS", 7200)))


def _is_transient_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc).strip().lower()
    if not message:
        return False
    if any(snippet in message for snippet in NON_TRANSIENT_ERROR_SNIPPETS):
        return False
    return any(snippet in message for snippet in TRANSIENT_ERROR_SNIPPETS)


def _run_exceeded_timeouts(run: InvestigationRun) -> tuple[bool, str, str]:
    if run.status in {RunStatus.CANCELLED, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.TIMED_OUT}:
        return False, "", ""

    now = timezone.now()
    total_timeout = _total_timeout_seconds()
    if total_timeout > 0 and run.started_at:
        if now - run.started_at >= timedelta(seconds=total_timeout):
            return True, "RUN_TOTAL_TIMEOUT", "Run timed out after exceeding total runtime limit."

    stage_timeout = _stage_timeout_seconds()
    if stage_timeout > 0 and run.status in {RunStatus.STARTING, RunStatus.RUNNING}:
        if now - run.updated_at >= timedelta(seconds=stage_timeout):
            return True, "RUN_STAGE_TIMEOUT", "Run timed out due to stage inactivity."
    return False, "", ""


def _record_worker_heartbeat(
    *,
    worker_id: str,
    state: str,
    run: InvestigationRun | None = None,
    error: str = "",
) -> None:
    RunWorkerHeartbeat.objects.update_or_create(
        worker_id=worker_id,
        defaults={
            "state": str(state or "").strip().lower()[:32],
            "last_run": run,
            "last_run_status": str(getattr(run, "status", "") or "")[:16],
            "last_error": str(error or "")[:2000],
            "last_seen_at": timezone.now(),
        },
    )


def _apply_timeout_if_needed(run: InvestigationRun) -> InvestigationRun:
    refreshed = _reload_run(run.id)
    exceeded, error_code, message = _run_exceeded_timeouts(refreshed)
    if not exceeded:
        return refreshed
    return set_run_status(
        run=refreshed,
        status=RunStatus.TIMED_OUT,
        message=message,
        progress_percent=refreshed.progress_percent or 0,
        error_code=error_code,
        error_message=message,
    )


def _schedule_transient_retry(*, run: InvestigationRun, exc: Exception) -> InvestigationRun | None:
    if not _retry_enabled():
        return None

    config = run.input_config_json if isinstance(run.input_config_json, dict) else {}
    current_attempt = int(config.get("_retry_attempt", 0) or 0)
    max_attempts = _retry_max_attempts()
    next_attempt = current_attempt + 1
    if next_attempt >= max_attempts:
        return None

    backoff_values = _retry_backoff_seconds()
    backoff_seconds = backoff_values[min(current_attempt, len(backoff_values) - 1)]
    jitter_pct = _retry_jitter_pct()
    jitter_window = (backoff_seconds * jitter_pct) / 100.0
    jitter_offset = random.uniform(-jitter_window, jitter_window) if jitter_window else 0.0
    wait_seconds = max(0, int(round(backoff_seconds + jitter_offset)))
    retry_at = timezone.now() + timedelta(seconds=wait_seconds)

    updated_config = deepcopy(config)
    updated_config["_retry_attempt"] = next_attempt
    updated_config["_retry_max_attempts"] = max_attempts
    updated_config["_retry_backoff_seconds"] = backoff_values
    updated_config["_retry_last_error"] = str(exc)[:512]

    retry_run = set_run_status(
        run=run,
        status=RunStatus.QUEUED,
        message=(
            f"Transient error detected. Auto-retrying in {wait_seconds}s "
            f"(attempt {next_attempt + 1}/{max_attempts})."
        ),
        event_type=RunEventType.WARNING,
        progress_percent=run.progress_percent or 0,
        error_code="TRANSIENT_RETRY_SCHEDULED",
        error_message=str(exc),
    )
    retry_run.input_config_json = updated_config
    retry_run.queued_at = retry_at
    retry_run.worker_id = ""
    retry_run.error_code = ""
    retry_run.error_message = ""
    retry_run.save(
        update_fields=[
            "input_config_json",
            "queued_at",
            "worker_id",
            "error_code",
            "error_message",
            "updated_at",
        ]
    )
    RunEvent.objects.create(
        run=retry_run,
        event_type=RunEventType.INFO,
        message="Run re-queued after transient failure.",
        payload_json={
            "retry_attempt": next_attempt,
            "retry_max_attempts": max_attempts,
            "retry_at": retry_at.isoformat(),
            "retry_backoff_seconds": wait_seconds,
        },
    )
    return retry_run


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


def _pipeline_index_from_config(config: dict, *, plan: list[str], fallback_run_type: str) -> int:
    raw_index = config.get("pipeline_index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(plan):
        return index
    try:
        return plan.index(str(fallback_run_type or "").strip().lower())
    except ValueError:
        return -1


def _latest_ready_artifact_for_run(*, run: InvestigationRun, preferred_types: list[str]) -> RunArtifact | None:
    for artifact_type in preferred_types:
        artifact = (
            run.artifacts.filter(
                status=ArtifactStatus.READY,
                artifact_type=artifact_type,
            )
            .order_by("-created_at")
            .first()
        )
        if artifact is not None:
            return artifact
    return run.artifacts.filter(status=ArtifactStatus.READY).order_by("-created_at").first()


def _upstream_chaining_artifact(run: InvestigationRun) -> RunArtifact | None:
    if run.run_type == RunType.FILTER:
        return _latest_ready_artifact_for_run(
            run=run,
            preferred_types=[ArtifactType.FILTERED_DATASET],
        )
    if run.run_type == RunType.THEMES:
        return _latest_ready_artifact_for_run(
            run=run,
            preferred_types=[ArtifactType.THEME_ASSIGNMENTS, ArtifactType.THEME_SUMMARY],
        )
    if run.run_type == RunType.EXTRACT:
        return _latest_ready_artifact_for_run(
            run=run,
            preferred_types=[ArtifactType.EXTRACTION_TABLE],
        )
    return None


def _queue_next_pipeline_run(current_run: InvestigationRun) -> InvestigationRun | None:
    config = current_run.input_config_json or {}
    pipeline_plan = _normalise_pipeline_plan(config.get("pipeline_plan"))
    if not pipeline_plan:
        return None

    continue_on_fail = bool(config.get("pipeline_continue_on_fail", False))
    if current_run.status == RunStatus.CANCELLED:
        return None
    if current_run.status not in PIPELINE_TERMINAL_CONTINUE_STATUSES:
        return None
    if not continue_on_fail and current_run.status != RunStatus.SUCCEEDED:
        return None

    current_index = _pipeline_index_from_config(
        config,
        plan=pipeline_plan,
        fallback_run_type=current_run.run_type,
    )
    if current_index < 0:
        return None
    next_index = current_index + 1
    if next_index >= len(pipeline_plan):
        return None

    next_run_type = pipeline_plan[next_index]
    next_config = deepcopy(config)
    next_config["pipeline_index"] = next_index
    next_config["pipeline_plan"] = pipeline_plan
    next_config["pipeline_continue_on_fail"] = continue_on_fail
    continued_after_failed_upstream = bool(
        continue_on_fail and current_run.status in PIPELINE_FAILED_UPSTREAM_STATUSES
    )

    upstream_artifact = _upstream_chaining_artifact(current_run)
    if upstream_artifact is not None:
        next_config["input_artifact_id"] = str(upstream_artifact.id)
    else:
        next_config.pop("input_artifact_id", None)
    next_config["pipeline_require_upstream_artifact"] = bool(
        next_run_type in {RunType.THEMES, RunType.EXTRACT}
    )

    next_run = InvestigationRun.objects.create(
        investigation=current_run.investigation,
        workspace=current_run.workspace,
        requested_by=current_run.requested_by,
        run_type=next_run_type,
        status=RunStatus.QUEUED,
        input_config_json=next_config,
        query_start_date=current_run.query_start_date,
        query_end_date=current_run.query_end_date,
    )
    RunEvent.objects.create(
        run=next_run,
        event_type=RunEventType.INFO,
        message="Run queued by investigation pipeline.",
        payload_json={
            "status": next_run.status,
            "pipeline_previous_run_id": str(current_run.id),
            "pipeline_index": next_index,
            "pipeline_plan": pipeline_plan,
            "input_artifact_id": str(upstream_artifact.id) if upstream_artifact else "",
            "continued_after_failed_upstream": continued_after_failed_upstream,
        },
    )
    return next_run


def _cancel_requested(run) -> bool:
    current = _reload_run(run.id)
    return current.status == RunStatus.CANCELLING or current.cancel_requested_at is not None


@transaction.atomic
def claim_next_runnable_run(worker_id: str) -> InvestigationRun | None:
    now = timezone.now()
    run = (
        InvestigationRun.objects.select_for_update(skip_locked=True)
        .filter(
            Q(status=RunStatus.CANCELLING)
            | Q(status=RunStatus.QUEUED, queued_at__lte=now)
        )
        .order_by("queued_at", "created_at")
        .first()
    )
    if run is None:
        return None

    run.worker_id = worker_id
    run.save(update_fields=["worker_id", "updated_at"])
    return run


def _create_success_artifact(
    run: InvestigationRun,
    *,
    artifact_type: str | None = None,
    storage_backend: str = ArtifactStorageBackend.DB,
    storage_uri: str | None = None,
    size_bytes: int | None = None,
    metadata: dict | None = None,
) -> None:
    selected_artifact_type = artifact_type or ARTIFACT_TYPE_BY_RUN_TYPE.get(
        run.run_type, ArtifactType.PREVIEW
    )
    resolved_size_bytes = size_bytes
    if resolved_size_bytes is None and storage_backend == ArtifactStorageBackend.FILE and storage_uri:
        try:
            resolved_size_bytes = Path(storage_uri).stat().st_size
        except OSError:
            resolved_size_bytes = None

    RunArtifact.objects.create(
        run=run,
        workspace=run.workspace,
        artifact_type=selected_artifact_type,
        status=ArtifactStatus.READY,
        storage_backend=storage_backend,
        storage_uri=storage_uri or f"db://run-artifacts/{run.id}",
        size_bytes=resolved_size_bytes,
        metadata_json={
            "generated_by_worker": run.worker_id or "",
            "run_type": run.run_type,
            "investigation_id": str(run.investigation_id),
            "completed_at": timezone.now().isoformat(),
            **(metadata or {}),
        },
        expires_at=timezone.now() + timedelta(days=_artifact_retention_days()),
    )


def _is_real_adapter_mode(run: InvestigationRun) -> bool:
    config = run.input_config_json or {}
    mode = str(config.get("execution_mode", "real")).strip().lower()
    if run.run_type not in REAL_ADAPTER_RUN_TYPES:
        return False
    return mode != "simulate"


def _resolve_real_adapter(run_type: str):
    # Resolve at call time so tests can monkeypatch adapter functions safely.
    if run_type == RunType.FILTER:
        return execute_filter_workflow
    if run_type == RunType.THEMES:
        return execute_themes_workflow
    if run_type == RunType.EXTRACT:
        return execute_extract_workflow
    if run_type == RunType.EXPORT:
        return execute_export_workflow
    return None


def _build_success_message(run: InvestigationRun, result: dict) -> str:
    if run.run_type == RunType.FILTER:
        return (
            "Run completed successfully. "
            f"Matched {result.get('matched_reports', 0):,} of {result.get('total_reports', 0):,} reports."
        )
    if run.run_type == RunType.THEMES:
        return (
            "Run completed successfully. "
            f"Discovered {result.get('discovered_themes', 0):,} themes from {result.get('total_reports', 0):,} reports."
        )
    if run.run_type == RunType.EXTRACT:
        return (
            "Run completed successfully. "
            f"Extracted features for {result.get('output_reports', 0):,} reports."
        )
    if run.run_type == RunType.EXPORT:
        return (
            "Run completed successfully. "
            f"Packaged {result.get('included_files', 0):,} files across "
            f"{result.get('selected_artifacts', 0):,} artifacts."
        )
    return "Run completed successfully."


def _execute_real_adapter_run(run: InvestigationRun) -> InvestigationRun:
    run = _reload_run(run.id)
    run_label = RUN_LABEL_BY_TYPE.get(run.run_type, run.run_type)
    adapter = _resolve_real_adapter(run.run_type)
    if adapter is None:
        raise ValueError(f"No real adapter configured for run_type={run.run_type!r}")

    run = set_run_status(
        run=run,
        status=RunStatus.RUNNING,
        message=f"Starting real {run_label} workflow.",
        progress_percent=2,
    )

    def _cancellation_check() -> bool:
        current = _reload_run(run.id)
        current = _apply_timeout_if_needed(current)
        if current.status == RunStatus.TIMED_OUT:
            raise AdapterCancelledError("Run timed out during cancellation checks.")
        return current.status == RunStatus.CANCELLING or current.cancel_requested_at is not None

    def _progress_update(progress_percent: int, message: str) -> None:
        current = _reload_run(run.id)
        current = _apply_timeout_if_needed(current)
        if current.status == RunStatus.TIMED_OUT:
            raise AdapterCancelledError("Run timed out during progress updates.")
        if current.status in {RunStatus.CANCELLING, RunStatus.CANCELLED}:
            raise AdapterCancelledError("Run cancelled during worker progress updates.")
        if is_terminal_status(current.status):
            raise AdapterCancelledError("Run already reached terminal status.")
        set_run_status(
            run=current,
            status=RunStatus.RUNNING,
            message=message,
            progress_percent=progress_percent,
        )

    try:
        result = adapter(
            run=_reload_run(run.id),
            progress_callback=_progress_update,
            cancellation_check=_cancellation_check,
        )
    except AdapterCancelledError:
        current = _reload_run(run.id)
        if is_terminal_status(current.status):
            return current
        return set_run_status(
            run=current,
            status=RunStatus.CANCELLED,
            message=f"Run cancelled during {run_label} workflow.",
            progress_percent=current.progress_percent or 0,
        )
    except AdapterConfigurationError as exc:
        current = _reload_run(run.id)
        error_code = str(getattr(exc, "error_code", "") or "ADAPTER_CONFIGURATION")
        return set_run_status(
            run=current,
            status=RunStatus.FAILED,
            message=f"Run failed due to {run_label} adapter configuration error.",
            progress_percent=current.progress_percent or 0,
            error_code=error_code,
            error_message=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive fail-safe path
        current = _reload_run(run.id)
        logger.exception(
            "Run %s failed during %s workflow execution.",
            run.id,
            run_label,
        )
        if _is_transient_exception(exc):
            retried = _schedule_transient_retry(run=current, exc=exc)
            if retried is not None:
                return retried
        return set_run_status(
            run=current,
            status=RunStatus.FAILED,
            message=f"Run failed during {run_label} workflow execution.",
            progress_percent=current.progress_percent or 0,
            error_code=EXECUTION_ERROR_CODE_BY_RUN_TYPE.get(run.run_type, "WORKFLOW_EXECUTION_ERROR"),
            error_message=str(exc),
        )

    current = _reload_run(run.id)
    current = _apply_timeout_if_needed(current)
    if current.status == RunStatus.TIMED_OUT:
        return current
    if _cancellation_check():
        return set_run_status(
            run=current,
            status=RunStatus.CANCELLED,
            message=f"Run cancelled after {run_label} workflow completed.",
            progress_percent=current.progress_percent or 95,
        )

    output_path = result.get("output_path")
    storage_backend = ArtifactStorageBackend.DB
    storage_uri = output_path
    size_bytes = None
    if output_path:
        selected_artifact_type = ARTIFACT_TYPE_BY_RUN_TYPE.get(run.run_type)
        try:
            stored_file = store_artifact_file(
                source_path=Path(output_path),
                run=current,
                artifact_type=selected_artifact_type or ArtifactType.PREVIEW,
            )
        except ArtifactStorageError as exc:
            return set_run_status(
                run=current,
                status=RunStatus.FAILED,
                message="Run failed while persisting output artifact.",
                progress_percent=current.progress_percent or 95,
                error_code="ARTIFACT_STORAGE_ERROR",
                error_message=str(exc),
            )
        storage_backend = stored_file.storage_backend
        storage_uri = stored_file.storage_uri
        size_bytes = stored_file.size_bytes

    metadata = {"adapter_workflow": run_label}
    metadata.update({key: value for key, value in result.items() if key != "output_path"})

    _create_success_artifact(
        current,
        artifact_type=ARTIFACT_TYPE_BY_RUN_TYPE.get(run.run_type),
        storage_backend=storage_backend,
        storage_uri=storage_uri,
        size_bytes=size_bytes,
        metadata=metadata,
    )

    if run.run_type == RunType.THEMES and result.get("theme_assignments_path"):
        assignments_backend = ArtifactStorageBackend.FILE
        assignments_uri = result.get("theme_assignments_path")
        assignments_size_bytes = None
        if assignments_uri:
            try:
                stored_assignments = store_artifact_file(
                    source_path=Path(assignments_uri),
                    run=current,
                    artifact_type=ArtifactType.THEME_ASSIGNMENTS,
                )
            except ArtifactStorageError as exc:
                return set_run_status(
                    run=current,
                    status=RunStatus.FAILED,
                    message="Run failed while persisting theme assignment artifact.",
                    progress_percent=current.progress_percent or 95,
                    error_code="ARTIFACT_STORAGE_ERROR",
                    error_message=str(exc),
                )
            assignments_backend = stored_assignments.storage_backend
            assignments_uri = stored_assignments.storage_uri
            assignments_size_bytes = stored_assignments.size_bytes

        _create_success_artifact(
            current,
            artifact_type=ArtifactType.THEME_ASSIGNMENTS,
            storage_backend=assignments_backend,
            storage_uri=assignments_uri,
            size_bytes=assignments_size_bytes,
            metadata={
                "adapter_workflow": run_label,
                "source": "theme_assignments",
                "theme_summary_path": result.get("theme_summary_path"),
                "theme_schema_path": result.get("theme_schema_path"),
                "report_summaries_path": result.get("report_summaries_path"),
            },
        )

    return set_run_status(
        run=current,
        status=RunStatus.SUCCEEDED,
        message=_build_success_message(current, result),
        progress_percent=100,
    )


def _execute_run_simulated(run: InvestigationRun, sleep_between_stages_seconds: float = 0.0) -> InvestigationRun:
    run = _reload_run(run.id)
    run = _apply_timeout_if_needed(run)
    if run.status == RunStatus.TIMED_OUT:
        return run

    if run.status == RunStatus.CANCELLING:
        return set_run_status(
            run=run,
            status=RunStatus.CANCELLED,
            message="Run was cancelled before processing started.",
        )

    run = set_run_status(
        run=run,
        status=RunStatus.RUNNING,
        message="Run execution started.",
        progress_percent=0,
    )

    stages = RUN_STAGE_MAP.get(
        run.run_type,
        [("Preparing run", 20), ("Processing run", 70), ("Finalizing run", 90)],
    )

    simulate_failure = bool((run.input_config_json or {}).get("simulate_failure"))
    simulate_timeout = bool((run.input_config_json or {}).get("simulate_timeout"))
    fail_stage = int((run.input_config_json or {}).get("simulate_failure_stage", 2))
    timeout_stage = int((run.input_config_json or {}).get("simulate_timeout_stage", 2))

    for index, (message, progress) in enumerate(stages, start=1):
        run = _reload_run(run.id)
        run = _apply_timeout_if_needed(run)
        if run.status == RunStatus.TIMED_OUT:
            return run
        if _cancel_requested(run):
            return set_run_status(
                run=run,
                status=RunStatus.CANCELLED,
                message="Run cancelled by user request.",
                progress_percent=run.progress_percent or progress,
            )

        if simulate_failure and index == fail_stage:
            return set_run_status(
                run=run,
                status=RunStatus.FAILED,
                message="Run failed during simulated worker execution.",
                progress_percent=progress,
                error_code="SIMULATED_FAILURE",
                error_message="Simulated worker failure (testing path).",
            )

        if simulate_timeout and index == timeout_stage:
            return set_run_status(
                run=run,
                status=RunStatus.TIMED_OUT,
                message="Run timed out during simulated worker execution.",
                progress_percent=progress,
                error_code="SIMULATED_TIMEOUT",
                error_message="Simulated worker timeout (testing path).",
            )

        run = set_run_status(
            run=run,
            status=RunStatus.RUNNING,
            message=message,
            progress_percent=progress,
        )
        if sleep_between_stages_seconds > 0:
            time.sleep(sleep_between_stages_seconds)

    run = _reload_run(run.id)
    _create_success_artifact(run)
    return set_run_status(
        run=run,
        status=RunStatus.SUCCEEDED,
        message="Run completed successfully.",
        progress_percent=100,
    )


def _execute_run(run: InvestigationRun, sleep_between_stages_seconds: float = 0.0) -> InvestigationRun:
    run = _reload_run(run.id)
    run = _apply_timeout_if_needed(run)
    if run.status == RunStatus.TIMED_OUT:
        return run
    if run.status == RunStatus.CANCELLING:
        return set_run_status(
            run=run,
            status=RunStatus.CANCELLED,
            message="Run was cancelled before processing started.",
        )

    run = set_run_status(
        run=run,
        status=RunStatus.STARTING,
        message=f"Worker {run.worker_id or 'unknown'} claimed run.",
    )

    if _is_real_adapter_mode(run):
        return _execute_real_adapter_run(run)

    return _execute_run_simulated(
        run=run,
        sleep_between_stages_seconds=sleep_between_stages_seconds,
    )


def reconcile_timed_out_runs(*, worker_id: str | None = None) -> int:
    effective_worker_id = worker_id or f"worker-{uuid.uuid4()}"
    count = 0
    runs = InvestigationRun.objects.filter(
        status__in=[RunStatus.STARTING, RunStatus.RUNNING],
        finished_at__isnull=True,
    ).order_by("queued_at")
    for run in runs:
        run.worker_id = run.worker_id or effective_worker_id
        run.save(update_fields=["worker_id", "updated_at"])
        refreshed = _reload_run(run.id)
        exceeded, error_code, message = _run_exceeded_timeouts(refreshed)
        if not exceeded:
            continue
        set_run_status(
            run=refreshed,
            status=RunStatus.TIMED_OUT,
            message=message,
            progress_percent=refreshed.progress_percent or 0,
            error_code=error_code,
            error_message=message,
        )
        count += 1
    return count


def process_single_available_run(
    *,
    worker_id: str | None = None,
    sleep_between_stages_seconds: float = 0.0,
) -> InvestigationRun | None:
    effective_worker_id = worker_id or f"worker-{uuid.uuid4()}"
    _record_worker_heartbeat(worker_id=effective_worker_id, state="polling")
    reconcile_timed_out_runs(worker_id=effective_worker_id)
    run = claim_next_runnable_run(effective_worker_id)
    if run is None:
        _record_worker_heartbeat(worker_id=effective_worker_id, state="idle")
        return None

    _record_worker_heartbeat(worker_id=effective_worker_id, state="claimed", run=run)
    logger.info("Worker %s processing run %s", effective_worker_id, run.id)
    try:
        final_run = _execute_run(run, sleep_between_stages_seconds=sleep_between_stages_seconds)
    except Exception as exc:  # pragma: no cover - defensive heartbeat for crash diagnostics
        _record_worker_heartbeat(
            worker_id=effective_worker_id,
            state="error",
            run=run,
            error=str(exc),
        )
        raise

    _record_worker_heartbeat(worker_id=effective_worker_id, state="processed", run=final_run)
    try:
        _queue_next_pipeline_run(final_run)
    except Exception:  # pragma: no cover - defensive so pipeline issues don't kill worker
        logger.exception("Worker %s failed while queueing pipeline continuation for run %s", effective_worker_id, run.id)
        _record_worker_heartbeat(
            worker_id=effective_worker_id,
            state="pipeline_queue_error",
            run=final_run,
            error="Failed while queueing pipeline continuation.",
        )
    logger.info("Worker %s finished run %s with status=%s", effective_worker_id, run.id, final_run.status)
    return final_run


def run_worker_loop(
    *,
    worker_id: str | None = None,
    poll_seconds: float = 5.0,
    max_runs: int | None = None,
    sleep_between_stages_seconds: float = 0.0,
) -> int:
    effective_worker_id = worker_id or f"worker-{uuid.uuid4()}"
    processed = 0
    while True:
        run = process_single_available_run(
            worker_id=effective_worker_id,
            sleep_between_stages_seconds=sleep_between_stages_seconds,
        )
        if run is None:
            if max_runs is not None and processed >= max_runs:
                return processed
            time.sleep(poll_seconds)
            continue

        processed += 1
        if max_runs is not None and processed >= max_runs:
            return processed


def finalize_stuck_cancellations(*, worker_id: str | None = None) -> int:
    effective_worker_id = worker_id or f"worker-{uuid.uuid4()}"
    count = 0
    cancelling_runs = InvestigationRun.objects.filter(
        status=RunStatus.CANCELLING,
        finished_at__isnull=True,
    ).order_by("queued_at")
    for run in cancelling_runs:
        run.worker_id = run.worker_id or effective_worker_id
        run.save(update_fields=["worker_id", "updated_at"])
        run = _reload_run(run.id)
        if not is_terminal_status(run.status):
            set_run_status(
                run=run,
                status=RunStatus.CANCELLED,
                message="Cancellation finalized by worker reconciliation.",
            )
            count += 1
    return count
