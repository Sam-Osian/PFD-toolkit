from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from .models import (
    ArtifactStatus,
    ArtifactStorageBackend,
    ArtifactType,
    InvestigationRun,
    RunStatus,
    RunType,
    RunArtifact,
)
from .services import is_terminal_status, set_run_status


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


def _reload_run(run_id):
    return InvestigationRun.objects.select_related("workspace", "investigation").get(id=run_id)


def _cancel_requested(run) -> bool:
    current = _reload_run(run.id)
    return current.status == RunStatus.CANCELLING or current.cancel_requested_at is not None


@transaction.atomic
def claim_next_runnable_run(worker_id: str) -> InvestigationRun | None:
    run = (
        InvestigationRun.objects.select_for_update(skip_locked=True)
        .filter(status__in=[RunStatus.QUEUED, RunStatus.CANCELLING])
        .order_by("queued_at", "created_at")
        .first()
    )
    if run is None:
        return None

    run.worker_id = worker_id
    run.save(update_fields=["worker_id", "updated_at"])
    return run


def _create_success_artifact(run: InvestigationRun) -> None:
    artifact_type = ARTIFACT_TYPE_BY_RUN_TYPE.get(run.run_type, ArtifactType.PREVIEW)
    RunArtifact.objects.create(
        run=run,
        workspace=run.workspace,
        artifact_type=artifact_type,
        status=ArtifactStatus.READY,
        storage_backend=ArtifactStorageBackend.DB,
        storage_uri=f"db://run-artifacts/{run.id}",
        metadata_json={
            "generated_by_worker": run.worker_id or "",
            "run_type": run.run_type,
            "investigation_id": str(run.investigation_id),
            "completed_at": timezone.now().isoformat(),
        },
        expires_at=timezone.now() + timedelta(days=365),
    )


def _execute_run(run: InvestigationRun, sleep_between_stages_seconds: float = 0.0) -> InvestigationRun:
    run = _reload_run(run.id)

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


def process_single_available_run(
    *,
    worker_id: str | None = None,
    sleep_between_stages_seconds: float = 0.0,
) -> InvestigationRun | None:
    effective_worker_id = worker_id or f"worker-{uuid.uuid4()}"
    run = claim_next_runnable_run(effective_worker_id)
    if run is None:
        return None

    logger.info("Worker %s processing run %s", effective_worker_id, run.id)
    final_run = _execute_run(run, sleep_between_stages_seconds=sleep_between_stages_seconds)
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
