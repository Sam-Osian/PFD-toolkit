# Operating Limits (MVP)

Status: Implemented  
Last updated: 2026-04-20

This file defines the default production guardrails and reliability controls for run execution.

## 1. Launch Guardrails

1. `MAX_RUNS_PER_USER_PER_DAY=20`
2. `MAX_RUNS_PER_WORKBOOK_PER_DAY=40`
3. `MAX_CONCURRENT_RUNS_PER_USER=2`
4. `MAX_CONCURRENT_RUNS_GLOBAL=12`
5. `RUN_LAUNCH_RATE_LIMIT_USER_PER_MINUTE=10`
6. `RUN_LAUNCH_RATE_LIMIT_IP_PER_MINUTE=30`

## 2. Retry and Timeout Policy

1. `RUN_RETRY_ENABLED=True`
2. `RUN_RETRY_MAX_ATTEMPTS=3`
3. `RUN_RETRY_BACKOFF_SECONDS=30,120,600`
4. `RUN_RETRY_JITTER_PCT=20`
5. `RUN_STAGE_TIMEOUT_SECONDS=1800`
6. `RUN_TOTAL_TIMEOUT_SECONDS=7200`

Retry scope:

1. Automatic retries are applied only to transient execution failures.
2. Validation/configuration failures are not retried automatically.
3. Retries requeue the same `InvestigationRun` record with a future `queued_at`.

## 3. Artifact Durability Defaults

1. `ARTIFACT_STORAGE_BACKEND=object_storage` (recommended production value)
2. `ARTIFACT_ENFORCE_OBJECT_STORAGE_IN_PRODUCTION=True`
3. `ARTIFACT_RETENTION_DAYS=365`
4. `ARTIFACT_SOFT_DELETE_DAYS=30`

## 4. Operations

Suggested scheduler tasks:

1. `uv run python manage.py run_lifecycle_maintenance`
2. `uv run python manage.py run_runs_worker --reconcile-timeouts-only`
3. `uv run python manage.py run_runs_worker --finalize-cancelling-only`
