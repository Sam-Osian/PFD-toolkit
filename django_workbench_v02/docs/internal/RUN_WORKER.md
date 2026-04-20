# Run Worker Execution (v0.2)

Status: Implemented adapter-backed execution + guardrails/retries  
Last updated: 2026-04-20

## 1. Purpose

This worker executes queued investigation runs server-side so jobs continue even after users leave the web page.

## 2. Entry Point

Management command:

```bash
uv run python manage.py run_runs_worker
```

Command file:

- `wb_runs/management/commands/run_runs_worker.py`

Core processing module:

- `wb_runs/worker.py`
- `wb_runs/pfd_toolkit_adapter.py`

## 3. Worker Behavior

1. Claims queued/cancelling runs using DB row locks (`select_for_update(skip_locked=True)`).
2. Only claims queued runs where `queued_at <= now` (supports retry backoff windows).
3. Reconciles stale in-flight runs to `timed_out` before claiming new runs.
4. Assigns `worker_id`.
5. Transitions statuses via `wb_runs.services.set_run_status`:
   - `queued -> starting -> running -> succeeded`
   - `running -> queued` (transient auto-retry path)
   - `cancelling -> cancelled`
   - failure path: `running -> failed`
   - timeout path: `running -> timed_out`
6. Writes `RunEvent` + `AuditEvent` on lifecycle transitions.
7. Creates a ready `RunArtifact` on success.

Run launch guardrails are enforced at queue time:

1. Daily caps (`MAX_RUNS_PER_USER_PER_DAY`, `MAX_RUNS_PER_WORKBOOK_PER_DAY`)
2. Concurrency caps (`MAX_CONCURRENT_RUNS_PER_USER`, `MAX_CONCURRENT_RUNS_GLOBAL`)
3. Rate limits (`RUN_LAUNCH_RATE_LIMIT_USER_PER_MINUTE`, `RUN_LAUNCH_RATE_LIMIT_IP_PER_MINUTE`)

## 4. Execution Modes

`input_config_json.execution_mode` controls worker behavior for adapter-backed run types:

1. `real` (default): executes the real `pfd_toolkit` adapter.
2. `simulate`: uses staged simulation behavior.

Adapter-backed run types:

1. `filter`
2. `themes`
3. `extract`
4. `export`

If a run sets `execution_mode=simulate`, staged simulation is used for testing regardless of type.

## 5. Real Adapter Inputs (`input_config_json`)

### 5.1 Shared keys (`filter`, `themes`, `extract`, `export`)

1. `provider` (`openai` or `openrouter`)
2. `model_name`
3. `max_parallel_workers`
4. `llm_timeout_seconds`
5. `report_limit` (int)
6. `refresh` (bool)
7. `start_date` / `end_date` (`YYYY-MM-DD`, optional override)
8. `artifact_dir` (optional filesystem override for run outputs)
9. `execution_mode` (`real` or `simulate`)

Credential source used by real adapter:

1. API keys are resolved from encrypted workspace credentials (`wb_workspaces.WorkspaceCredential`) for the run requester.
2. Runs fail with adapter-configuration error if no matching credential exists for the selected provider.

Artifact persistence keys:

1. `ARTIFACT_STORAGE_BACKEND` (`file` or `object_storage`)
2. `ARTIFACT_OBJECT_STORAGE_BUCKET` (required for object storage mode)
3. `ARTIFACT_OBJECT_STORAGE_REGION` (optional)
4. `ARTIFACT_OBJECT_STORAGE_ENDPOINT_URL` (optional, for S3-compatible providers)
5. `ARTIFACT_OBJECT_STORAGE_ACCESS_KEY_ID` (optional; falls back to environment/instance auth)
6. `ARTIFACT_OBJECT_STORAGE_SECRET_ACCESS_KEY` (optional; falls back to environment/instance auth)
7. `ARTIFACT_OBJECT_STORAGE_PREFIX` (optional object key prefix)
8. `ARTIFACT_STORAGE_DELETE_LOCAL_AFTER_UPLOAD` (default `true`)
9. `ARTIFACT_ENFORCE_OBJECT_STORAGE_IN_PRODUCTION` (default `true`)
10. `ARTIFACT_RETENTION_DAYS` (default `365`)

### 5.2 Filter-specific keys

1. `search_query` (optional; defaults to investigation question text)
2. `filter_df` (bool)
3. `produce_spans` (bool)
4. `drop_spans` (bool)

### 5.3 Themes-specific keys

1. `trim_approach` (`truncate` or `summarise`)
2. `summarise_intensity` (`low`, `medium`, `high`, `very high`)
3. `max_tokens` or `max_words` (truncate mode only)
4. `extra_theme_instructions` (optional)
5. `warning_threshold` / `error_threshold`
6. `min_themes` / `max_themes`
7. `seed_topics` (JSON or newline-separated text)

Theme runs produce:

1. Summary artifact (`theme_summary.csv`)
2. Assignments artifact (`theme_assignments.csv`)
3. Schema path and report summaries path in artifact metadata

### 5.4 Extract-specific keys

1. `feature_fields` (required list of `{name, type, description}`)
2. `produce_spans` (bool)
3. `drop_spans` (bool)
4. `force_assign` (bool)
5. `allow_multiple` (bool)
6. `skip_if_present` (bool)
7. `extra_instructions` (optional)

Extract runs produce:

1. Extraction table artifact (`extraction_table.csv`)
2. Feature schema path in artifact metadata

### 5.5 Export-specific keys

1. `include_run_types` (optional list, e.g. `["filter","themes","extract"]`)
2. `latest_per_artifact_type` (bool, default `true`)
3. `max_artifacts` (int, capped for safety)
4. `bundle_name` (optional custom zip filename)

Export runs produce:

1. Bundle artifact (`bundle_export`) as zip
2. `manifest.json` inside zip with included/skipped artifact metadata

## 6. Testing/Simulation Inputs

`InvestigationRun.input_config_json` supports simulation keys:

1. `simulate_failure: true`
2. `simulate_failure_stage: <int>`
3. `simulate_timeout: true`
4. `simulate_timeout_stage: <int>`

These are for worker and lifecycle testing when `execution_mode=simulate`.

## 7. Useful Command Modes

One run then exit:

```bash
uv run python manage.py run_runs_worker --once
```

Loop worker with explicit id:

```bash
uv run python manage.py run_runs_worker --worker-id railway-worker-1 --poll-seconds 3
```

Process only cancellations reconciliation:

```bash
uv run python manage.py run_runs_worker --finalize-cancelling-only
```

Process only timeout reconciliation:

```bash
uv run python manage.py run_runs_worker --reconcile-timeouts-only
```

## 8. Railway Deployment Shape

Recommended services:

1. `web`: Django HTTP app
2. `worker`: command `uv run python manage.py run_runs_worker --worker-id railway-worker-1 --poll-seconds 3`
3. `scheduler` (or cron): command `uv run python manage.py run_lifecycle_maintenance`
4. `notification-dispatcher`: command `uv run python manage.py run_notification_dispatcher --poll-seconds 5 --max-items 50`

Optional later:

1. Redis-backed queue and push progress transport
2. Dedicated credential KMS integration for key wrapping/rotation beyond app-level encryption key

## 9. Artifact Download Endpoint

Run artifacts with `status=ready` can be downloaded via:

`/workspaces/<workspace_id>/runs/<run_id>/artifacts/<artifact_id>/download/`

Behavior:

1. Uses workspace-level view permissions.
2. Supports both local-file and object-storage backends.
3. Emits `run.artifact_downloaded` audit events.
4. Updates `last_viewed_at` keepalive timestamps for artifact/workspace/investigation only for likely human user-agents.

## 10. Next Upgrade Path

1. Add run-attempt model if per-attempt analytics become a requirement.
2. Add metrics (queue depth, processing duration, failure rates) and Sentry instrumentation.
3. Validate scheduler cadence and alerting thresholds in Railway production.
