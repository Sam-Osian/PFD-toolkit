# Run Worker Execution (v0.2)

Status: Implemented adapter-backed execution  
Last updated: 2026-04-18

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
2. Assigns `worker_id`.
3. Transitions statuses via `wb_runs.services.set_run_status`:
   - `queued -> starting -> running -> succeeded`
   - `cancelling -> cancelled`
   - failure path: `running -> failed`
   - timeout path: `running -> timed_out`
4. Writes `RunEvent` + `AuditEvent` on lifecycle transitions.
5. Creates a ready `RunArtifact` on success.

## 4. Execution Modes

`input_config_json.execution_mode` controls worker behavior for adapter-backed run types:

1. `real` (default): executes the real `pfd_toolkit` adapter.
2. `simulate`: uses staged simulation behavior.

Adapter-backed run types:

1. `filter`
2. `themes`
3. `extract`

Non-adapter run types (currently `export`) continue to use simulation stages.

## 5. Real Adapter Inputs (`input_config_json`)

### 5.1 Shared keys (`filter`, `themes`, `extract`)

1. `provider` (`openai` or `openrouter`)
2. `model_name`
3. `max_parallel_workers`
4. `llm_timeout_seconds`
5. `report_limit` (int)
6. `refresh` (bool)
7. `start_date` / `end_date` (`YYYY-MM-DD`, optional override)
8. `artifact_dir` (optional filesystem override for run outputs)
9. `execution_mode` (`real` or `simulate`)

Environment keys used by real adapter:

1. `OPENAI_API_KEY` for `provider=openai`
2. `OPENROUTER_API_KEY` for `provider=openrouter`

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

## 8. Railway Deployment Shape

Recommended services:

1. `web`: Django HTTP app
2. `worker`: command `uv run python manage.py run_runs_worker --worker-id railway-worker-1 --poll-seconds 3`

Optional later:

1. Redis-backed queue and push progress transport
2. Dedicated scheduler service for expiry/maintenance tasks

## 9. Next Upgrade Path

1. Add retry policy for transient provider failures (network/rate limit classes).
2. Add run-attempt model if multiple attempts per run are required.
3. Add metrics (queue depth, processing duration, failure rates) and Sentry instrumentation.
4. Move artifact files from local filesystem to managed object storage for production durability.
