# Run Worker Execution (v0.2)

Status: Implemented baseline  
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

## 4. Testing/Simulation Inputs

`InvestigationRun.input_config_json` supports simulation keys:

1. `simulate_failure: true`
2. `simulate_failure_stage: <int>`
3. `simulate_timeout: true`
4. `simulate_timeout_stage: <int>`

These are for worker and lifecycle testing; replace with real pipeline execution adapters later.

## 5. Useful Command Modes

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

## 6. Railway Deployment Shape

Recommended services:

1. `web`: Django HTTP app
2. `worker`: command `uv run python manage.py run_runs_worker --worker-id railway-worker-1 --poll-seconds 3`

Optional later:

1. Redis-backed queue and push progress transport
2. Dedicated scheduler service for expiry/maintenance tasks

## 7. Next Upgrade Path

1. Replace simulated stage logic in `wb_runs/worker.py` with real toolkit adapters.
2. Add retry policy for transient failures.
3. Add run-attempt model if multiple attempts per run are required.
4. Add metrics (queue depth, processing duration, failure rates) and Sentry instrumentation.
