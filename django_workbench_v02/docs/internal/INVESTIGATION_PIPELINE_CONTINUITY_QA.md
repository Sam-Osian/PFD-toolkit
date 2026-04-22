# Investigation Pipeline Continuity QA (Slice 2)

Status: Active continuity gate checklist  
Last updated: 2026-04-22

## 1. Purpose

Define explicit validation for worker handoff and end-to-end continuity so investigation pipelines complete server-side regardless of browser session state.

## 2. Continuity Release Gates

A release candidate passes Slice 2 only if all checks below are true:

1. Worker service is deployed and healthy.
2. Worker heartbeat is fresh (`check_run_worker_health` passes).
3. Launch -> close browser -> terminal run status still reached.
4. Cancellation path reaches terminal cancelled state.
5. Timeout reconciliation path reaches terminal timed_out state.
6. Transient retry path re-queues and then either succeeds or fails with explicit terminal status.

## 3. Test Matrix

## 3.1 Automated (local/CI)

1. `wb_runs.tests.RunWorkerTests.test_worker_processes_queued_run_to_success`
2. `wb_runs.tests.RunWorkerTests.test_worker_honors_pre_requested_cancellation`
3. `wb_runs.tests.RunWorkerTests.test_worker_records_timeout`
4. `wb_runs.tests.RunWorkerTests.test_transient_real_adapter_failure_requeues_run`
5. `wb_runs.tests.RunWorkerTests.test_transient_failure_exhausts_retries_then_fails`
6. `wb_runs.tests.RunWorkerTests.test_reconcile_timed_out_runs_marks_stale_run`
7. `wb_runs.tests.RunWorkerTests.test_worker_records_heartbeat_when_idle`
8. `wb_runs.tests.RunWorkerTests.test_worker_healthcheck_passes_when_recent_heartbeat_exists`
9. `wb_runs.tests.RunWorkerTests.test_worker_healthcheck_fails_when_heartbeat_stale`

## 3.2 Staging Manual

1. Launch a real investigation pipeline.
2. Close browser/session immediately after launch.
3. Confirm terminal run status and artifacts are produced.
4. Launch another run, request cancellation, confirm `cancelled` terminal status.
5. Trigger timeout path (staging config), confirm `timed_out` terminal status.
6. Trigger transient error path, confirm requeue backoff event then terminal outcome.

## 4. Operational Commands

Worker loop:

```bash
uv run python manage.py run_runs_worker --worker-id railway-worker-1 --poll-seconds 3
```

Heartbeat gate:

```bash
uv run python manage.py check_run_worker_health --stale-seconds 180
```

Timeout reconciliation only:

```bash
uv run python manage.py run_runs_worker --reconcile-timeouts-only
```

## 5. Notes

1. Heartbeat freshness gate is intentionally strict and should block "ready" claims if stale.
2. This slice validates continuity only; notification semantics are handled in Slice 3.
