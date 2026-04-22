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

## 6. Evidence Log (2026-04-22)

Production verification captured via Railway CLI + Django management commands against `DATABASE_PUBLIC_URL`.

CLI gates:

1. Services healthy: `web`, `worker`, `notification-dispatcher`, `Postgres` all `SUCCESS`.
2. Worker heartbeat gate passed:
   `Run worker heartbeat healthy: worker=railway-worker-1 state=idle last_seen_at=2026-04-22T16:36:56.532861+00:00`
3. Deployed smoke checks passed:
   `/` returned `HTTP/2 200`, `/auth/login/` returned `HTTP/2 302` to Auth0.

Manual gate evidence:

1. Live run terminal success + artifact present (deployed env):
   - `run_id`: `9fa4ae7f-796c-4f4c-993d-f3f811ad8c2e`
   - `run_type`: `filter`
   - `status`: `succeeded`
   - `worker_id`: `railway-worker-1`
   - `created_at`: `2026-04-22T16:41:18.334620+00:00`
   - `started_at`: `2026-04-22T16:41:19.310857+00:00`
   - `finished_at`: `2026-04-22T16:42:36.115430+00:00`
   - terminal event message: `Run completed successfully. Matched 25 of 100 reports.`
   - artifact:
     - `artifact_id`: `63b13393-2a39-42be-b05c-278dda060c6c`
     - `artifact_type`: `filtered_dataset`
     - `status`: `ready`
     - `storage_backend`: `object_storage`
     - `storage_uri`: `s3://pfd-workbench-artifacts/workbench-artifacts/workspace_697a7641-1be3-47cb-be80-9ab7ba180068/investigation_b171d276-25b5-4b7e-8e86-45b60750556f/run_9fa4ae7f-796c-4f4c-993d-f3f811ad8c2e/filtered_dataset/20260422T164235Z_67fd940e8e1e4d689a54de39c6ad7271_filtered_reports.csv`
2. Browser/session-closure continuity condition validated (user-confirmed browser closed immediately after launch) with terminal server-side completion:
   - `run_id`: `75b2526d-04a0-4345-8ab7-22b027545f9f`
   - `run_type`: `filter`
   - `status`: `succeeded`
   - `worker_id`: `railway-worker-1`
   - `created_at`: `2026-04-22T16:45:16.187527+00:00`
   - `started_at`: `2026-04-22T16:45:16.328753+00:00`
   - `finished_at`: `2026-04-22T16:45:22.503784+00:00`
   - terminal event message: `Run completed successfully. Matched 24 of 100 reports.`
   - artifact:
     - `artifact_id`: `cb65226e-0dc5-4b1a-96b0-a35608b00f6f`
     - `artifact_type`: `filtered_dataset`
     - `status`: `ready`
     - `storage_backend`: `object_storage`
3. Cancellation path validated (deployed env):
   - `run_id`: `759ba6b8-51be-4ab9-95f4-6c4c6d7a5b02`
   - `run_type`: `filter`
   - `status`: `cancelled`
   - `worker_id`: `railway-worker-1`
   - `created_at`: `2026-04-22T17:02:10.719341+00:00`
   - `cancel_requested_at`: `2026-04-22T17:02:12.941291+00:00`
   - `finished_at`: `2026-04-22T17:02:13.064278+00:00`
   - cancellation reason: `Cancelled from workspace dashboard.`
   - key events:
     - `cancel_check`: `Cancellation requested.`
     - terminal stage event: `Run was cancelled before processing started.`
4. Remaining manual checks still pending:
   - timeout path terminal `timed_out`
   - transient retry path requeue/backoff + terminal outcome
