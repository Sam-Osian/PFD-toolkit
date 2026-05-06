# DGX Spark Migration Staged Plan

## Stage 1 Decisions (Locked 2026-05-06)

1. Routing and execution defaults:
   - Default path uses local DGX execution with Ollama (`local_ollama` provider route).
   - Advanced users can explicitly choose API providers (OpenAI/OpenRouter path remains available).
2. Worker topology:
   - Option A selected: DGX worker handles local-ollama runs.
   - Railway worker remains for API-provider runs.
3. Approval policy scope:
   - Option A selected: approval/scheduling applies to local DGX runs only.
   - API runs continue to launch immediately through existing behavior.
4. Model and inference defaults for local route:
   - Model tag: `gemma4:27b`.
   - Reasoning off (`reasoning_effort=none`).
   - `temperature=0`.
5. Approval UX direction:
   - Approval should be tied to existing admin workflows and include scheduling controls (not only approve/reject links).
6. Artifact storage architecture check:
   - Confirmed on Railway `web` and `worker`: `ARTIFACT_STORAGE_BACKEND=object_storage`.
   - This is compatible with DGX-run artifact generation and Railway web artifact downloads.
7. Operational recommendation for DGX worker:
   - Run the DGX worker under `systemd` for restart-on-failure, boot persistence, and straightforward logging.

## Stage 1: Lock Target Architecture

1. Confirm runtime split:
   - Railway runs `web + postgres + notification-dispatcher`.
   - DGX runs the single `run_runs_worker`.
2. Confirm default model route:
   - `local_ollama` with Gemma 4, reasoning off, temperature 0.
3. Define env contract (for example `LOCAL_OLLAMA_BASE_URL`, `LOCAL_OLLAMA_MODEL`, and related settings).
4. Exit criteria:
   - Architecture and configuration spec agreed.

## Stage 2: Data Model for Approval Gate

1. Add run approval fields/statuses (for example `pending_approval`, `approved_at`, `approved_by`, `approval_token`).
2. Add migration and admin visibility for approval state.
3. Exit criteria:
   - Runs can be queued but blocked until approved.

### Stage 2 Implementation Notes (Completed 2026-05-06)

1. Added approval data model fields on `InvestigationRun`:
   - `requires_approval`
   - `approval_status` (`not_required`, `pending`, `approved`, `rejected`)
   - `approval_requested_at`, `approved_at`, `approved_by`, `rejected_at`, `rejected_by`
   - `approval_note`
2. Added migration:
   - `wb_runs/migrations/0004_investigationrun_approval_note_and_more.py`
3. Added admin visibility and controls:
   - Approval columns and filters in run admin list.
   - Superuser-only bulk actions to approve now or reject selected runs.
   - `queued_at` is editable in admin to support scheduling semantics.
4. Added worker claim gate:
   - Queued runs requiring approval are only claimable when `approval_status=approved`.
   - Existing API/provider runs are unaffected by default (`requires_approval=False`).
5. Added service methods:
   - `approve_run_for_execution(...)`
   - `reject_run_for_execution(...)` (marks run cancelled + rejected)
6. Verification:
   - `RunServiceTests` and `RunWorkerTests` executed and passed.

## Stage 3: Provider Plumbing (`local_ollama`)

1. Add `local_ollama` as a first-class provider in forms/services.
2. Bypass user API key requirements for this provider.
3. Keep existing OpenAI/OpenRouter advanced path unchanged.
4. Exit criteria:
   - Run config can select local Ollama without credential validation errors.

### Stage 3 Implementation Notes (Completed 2026-05-06)

1. Added first-class provider support:
   - New provider enum value: `local_ollama`.
   - Default user/workspace provider switched to local route.
2. Local defaults wired:
   - Model default: `gemma4:27b`.
   - Base URL default: `LOCAL_OLLAMA_BASE_URL` (default `http://127.0.0.1:11434/v1`).
   - Reasoning set to `none` for local route.
3. Credential behavior split:
   - `local_ollama` no longer requires saved user/workspace API credentials.
   - OpenAI/OpenRouter credential flow remains unchanged for advanced users.
4. Concurrency clamp:
   - Local route settings normalize to single-worker semantics (`max_parallel_workers=1`).
5. UI/provider plumbing updates:
   - Wizard and global LLM config now include `Local (Ollama)` as default provider option.
   - Local route presents no API-key requirement in UI behavior.
6. Migration added:
   - `wb_workspaces/migrations/0009_alter_userllmcredential_provider_and_more.py`
7. Verification:
   - Targeted tests passed:
     - `RunServiceTests`
     - `RunWorkerTests`
     - `RunAdapterTests`
     - selected run-view local credential bypass tests
     - `InvestigationServiceTests`
     - selected workspace LLM-setting tests

## Stage 4: Approval Email + Approval Action

1. On queue, send approval email to `sam.osian@oreliandata.co.uk` with secure approve link.
2. Add approve endpoint/service to atomically move run to runnable state.
3. Add audit log events for requested/approved/rejected.
4. Exit criteria:
   - No run executes before approval.

### Stage 4 Implementation Notes (Completed 2026-05-06)

1. Automatic approval gate is now applied for local investigation runs:
   - `provider=local_ollama`
   - non-simulated execution
   - LLM run types (`filter`, `themes`, `extract`)
   - pipeline continuation runs (`pipeline_index > 0`) are not re-gated
2. Queue-time approval request email is now sent to:
   - `PFD_ADMIN_EMAIL` (default `sam.osian@oreliandata.co.uk`)
   - from `DEFAULT_FROM_EMAIL`
3. Approval email includes:
   - Django admin run change link (approve/schedule flow)
   - workspace run-log link
4. Audit and event logging added:
   - `run.approval_requested`
   - `run.approval_email_sent`
   - `run.approval_email_failed`
   - corresponding run events (`INFO`/`WARNING`)
5. Existing approval/reject service actions from Stage 2 remain the canonical atomic transition path:
   - `approve_run_for_execution(...)`
   - `reject_run_for_execution(...)`
6. Admin scheduling behavior preserved:
   - "Approve selected runs now" now uses each run's `queued_at` timestamp, allowing pre-scheduled execution.
7. Verification:
   - Targeted run service/worker/view and investigation readiness tests passed.

## Stage 5: Worker Gating + Single-Worker Semantics

1. Update claim logic so worker only claims approved runnable runs.
2. Enforce one worker process operationally.
3. Clamp `max_parallel_workers=1` for local route.
4. Exit criteria:
   - Queue respects approval gate and processes sequentially on DGX.

### Stage 5 Implementation Notes (Completed 2026-05-06)

1. Strict worker-lane isolation implemented:
   - `local` lane workers claim only runs with `provider=local_ollama`
   - `api` lane workers claim runs where provider is not `local_ollama`
   - `all` lane remains available for backward compatibility
2. Route isolation applied consistently to:
   - queued run claiming
   - cancelling-run reconciliation
   - timed-out run reconciliation
3. Worker runtime controls added:
   - New setting: `RUN_WORKER_ROUTE_MODE` (`all`, `api`, `local`)
   - `run_runs_worker` now supports `--route-mode` with same values
4. Queue config normalization:
   - `input_config_json.execution_route` is persisted at queue time (`local`/`api`) for traceability.
5. Approval gate and local single-worker semantics remain in effect from Stages 2-4.
6. Verification:
   - Added worker-route isolation tests (API worker skips local runs; local worker skips API runs; local worker claims local runs).
   - Targeted run service/worker/view tests passed.

### Stage 5 Operational Lane Config

1. Railway worker service (API route):
   - Run worker with `--route-mode api` (or set `RUN_WORKER_ROUTE_MODE=api`)
2. DGX worker service (local route):
   - Run worker with `--route-mode local` (or set `RUN_WORKER_ROUTE_MODE=local`)
3. Result:
   - API and local routes are isolated and independent while sharing existing queue infrastructure.

## Stage 6: UI Updates

1. Wizard/defaults:
   - "Use local DGX (recommended)" route.
   - No API key field required for that route.
2. Advanced section still exposes OpenAI/OpenRouter and key flow.
3. Show approval state in run timeline (`pending approval`, `approved`, and related states).
4. Exit criteria:
   - Users understand queue and approval lifecycle clearly.

## Stage 7: Deployment and Runtime Cutover

1. Railway:
   - Deploy web changes.
2. DGX:
   - Run worker process against Railway DB with production environment variables.
3. Ensure artifact backend is object storage so web can download DGX-generated outputs.
4. Exit criteria:
   - End-to-end production path works with DGX worker.

## Stage 8: QA, Guardrails, and Rollout

1. Add tests for:
   - Provider selection.
   - No-key local route.
   - Approval gate.
   - Claim gating.
2. Run smoke test matrix:
   - Queue, approve, execute, cancel, notifications.
3. Rollout with rollback instructions (including re-enabling prior path if needed).
4. Exit criteria:
   - Production-ready and documented.
