# Investigation Pipeline User-Readiness Slices (v0.2)

Status: Active execution plan  
Last updated: 2026-04-22

## 1. Purpose

Break investigation launch readiness into clear implementation slices so we can ship in controlled increments and verify behavior at each step.

This document reflects current code assessment across:

1. `django_workbench_v02/` (v0.2 UI + services + worker)
2. `django_workbench/` (v0.1 baseline behavior)
3. `src/` (`pfd_toolkit` execution package)

## 2. Confirmed Product Intent (aligned)

Per product clarification:

1. An investigation **creates a new workspace**.
2. Investigation title/description become workspace title/description.
3. The investigation flow is the process for creating and launching that new workspace pipeline.

Implication:

1. Earlier concern about wizard posting to `investigation-start` is **not a blocker** for this product contract.

## 3. Current Snapshot (high level)

What is already strong:

1. Real worker-backed async execution is implemented.
2. Wizard-to-run-config mapping exists for filter/themes/extract.
3. Continue-on-fail chaining exists, with explicit upstream-artifact error code support.
4. Credential encryption-at-rest + runtime resolution exists.
5. S3-compatible artifact backend exists (R2-compatible via config).

What still blocks user-ready behavior:

1. Notification semantics are run-level, not pipeline-level.
2. Workspace card/detail UX does not yet match the Claude design target behavior.
3. Production readiness still depends on deploy/runtime confirmation (worker + dispatcher + object storage env + SMTP).

Accepted risk for now:

1. Lifecycle auto-archive/auto-purge policy is intentional.

## 4. Slice Plan

## Slice 0: Contract Lock (Intent + Terminology)

Goal:

1. Lock the canonical contract so implementation and QA use one definition.

Scope:

1. Confirm and document: investigation creates workspace.
2. Confirm title/description carry-over rules.
3. Confirm pipeline-stage ordering contract and state labels.

Done when:

1. This contract is documented and referenced by implementation/QA docs.
2. Source of truth is `docs/internal/INVESTIGATION_WORKSPACE_CONTRACT.md`.

Status:

1. Completed on 2026-04-22 (documentation contract lock).

---

## Slice 1: Launch Payload Integrity

Goal:

1. Ensure user wizard inputs are faithfully represented in run config and persisted investigation state.

Current:

1. `launch_investigation_wizard_pipeline` maps scope/method/filter/themes/extract/concurrency into run config.

Gaps to verify/close:

1. No silent/default behavior that changes user intent in edge cases.
2. Confirm all optional controls map exactly as intended by product copy.

Done when:

1. Input-output contract table exists and is tested for every wizard option.

Slice artifact:

1. `docs/internal/INVESTIGATION_WIZARD_LAUNCH_MAPPING_CONTRACT.md`

Primary code:

1. `wb_investigations/services.py`
2. `wb_investigations/forms.py`
3. `wb_runs/scope.py`

---

## Slice 2: Worker Handoff + End-to-End Pipeline Continuity

Goal:

1. Guarantee queued pipelines complete server-side independent of browser session.

Current:

1. Worker loop claims queued runs and executes real adapters.
2. Pipeline continuation queues next stages with upstream artifact chaining.

Gaps to verify/close:

1. Deployment/runtime process checks must be treated as release gate (worker always running).
2. Failure modes (timeouts, transient retries, cancellation) need explicit QA matrix signoff.

Done when:

1. Launching then closing browser still yields terminal state and artifacts.
2. Cancellation/timeout/retry behaviors are validated in staging.

Status:

1. Implementation completed on 2026-04-22 (worker heartbeat + healthcheck gate + continuity QA matrix).

Slice artifact:

1. `docs/internal/INVESTIGATION_PIPELINE_CONTINUITY_QA.md`

Primary code:

1. `wb_runs/worker.py`
2. `wb_runs/management/commands/run_runs_worker.py`
3. `docs/internal/RUN_WORKER.md`

---

## Slice 3: Pipeline-Level Notification Semantics

Goal:

1. Align email notification with full pipeline completion semantics.

Current:

1. Notification request is attached to queued run and dispatches on that run’s terminal status.

Gap:

1. For multi-stage pipelines, email may trigger after stage 1 terminal rather than pipeline terminal.

Done when:

1. Notification sends only when final pipeline outcome is reached (or intentionally configured otherwise).
2. Review copy matches actual semantics exactly.

Primary code:

1. `wb_investigations/views.py`
2. `wb_notifications/services.py`
3. `docs/internal/NOTIFICATIONS.md`

---

## Slice 4: Workspace Card UX (Claude Design Parity)

Goal:

1. Make workspace status card behavior match design intent: strong visual state, loading treatment, clear metrics.

Current:

1. Status is row/tag-based and functional but not design-parity.

Gaps:

1. Card layout/style parity with mock.
2. Loading/disabled visual state polish during active pipeline.
3. Summary metrics (reports/themes/extractions) presentation parity.

Done when:

1. Dashboard renders workspace cards matching agreed visual spec.
2. Active pipeline state is visually unmistakable.

Primary code:

1. `templates/wb_workspaces/dashboard.html`
2. `wb_workspaces/views.py`
3. `static/css/claude_app.css`

---

## Slice 5: Workspace Detail / Investigation Results Experience

Goal:

1. Clicking a workspace should open a result experience analogous to Explore/Collections expectations, with carried title + description.

Current:

1. Detail view is operational but utilitarian/admin-heavy.

Gaps:

1. UX needs result-first dashboard experience.
2. Investigation metadata prominence (title/description) needs product-grade placement.
3. Artifact/result discovery flow needs simplification for non-technical users.

Done when:

1. Workspace open action leads to user-facing results dashboard, not admin-style scaffolding.

Primary code:

1. `templates/wb_workspaces/workspace_detail.html`
2. `templates/wb_investigations/investigation_detail.html`
3. `templates/wb_runs/run_detail.html`

---

## Slice 6: Data Shape Guarantees for Themes + Extract

Goal:

1. Guarantee output tables contain expected columns and semantics for downstream use.

Current:

1. Themes produce `theme_assignments.csv` and `theme_summary.csv`.
2. Extract produces `extraction_table.csv` with requested fields.

Gaps to verify/close:

1. True/False representation consistency for theme assignment columns in downstream UI.
2. Merge/display behavior in user-facing tables for extracted/theme fields.
3. Contract tests against realistic datasets.

Done when:

1. Acceptance tests assert output column shapes and value semantics.

Primary code:

1. `wb_runs/pfd_toolkit_adapter.py`
2. `src/pfd_toolkit/extractor.py`
3. `src/pfd_toolkit/screener.py`

---

## Slice 7: Artifact Storage + Deploy Runtime Hardening

Goal:

1. Prove production artifact and notification infrastructure is correctly configured and reliable.

Current:

1. Object storage backend exists and is production-enforced by config.
2. Deploy runbook includes web/worker/notification-dispatcher services.

Gaps:

1. Need explicit environment confirmation in deploy target:
1. `ARTIFACT_STORAGE_BACKEND=object_storage`
2. object storage bucket/endpoint/credentials configured
3. `CREDENTIAL_ENCRYPTION_KEY` set on all relevant services
4. SMTP settings validated for real email delivery

Done when:

1. Staging/prod smoke test passes: real run, artifact download, completion email.

Primary code/docs:

1. `wb_runs/artifact_storage.py`
2. `pfd_workbench_v02/settings.py`
3. `docs/internal/DEPLOY_RUNBOOK.md`

---

## Slice 8: Access + Durability Policy Finalization

Goal:

1. Finalize policy and controls for workspace ownership, delete/archive rights, and lifecycle behavior.

Current:

1. Ownership/membership model is strong and linked to users.
2. Admin hard delete exists.
3. Auto-archive/purge exists and is currently accepted.

Near-term optional:

1. Allow user self-delete/archive UX path (if desired this phase).

Done when:

1. Policy is explicit and reflected in UI affordances and docs.

Primary code:

1. `wb_workspaces/models.py`
2. `wb_workspaces/services.py`
3. `wb_workspaces/lifecycle.py`

## 5. Recommended Execution Order

1. Slice 0 (contract lock)
2. Slice 3 (pipeline-level notifications)
3. Slice 4 (workspace card design/status parity)
4. Slice 5 (workspace detail results UX)
5. Slice 6 (data-shape guarantees + tests)
6. Slice 7 (deploy/runtime hardening + smoke proof)
7. Slice 8 (policy/UI completion)

## 6. Release Gate (user-ready definition)

“User-ready” for pipeline launch means all are true:

1. Investigation launch creates workspace and queues real worker pipeline.
2. User can close browser without interrupting pipeline.
3. Workspace status card accurately reflects pending/running/terminal states in intended design style.
4. Output artifacts include expected themes/extract columns and are accessible in product UI.
5. Completion email semantics are pipeline-accurate.
6. Credential handling remains encrypted, scoped, and operational in production.
7. Object storage and dispatcher/worker services are verified live in deploy environment.
