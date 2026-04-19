# PFD Toolkit Workbench v0.2 - Rewrite Blueprint and Live Status

Last updated: 2026-04-19

## 0. Current Status Snapshot

This file started as a "starting points" brief. It now also tracks delivery status.

### 0.1 Phase Status

1. Phase 0 (Foundation Setup): Completed
2. Phase 1 (Auth + Workspace Core): Completed
3. Phase 2 (Investigation + Run Backbone): Completed
4. Phase 3 (First AI Vertical Slice): Completed at backend level
5. Phase 4 (Themes + Extract + Sharing): In progress (lifecycle/scheduler hardening pending)
6. Phase 5 (Design System Port + Hardening): Not started
7. Phase 6 (Migration + Cutover): Not started

### 0.2 What Is Implemented

1. New non-destructive v0.2 codebase in `django_workbench_v02/` (v0.1 still intact).
2. Custom user model with first and last name.
3. Auth0 login flow and local user linking.
4. Admin path and superuser tooling.
5. Workspace + membership model with per-user access mode (`edit`/`read_only`).
6. Team/shared workspace support in beta.
7. Investigation and run domain models with async worker lifecycle statuses.
8. Run cancellation support.
9. Audit events across key actions.
10. Real `pfd_toolkit` adapters for `filter`, `themes`, `extract`, and `export`.
11. Share links with public/private behavior and expiry/revocation.
12. Internal architecture docs:
   - `docs/internal/DATA_MODEL.md`
   - `docs/internal/DATA_MODEL_CONSTRAINTS.md`
   - `docs/internal/AUTH_PERMISSION_SPINE.md`
   - `docs/internal/RUN_WORKER.md`
   - `docs/internal/PFD_TOOLKIT_ADAPTERS.md`
   - `docs/internal/ARTIFACT_STORAGE.md`

### 0.3 Major Open Items For Phase 4 Close

1. Artifact delivery/storage strategy for production:
   - object storage backend implemented; rollout validation on Railway still needed
   - stable download endpoints now support file and object-storage artifacts
2. Complete "human-view-only" keepalive/expiry lifecycle logic for artifacts/workspaces (baseline now implemented for artifact downloads).
3. Optional completion notifications (email path later, per your plan).

## 1. Objective

Build a new Workbench codebase in parallel to `django_workbench/` (v0.1), with:

1. real account creation and sign-in
2. durable user workspaces
3. safer and more maintainable backend architecture
4. cleaner delivery path for the Claude design refresh

This v0.2 build is non-destructive. v0.1 stays online and untouched until final cutover.

## 2. Why Rewrite Instead of Iterating v0.1

v0.1 proved demand and core workflows, but architecture constraints made extension risky:

1. one very large views module controlling too many concerns
2. session-heavy runtime state
3. DataFrame/temp-file state leakage across concerns
4. limited durable domain modeling
5. tight coupling between AI execution and UI flow

v0.2 optimizes for correctness, observability, and explicit domain boundaries.

## 3. Early Non-Goals (Still Valid)

1. No instant full parity on day 1.
2. No destructive decommissioning of v0.1 before acceptance.
3. No blind copy-paste of old UI logic into new architecture.

## 4. Target Architecture

## 4.1 App Boundaries (Implemented Pattern)

Current app split (with `wb_` prefixes) follows the intended boundaries:

1. `accounts`: authentication and user account model
2. `wb_workspaces`: workspaces, memberships, permissions, revision records
3. `wb_investigations`: investigation definitions and lifecycle
4. `wb_runs`: runs, events, artifacts, async worker
5. `wb_sharing`: share links and public/private access behavior
6. `wb_auditlog`: audit trail
7. `wb_notifications`: notification request scaffolding

## 4.2 Service Layer + Adapters

Pattern in use:

1. Domain services for permissions and lifecycle invariants.
2. Worker orchestration in `wb_runs/worker.py`.
3. Toolkit integration isolated in `wb_runs/pfd_toolkit_adapter.py`.

## 4.3 Async Execution Model

Current model:

1. Web process enqueues run records in Postgres.
2. Worker command claims and executes runs server-side.
3. Progress and status are persisted in DB (`RunEvent` + run status fields).
4. Cancellation is cooperative and persisted.

Note: current queue behavior is DB-backed polling. Redis-backed queueing remains an optional upgrade.

## 4.4 State Strategy

Current model:

1. Session state minimized.
2. Domain state persisted in Postgres.
3. Artifacts currently written to local filesystem paths under runtime artifact directories.

Phase 4/5 upgrade target:

1. object storage-backed artifacts for production durability.

## 5. Data Model Foundation

Core persisted entities are now present:

1. `User` (custom auth user)
2. `Workspace`
3. `WorkspaceMembership`
4. `WorkspaceRevision`
5. `Investigation`
6. `InvestigationRun`
7. `RunEvent`
8. `RunArtifact`
9. `WorkspaceShareLink`
10. `AuditEvent`

See internal docs for full schema rationale and constraints.

## 6. Security and Privacy Recommendations

Confirmed direction for beta:

1. Do not store user API keys by default.
2. Snapshot sharing as default mode.
3. Public links may be viewable without login when explicitly configured.
4. Keep strict object-level permission checks across workspace-bound resources.
5. Keep audit logging on critical state transitions.

Future if live/shared-key mode is enabled:

1. encrypted credential storage
2. key-rotation policy
3. breach response playbook

## 7. Infrastructure Recommendations (Railway + Postgres)

Current:

1. Web and worker split pattern is implemented at app level.
2. Postgres-backed persistence is in place.

Recommended next infra hardening:

1. add Redis only when needed for queue semantics or push progress
2. move artifact binaries to object storage
3. add Sentry and metrics dashboards
4. codify CI gates (tests + migration check + lint)

## 8. Frontend/UX Integration Strategy

Still valid:

1. Keep backend contracts stable first.
2. Port Claude design shell in slices.
3. Avoid introducing UX polish that hides backend uncertainty.

Suggested order:

1. workspace/investigation shell
2. run detail + artifact views
3. export/download UX
4. broader design-system migration

## 9. Delivery Phases (Live Tracker)

## Phase 0: Foundation Setup

Status: Completed

Delivered:

1. v0.2 project scaffold
2. app structure and base settings
3. initial migration baseline

## Phase 1: Auth + Workspace Core

Status: Completed

Delivered:

1. Auth0-backed login flow
2. custom user model (email + first/last name)
3. workspace CRUD baseline
4. workspace membership and permission controls

## Phase 2: Investigation and Run Backbone

Status: Completed

Delivered:

1. investigation domain lifecycle
2. run queue/status model
3. run events and artifact persistence

## Phase 3: First AI Vertical Slice

Status: Completed at backend level

Delivered:

1. real filter workflow adapter integration
2. async worker execution path
3. tests for run orchestration and permissions

## Phase 4: Themes + Extract + Sharing

Status: In progress

Delivered:

1. real themes workflow adapter path
2. real extract workflow adapter path
3. real export workflow adapter path (zip bundle + manifest)
4. share link lifecycle (create/update/revoke/public/private)
5. artifact download endpoint with permission checks + download audit events
6. artifact storage abstraction with local and S3-compatible object storage backends

Remaining:

1. wider keepalive/expiry policy enforcement job (scheduler path)

## Phase 5: Design System Port + Hardening

Status: Not started

Planned:

1. Claude design token/system port in slices
2. accessibility and performance hardening
3. end-to-end bug bash

## Phase 6: Migration + Cutover

Status: Not started

Planned:

1. decide v0.1 data migration scope
2. staged beta cutover
3. rollback-tested production switch
4. retire v0.1 only after acceptance

## 10. Next 1-2 Week Plan (Updated)

1. Implement wider human-view keepalive/expiry enforcement jobs (scheduler path).
2. Add ADRs for:
   - export architecture
   - artifact storage strategy
   - run retry policy

## 11. Confirmed Decision Log (Current)

1. Team/shared workspaces: Yes (beta).
2. Workspace access model: per user-workspace membership; read-only is membership-level.
3. Public sharing: allowed without login when explicitly public.
4. Default sharing mode: snapshot.
5. Live mode: technically possible later; keep API-key storage out of beta by default.
6. Run cancellation: required and implemented.
7. Audit trail: required and implemented.
8. Artifact/activity expiry intent: long window with activity-based keepalive, human-view-only.

## 12. v0.1 Safety Rule

`django_workbench/` remains intact until:

1. v0.2 has stable auth + workspace + core workflows
2. parity requirements for beta are accepted
3. deployment rollback has been tested

Only then should v0.1 decommissioning be planned.
