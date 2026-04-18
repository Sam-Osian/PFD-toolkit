# PFD Toolkit Workbench v0.2 - Starting Points and Rewrite Blueprint

## 1. Objective

Build a new Workbench codebase in parallel to `django_workbench/` (v0.1), with:

- real account creation and sign-in
- durable user workspaces
- safer and more maintainable backend architecture
- cleaner delivery path for the Claude design refresh

This v0.2 build is **non-destructive**. v0.1 stays online and untouched until final cutover.


## 2. Why Rewrite Instead of Iterating v0.1

v0.1 proved demand and core workflows, but the current architecture makes extension risky:

- one very large views module controls too many concerns
- large amounts of state are session-bound
- DataFrame payloads and temp-file storage are used as runtime state
- limited domain model persistence (workbook-focused, not account/workspace-native)
- AI workflow execution and UI flow are tightly coupled

For v0.2, we should optimize for correctness, observability, and explicit domain boundaries.


## 3. Non-Goals for Early v0.2

- No immediate full feature parity on day 1
- No immediate migration of all old templates/scripts
- No immediate destructive decommissioning of v0.1


## 4. Target Architecture (Recommended)

## 4.1 App Boundaries (Django)

Create multiple apps with clear responsibilities:

- `accounts`: auth, profile, user preferences
- `workspaces`: workspace CRUD, ownership, collaboration roles
- `datasets`: dataset references, report snapshots, metadata
- `investigations`: question/scope/method definitions, status lifecycle
- `runs`: async run tracking for filter/themes/extract operations
- `artifacts`: run outputs (tables, summaries, exports, previews)
- `sharing`: public share links, permissions, revocation
- `ui` (or `web`): page views/templates/API endpoints only

This keeps domain logic out of template/view monoliths.

## 4.2 Service Layer + Adapters

Use service modules per domain (application services), e.g.:

- `services/workspace_service.py`
- `services/investigation_service.py`
- `services/run_orchestrator.py`

Keep calls into `pfd_toolkit` behind adapter interfaces so UI code does not call toolkit objects directly.

## 4.3 Async Execution Model

Move long-running AI operations to worker processes:

- web process: enqueue, read status, render progress
- worker process: execute `filter_reports`, `discover_themes`, `extract_features`
- persistence: job state in Postgres + progress events in Redis/pubsub (or DB polling fallback)

This avoids request-thread blocking and reduces coupling to SSE internals.

## 4.4 State Strategy

Replace session-heavy state with database-backed state:

- session: only identity + minimal transient UI state
- DB: workspace configuration, selected filters, active dataset refs, run records, outputs
- object storage: large exports/artifacts (CSV/zip/json)


## 5. Data Model Foundation (v0.2 Phase 1)

Minimum core tables:

- `User` (Django auth)
- `Workspace`
  - `id`, `owner_id`, `name`, `description`, `created_at`, `updated_at`
- `WorkspaceMember`
  - `workspace_id`, `user_id`, `role` (`owner|editor|viewer`)
- `Investigation`
  - `workspace_id`, `title`, `question`, `scope_json`, `method_json`, `status`
- `InvestigationRun`
  - `investigation_id`, `run_type`, `status`, `started_at`, `finished_at`, `error`
- `RunArtifact`
  - `run_id`, `artifact_type`, `storage_uri`, `metadata_json`, `size_bytes`
- `WorkspaceShareLink`
  - `workspace_id`, `token`, `is_public`, `expires_at`, `revoked_at`

Optional early additions:

- `ApiCredential` (encrypted key storage if you decide to support saved keys)
- `AuditEvent` for critical changes


## 6. Security and Privacy Recommendations

- Never store plaintext API keys in session.
- If saved keys are needed, encrypt at rest (application-level encryption with key in env/secret manager).
- Keep CSRF/session security defaults strict in production.
- Add object-level permissions on all workspace/investigation endpoints.
- Add signed, revocable share links with expiry support.
- Keep network/invite-only gating out of hardcoded passwords; use feature flags + role checks.


## 7. Infrastructure Recommendations (Railway + Postgres)

## 7.1 Runtime Topology

Deploy as separate services:

- `web` (Django app)
- `worker` (async jobs)
- `scheduler` (if periodic tasks needed)
- `postgres`
- `redis` (queue + cache + progress pubsub)

## 7.2 Storage

- Postgres for relational state
- S3-compatible object storage for large artifacts
- avoid `/tmp` as authoritative storage

## 7.3 Observability

- Sentry for app and worker errors
- structured JSON logs with request/workspace/run IDs
- health endpoints for web and worker
- add metrics (queue depth, run duration, run failure rate)

## 7.4 Delivery Pipeline

- CI: lint + unit tests + migration checks
- staging environment before production
- release tags + rollback strategy


## 8. Frontend/UX Integration Strategy

Do not port the full Claude prototype immediately. Stage it:

1. Foundation UI shell with new auth + workspace navigation
2. Wizard shell wired to real backend models
3. One vertical slice working end-to-end (`Filter reports`)
4. Add `Themes` and `Extract` flows
5. Port advanced network UX after data contract stabilizes

The prototype includes account/workspace assumptions; backend should be made real first.


## 9. Suggested Delivery Phases

## Phase 0: Foundation Setup

- scaffold v0.2 project structure
- create core apps + settings split (`base/dev/prod`)
- configure Postgres + Redis + basic worker framework

## Phase 1: Auth + Workspace Core

- sign up / sign in / sign out
- workspace CRUD
- workspace membership + permissions
- basic dashboard listing user workspaces

## Phase 2: Investigation and Run Backbone

- investigation model + lifecycle
- queue async runs and track status
- persist artifacts and expose run history

## Phase 3: First AI Vertical Slice

- implement `filter_reports` end-to-end through new run system
- render progress and final artifact in UI
- add integration tests around run orchestration

## Phase 4: Themes + Extract + Sharing

- implement remaining workflows
- share links + clone/copy semantics
- export/download pipeline

## Phase 5: Design System Port + Hardening

- port Claude design token system and shells
- improve accessibility/performance
- load/perf tests and bug hardening

## Phase 6: Migration + Cutover

- migrate selected workbook/share data from v0.1 if needed
- controlled beta with real users
- switch traffic when stable
- retire v0.1 only after acceptance


## 10. Immediate Starting Tasks (Next 1-2 Weeks)

1. Create apps: `accounts`, `workspaces`, `investigations`, `runs`, `artifacts`, `sharing`.
2. Define and migrate foundational models (`Workspace`, `WorkspaceMember`, `Investigation`, `InvestigationRun`).
3. Implement auth pages and workspace list/create flow.
4. Set up worker stack and a test background job.
5. Add architecture tests for permission boundaries.
6. Add initial ADRs (architecture decision records) for:
   - queue framework choice
   - API key handling strategy
   - artifact storage strategy


## 11. Decision Checkpoints to Confirm Together

Please confirm these early so we do not build on shaky assumptions:

1. Saved API keys vs session-only BYOK (and encryption policy)
2. Queue stack choice (Celery/RQ/Dramatiq) for Railway
3. Whether team/shared workspaces are needed in v0.2 beta or post-beta
4. What v0.1 data must be migrated (if any) before launch
5. Which Claude prototype screens are required for first beta cut


## 12. v0.1 Safety Rule

`django_workbench/` remains intact until:

- v0.2 has auth + workspace + core AI workflow stability
- parity requirements are met for launch
- deployment rollback plan is tested

Only then should we plan decommissioning of v0.1.
