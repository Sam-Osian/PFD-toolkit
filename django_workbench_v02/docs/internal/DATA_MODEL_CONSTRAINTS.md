# PFD Toolkit Workbench v0.2 Data Model Constraints

Status: Approved baseline
Last updated: 2026-04-18
Companion: `docs/internal/DATA_MODEL.md`

## 1. Purpose

This document converts the high-level data model into concrete implementation rules:

- nullability and defaults
- enum/domain contracts
- foreign-key delete policies
- unique/check constraints
- index plan
- revision behavior
- inactivity/expiry behavior (human-only activity)

## 2. Global Conventions

1. Use UUID primary keys for externally-referenced entities.
2. Use timezone-aware UTC datetimes.
3. Use strict enums via `TextChoices` (no free-text statuses).
4. Prefer `NOT NULL` for core fields unless optionality is intentional.
5. JSON fields are only for flexible payload/config, not core relational identity.

## 3. Table Contracts

## 3.1 Workspace

Required:

- `id` UUID PK
- `created_by` FK User (`on_delete=PROTECT`)
- `title` non-null
- `slug` non-null
- `visibility` enum (`private`, `public`) non-null
- `is_listed` bool default `False`
- `created_at`, `updated_at` non-null

Optional:

- `description`
- `last_viewed_at`
- `archived_at`

Constraints:

- `UniqueConstraint(fields=["created_by", "slug"])`

Indexes:

- `Index(fields=["visibility", "is_listed"])`
- `Index(fields=["updated_at"])`

## 3.2 WorkspaceMembership

Required:

- `workspace` FK Workspace (`on_delete=CASCADE`)
- `user` FK User (`on_delete=CASCADE`)
- `role` enum (`owner`, `editor`, `viewer`)
- `access_mode` enum (`edit`, `read_only`)
- `can_manage_members` bool default `False`
- `can_manage_shares` bool default `False`
- `can_run_workflows` bool default `True`
- timestamps

Constraints:

- `UniqueConstraint(fields=["workspace", "user"])`

Business invariants (service layer):

- At least one owner per workspace.
- Only owners can toggle membership access and role.

Indexes:

- `Index(fields=["workspace", "role"])`
- `Index(fields=["user"])`

## 3.3 WorkspaceRevision

Required:

- `workspace` FK Workspace (`on_delete=CASCADE`)
- `revision_number` int > 0
- `state_json` JSON non-null
- `change_type` enum (`edit`, `undo`, `redo`, `restore`, `system`)
- `created_at`

Optional:

- `created_by` FK User (`on_delete=SET_NULL`)
- `parent_revision` self FK (`on_delete=SET_NULL`)

Constraints:

- `UniqueConstraint(fields=["workspace", "revision_number"])`
- `CheckConstraint(revision_number__gt=0)`

Indexes:

- `Index(fields=["workspace", "-revision_number"])`
- `Index(fields=["created_at"])`

## 3.4 Investigation

Required:

- `workspace` FK Workspace (`on_delete=CASCADE`)
- `created_by` FK User (`on_delete=PROTECT`)
- `title` non-null
- `scope_json` JSON non-null default `{}`
- `method_json` JSON non-null default `{}`
- `status` enum (`draft`, `active`, `archived`)
- timestamps

Optional:

- `question_text`
- `last_viewed_at`

Indexes:

- `Index(fields=["workspace", "status"])`
- `Index(fields=["updated_at"])`

## 3.5 InvestigationRun

Required:

- `investigation` FK Investigation (`on_delete=CASCADE`)
- `workspace` FK Workspace (`on_delete=CASCADE`)
- `requested_by` FK User (`on_delete=PROTECT`)
- `run_type` enum (`filter`, `themes`, `extract`, `export`)
- `status` enum (`queued`, `starting`, `running`, `cancelling`, `cancelled`, `succeeded`, `failed`, `timed_out`)
- `queued_at`, `created_at`, `updated_at`
- `input_config_json` JSON non-null default `{}`

Optional:

- `progress_percent` int
- `started_at`, `finished_at`
- `cancel_requested_at`
- `cancel_requested_by` FK User (`on_delete=SET_NULL`)
- `cancel_reason`
- `worker_id`, `error_code`, `error_message`
- `query_start_date`, `query_end_date`

Constraints:

- `CheckConstraint(progress_percent__gte=0 & progress_percent__lte=100)` when not null
- `CheckConstraint(query_end_date__gte=query_start_date)` when both present

Indexes:

- `Index(fields=["workspace", "status", "queued_at"])`
- `Index(fields=["investigation", "-created_at"])`
- `Index(fields=["status", "updated_at"])`

## 3.6 RunEvent

Required:

- `run` FK InvestigationRun (`on_delete=CASCADE`)
- `event_type` enum (`stage`, `progress`, `warning`, `error`, `info`, `cancel_check`)
- `message` non-null
- `payload_json` JSON non-null default `{}`
- `created_at`

Indexes:

- `Index(fields=["run", "created_at"])`

## 3.7 RunArtifact

Required:

- `run` FK InvestigationRun (`on_delete=CASCADE`)
- `workspace` FK Workspace (`on_delete=CASCADE`)
- `artifact_type` enum (`filtered_dataset`, `theme_summary`, `theme_assignments`, `extraction_table`, `bundle_export`, `preview`)
- `status` enum (`pending`, `building`, `ready`, `failed`, `expired`)
- `storage_backend` enum (`db`, `object_storage`, `file`)
- `metadata_json` JSON non-null default `{}`
- timestamps

Optional:

- `storage_uri`
- `content_hash`
- `size_bytes`
- `expires_at`
- `last_viewed_at`

Constraints:

- `CheckConstraint(size_bytes__gte=0)` when not null

Indexes:

- `Index(fields=["workspace", "status", "expires_at"])`
- `Index(fields=["run", "artifact_type"])`
- `Index(fields=["last_viewed_at"])`

## 3.8 WorkspaceShareLink

Required:

- `id` UUID token PK
- `workspace` FK Workspace (`on_delete=CASCADE`)
- `created_by` FK User (`on_delete=PROTECT`)
- `mode` enum (`snapshot`, `live`) default `snapshot`
- `is_public` bool default `False`
- `is_active` bool default `True`
- timestamps

Optional:

- `snapshot_revision` FK WorkspaceRevision (`on_delete=SET_NULL`)
- `expires_at`
- `last_viewed_at`

Indexes:

- `Index(fields=["workspace", "is_active"])`
- `Index(fields=["is_public", "is_active"])`
- `Index(fields=["last_viewed_at"])`

## 3.9 AuditEvent

Required:

- `action_type` non-null
- `target_type` non-null
- `target_id` non-null
- `payload_json` JSON non-null default `{}`
- `created_at`

Optional:

- `workspace` FK Workspace (`on_delete=SET_NULL`)
- `user` FK User (`on_delete=SET_NULL`) for guest/system events
- `ip_hash`
- `user_agent`

Indexes:

- `Index(fields=["workspace", "-created_at"])`
- `Index(fields=["user", "-created_at"])`
- `Index(fields=["action_type", "-created_at"])`

## 3.10 NotificationRequest

Required:

- `run` FK InvestigationRun (`on_delete=CASCADE`)
- `user` FK User (`on_delete=CASCADE`)
- `channel` enum (`email`)
- `notify_on` enum (`success`, `failure`, `any`)
- `status` enum (`pending`, `sent`, `failed`, `cancelled`)
- timestamps

Optional:

- `sent_at`
- `error_message`

Indexes:

- `Index(fields=["status", "created_at"])`
- `Index(fields=["user", "status"])`
- `Index(fields=["run"])`

## 4. Foreign Key Delete Policy Summary

Principles:

1. Prevent accidental deletion of core owners (`PROTECT`) where warranted.
2. Cascade purely dependent technical records (`run_events`, `run_artifacts`).
3. Preserve log history attribution where practical (`SET_NULL` on optional actors).

Recommended map:

- `Workspace.created_by -> User`: `PROTECT`
- `WorkspaceMembership.workspace -> Workspace`: `CASCADE`
- `WorkspaceMembership.user -> User`: `CASCADE`
- `WorkspaceRevision.workspace -> Workspace`: `CASCADE`
- `WorkspaceRevision.created_by -> User`: `SET_NULL`
- `Investigation.workspace -> Workspace`: `CASCADE`
- `Investigation.created_by -> User`: `PROTECT`
- `InvestigationRun.investigation -> Investigation`: `CASCADE`
- `InvestigationRun.requested_by -> User`: `PROTECT`
- `InvestigationRun.cancel_requested_by -> User`: `SET_NULL`
- `RunEvent.run -> InvestigationRun`: `CASCADE`
- `RunArtifact.run -> InvestigationRun`: `CASCADE`
- `WorkspaceShareLink.workspace -> Workspace`: `CASCADE`
- `WorkspaceShareLink.created_by -> User`: `PROTECT`
- `WorkspaceShareLink.snapshot_revision -> WorkspaceRevision`: `SET_NULL`
- `AuditEvent.workspace -> Workspace`: `SET_NULL`
- `AuditEvent.user -> User`: `SET_NULL`
- `NotificationRequest.run -> InvestigationRun`: `CASCADE`
- `NotificationRequest.user -> User`: `CASCADE`

## 5. Revision Semantics (Undo/Redo)

Approved behavior:

1. Revisions are immutable.
2. Undo restores a previous revision state and writes a new revision (`change_type=undo`).
3. Redo restores a later revision state and writes a new revision (`change_type=redo`).
4. No in-place mutation of historical revisions.
5. Final “current state” is the most recent revision for workspace scope.

## 6. Run Status Lifecycle + Cancellation

Valid statuses:

- `queued`, `starting`, `running`, `cancelling`, `cancelled`, `succeeded`, `failed`, `timed_out`

Lifecycle rules:

1. Terminal statuses: `cancelled`, `succeeded`, `failed`, `timed_out`.
2. Cancellation request sets `cancel_requested_at` and transitions to `cancelling`.
3. Worker checks cancellation and exits cleanly to `cancelled` where possible.
4. Every status change writes `RunEvent` + `AuditEvent`.

## 7. Activity and Expiration (Human-Only)

Approved policy:

1. Sliding inactivity window: 365 days.
2. Activity = human view access to workspace/share/artifact URLs.
3. Bot/crawler traffic must NOT extend lifetime.

Implementation guidance:

1. Maintain `last_viewed_at` on Workspace, WorkspaceShareLink, and RunArtifact.
2. Use bot filtering by user-agent + optional reverse-DNS/known crawler list.
3. Debounce writes (e.g., no more than once per 12-24h per viewer/resource).
4. Scheduled task marks stale artifacts `expired` when inactivity exceeds 365 days.

## 8. Migrations and Rollout Advice

1. Create enums/choices up front to avoid string drift.
2. Add indexes in initial migrations for known hot paths.
3. Use data migrations only when introducing non-null fields to existing tables.
4. Keep domain invariants in services (plus DB constraints where enforceable).
