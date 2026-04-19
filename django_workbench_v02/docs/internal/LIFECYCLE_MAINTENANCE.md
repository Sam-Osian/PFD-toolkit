# Lifecycle Maintenance Jobs (v0.2)

Status: Implemented  
Last updated: 2026-04-19

## 1. Purpose

This job enforces inactivity-based keepalive/expiry policies using a sliding window.

Core policy:

1. Inactivity window defaults to 365 days.
2. Only likely human views keep resources alive.
3. Bot/crawler requests do not extend lifetime.

## 2. Human Activity Tracking

Human-only activity checks are centralized in:

- `wb_workspaces/activity.py`

Tracked on likely human views:

1. `Workspace.last_viewed_at`
2. `Investigation.last_viewed_at`
3. `WorkspaceShareLink.last_viewed_at`
4. `RunArtifact.last_viewed_at`

Current write paths:

1. `wb_workspaces.views.workspace_detail`
2. `wb_investigations.services.record_investigation_view`
3. `wb_sharing.services.record_share_link_view`
4. `wb_runs.services.record_run_view`
5. `wb_runs.services.record_artifact_download`

Optional debounce:

- `LIFECYCLE_VIEW_DEBOUNCE_SECONDS` (default `0`, disabled)

## 3. Maintenance Command

Command:

```bash
uv run python manage.py run_lifecycle_maintenance
```

Options:

1. `--inactivity-days <int>` override inactivity window.
2. `--dry-run` compute actions without DB writes.
3. `--skip-workspace-archive` run artifact checks only.

## 4. Enforcement Behavior

Implemented in `wb_workspaces/lifecycle.py`.

### 4.1 RunArtifact enforcement

Target rows:

- `RunArtifact(status="ready")`

Rules:

1. Compute `effective_expires_at = (last_viewed_at or created_at) + inactivity_window`.
2. If now >= `effective_expires_at`, mark artifact `status="expired"`.
3. Otherwise refresh `expires_at` to `effective_expires_at`.

Audit event on expiry:

- `run.artifact_expired_inactive`

### 4.2 Workspace enforcement

Target rows:

- `Workspace(archived_at is null)`

Rules:

1. Compute `last_activity_at = last_viewed_at or created_at`.
2. If inactivity exceeds window, set `archived_at=now`.

Audit event on archive:

- `workspace.auto_archived_inactive`

## 5. Settings

Defined in `pfd_workbench_v02/settings.py`:

1. `LIFECYCLE_INACTIVITY_DAYS` (default `365`)
2. `LIFECYCLE_VIEW_DEBOUNCE_SECONDS` (default `0`)

## 6. Railway Scheduler Shape

Recommended additional service/job:

1. Scheduler/Cron job command:
   `uv run python manage.py run_lifecycle_maintenance`
2. Suggested cadence: daily.
3. Use `--dry-run` first in staging to validate counts before enabling write mode.
