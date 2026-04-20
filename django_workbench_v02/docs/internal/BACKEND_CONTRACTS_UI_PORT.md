# Backend Contracts for UI Port (v0.2)

Status: Locked for Claude-design UI port  
Last updated: 2026-04-20

## 1. Purpose

This document freezes the backend behavior contracts that the UI will consume during the design port.

Rule:

1. UI refactors should consume these contracts.
2. Backend contract changes require explicit doc + test updates in the same PR.

## 2. Contract Scope

Locked in this file:

1. Investigation wizard flow and launch payload behavior.
2. Run detail page data contract.
3. Workbook dashboard/list and investigation entry behavior.
4. Completion notification request/dispatch contract.

Out of scope (can change freely in UI work):

1. Layout, typography, color, animation, component styling.
2. Copy polish, as long as meaning and validation behavior stay intact.

## 3. Investigation Wizard Contract

Primary route:

1. `GET/POST /workbooks/<workbook_id>/investigations/<investigation_id>/wizard/`

Permission gate:

1. Requires `can_edit_workspace`.

Source files:

1. `wb_investigations/views.py`
2. `wb_investigations/forms.py`
3. `templates/wb_investigations/investigation_wizard.html`

### 3.1 Stage Model

Canonical stage ids:

1. `question`
2. `scope`
3. `method`
4. `themes` (conditional)
5. `extract` (conditional)
6. `review`

Sequence rule:

1. Fixed order.
2. `themes` appears only when `run_themes=true`.
3. `extract` appears only when `run_extract=true`.

### 3.2 Stage Inputs and Validation

`question`:

1. `title` (required)
2. `question_text` (required, non-empty)

`scope`:

1. `scope_option` in:
   - `all_reports`
   - `last_3_years`
   - `last_year`
   - `last_6_months`
   - `most_recent_100`

`method`:

1. `run_filter` (bool, default true)
2. `run_themes` (bool, default false)
3. `run_extract` (bool, default false)
4. At least one of filter/themes/extract must be true.

`themes`:

1. `seed_topics`
2. `min_themes`
3. `max_themes`
4. `extra_theme_instructions`
5. Validation: if both min/max exist, `min_themes <= max_themes`.

`extract`:

1. `feature_fields` required when extract enabled.
2. Each field row must contain:
   - `name`
   - `description`
   - `type`
3. Supports line format `name | description | type` and legacy compatibility parsing.
4. Additional flags:
   - `allow_multiple`
   - `force_assign`
   - `skip_if_present`

`review`:

1. `execution_mode` (`real` or `simulate`)
2. `provider` (`openai` or `openrouter`)
3. `model_name`
4. `request_completion_email` (bool)
5. `notify_on` (`success`, `failure`, `any`)

### 3.3 Launch Semantics

Launch action:

1. Triggered from `review` with `wizard_action=launch`.
2. Updates investigation `title`, `question_text`, `scope_json`, `method_json`.
3. Queues first run from pipeline plan index `0`.
4. Pipeline is chained via run config:
   - `pipeline_plan`
   - `pipeline_index`
   - `pipeline_continue_on_fail=true`

Temporal scope mapping:

1. `all_reports` -> no date bounds, no report limit.
2. `last_3_years` -> bounded dates.
3. `last_year` -> bounded dates.
4. `last_6_months` -> bounded dates.
5. `most_recent_100` -> `report_limit=100`.

Credential rule:

1. For `execution_mode=real`, launch requires saved workbook credential for selected provider.

Session state:

1. Stored under `investigation_wizard_state:<investigation_id>`.
2. Cleared on successful launch or explicit reset.

## 4. Run Detail Contract

Primary route:

1. `GET /workbooks/<workbook_id>/runs/<run_id>/`

Permission gate:

1. Requires `can_view_workspace`.

Source files:

1. `wb_runs/views.py`
2. `templates/wb_runs/run_detail.html`

### 4.1 Required Render Context Keys

Page expects:

1. `run`
2. `investigation`
3. `workspace`
4. `run_journey`
5. `outcome`
6. `events`
7. `artifacts`
8. `artifact_groups`
9. `config_summary_rows`
10. `raw_config_json`
11. `can_request_cancellation`
12. `cancel_form`
13. `cancellation_message`
14. `notifications`

### 4.2 `run_journey` Entry Shape

Each entry:

1. `label`
2. `state` (`done`, `current`, `pending`, `skipped`)
3. `state_label`
4. `timestamp` (nullable)
5. `note`

### 4.3 Cancellation Contract

Cancel route:

1. `POST /workbooks/<workbook_id>/runs/<run_id>/cancel/`

Rules:

1. Terminal runs cannot be cancelled.
2. Cancellation request sets status to `cancelling`.
3. Worker finalizes to `cancelled`.

### 4.4 Artifact Group Contract

Grouping keys:

1. `artifact_type`
2. `label`
3. `intent`
4. `entries[]` where each entry includes:
   - `artifact`
   - `downloadable`
   - `size_display`

Download route:

1. `GET /workbooks/<workbook_id>/runs/<run_id>/artifacts/<artifact_id>/download/`

## 5. Workbook Dashboard and Investigation Entry Contract

Primary routes:

1. `GET/POST /workbooks/` (dashboard)
2. `GET /workbooks/<workbook_id>/investigation/` (entry route)

Source files:

1. `wb_workspaces/views.py`
2. `wb_investigations/views.py`
3. `templates/wb_workspaces/dashboard.html`

### 5.1 Dashboard Context Contract

Dashboard expects:

1. `memberships`
2. `active_rows`
3. `archived_rows`
4. `active_workspace_id`
5. `form` (create workbook form)

Row shape:

1. `membership`
2. `workspace`
3. `is_archived`
4. `can_restore`

### 5.2 Investigation Entry Behavior

Rules:

1. Each workbook has at most one investigation.
2. If none exists and user can edit, default investigation is auto-created.
3. If user can edit, entry redirects to wizard.
4. If user is read-only, entry redirects to investigation detail.

This is the canonical behavior for "Open investigation" from dashboard/workbook views.

## 6. Completion Notification Contract

Source files:

1. `wb_notifications/models.py`
2. `wb_notifications/services.py`
3. `docs/internal/NOTIFICATIONS.md`

### 6.1 Request Creation

Notification request can be created by:

1. Wizard launch (review step).
2. Export queue form.
3. Direct run queue form.

Allowed values:

1. `channel`: `email` only.
2. `notify_on`: `success`, `failure`, `any`.

### 6.2 Dispatch Contract

Dispatcher only processes pending requests for terminal runs:

1. `succeeded`
2. `failed`
3. `timed_out`
4. `cancelled`

Status transitions:

1. `pending -> sent`
2. `pending -> failed`
3. `pending -> cancelled` (trigger mismatch)

### 6.3 Email Template Data Contract

Templates consume:

1. `subject`
2. `investigation_title`
3. `run_id`
4. `run_type`
5. `run_status`
6. `queued_at`
7. `started_at`
8. `finished_at`
9. `run_detail_url`
10. `notification_preferences_url`
11. `workbench_base_url`
12. status presentation fields:
    - `subject_label`
    - `headline`
    - `status_label`
    - `status_color`
    - `summary`
    - `button_label`
    - `variant`

## 7. Change Control Checklist

If a backend PR changes any item above, it must include:

1. Contract doc update.
2. Relevant test updates (`wb_investigations`, `wb_runs`, `wb_notifications`, `wb_workspaces`).
3. Note in PR/commit message that contract changed.
