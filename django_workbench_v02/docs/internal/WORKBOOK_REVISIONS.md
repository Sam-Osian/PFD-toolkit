# Workbook Revisions (Slice 1)

Status: Implemented  
Last updated: 2026-04-20

## What Was Added

1. Per-workbook revision cursor: `Workspace.current_revision`.
2. Revision writer service: `write_workspace_revision(...)`.
3. User actions:
   - undo
   - redo
   - start over
   - revert reports
4. Baseline state snapshot on workbook creation.
5. Revision writes on:
   - investigation create/update
   - report exclusion add/restore
6. Share snapshot compatibility:
   - snapshot links now resolve against `current_revision`;
   - if no valid revision exists, a system revision is seeded.

## State Payload (v1)

Revision `state_json` currently captures:

1. Investigation payload (`title`, `question_text`, `scope_json`, `method_json`, `status`) or `null`.
2. Excluded-report list (`report_identity`, `reason`, `report_title`, `report_date`, `report_url`).

## Behavior Notes

1. `undo` moves cursor to parent revision and applies that state.
2. `redo` moves cursor to newest child revision of current cursor.
3. `start over` restores baseline state and writes a new `restore` revision.
4. `revert reports` clears exclusions and writes a new `restore` revision.
5. Revision rows are append-only for edit/restore writes; undo/redo moves cursor to existing revisions.

## User Endpoints

1. `POST /workbooks/<id>/state/undo/`
2. `POST /workbooks/<id>/state/redo/`
3. `POST /workbooks/<id>/state/start-over/`
4. `POST /workbooks/<id>/state/revert-reports/`

(`workspace/*` aliases also exist for compatibility.)

## Current Limitations (Intentional for Slice 1)

1. Workbook state snapshot is focused on investigation + exclusions only.
2. Redo branch selection is deterministic newest-child behavior.
3. UI is functional baseline, not final Claude design.
