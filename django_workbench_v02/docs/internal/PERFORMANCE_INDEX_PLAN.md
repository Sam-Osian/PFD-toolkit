# Performance Index Plan (MVP Baseline)

Status: Implemented  
Date: 2026-04-20

## Scope

This pass adds only high-confidence indexes tied to existing hot query patterns in web + worker paths.

## Added Indexes

## 1) Investigation run queue claim order

Model: `wb_runs.InvestigationRun`  
Index: `idx_run_stat_queue_cr` on `(status, queued_at, created_at)`

Why:

1. Worker claims use `status IN (...)` then `ORDER BY queued_at, created_at`.
2. Existing indexes did not align to that order without a leading unrelated column.
3. This improves queue polling/claim latency as run volume grows.

## 2) Artifact lifecycle scan

Model: `wb_runs.RunArtifact`  
Index: `idx_art_stat_exp` on `(status, expires_at)`

Why:

1. Lifecycle maintenance scans artifacts filtered by `status=READY`.
2. Expiry updates and inactive transitions are driven by status + expiry logic.
3. Existing index was workspace-leading, which is weaker for global status scans.

## 3) Workbook archived/live ordering

Model: `wb_workspaces.Workspace`  
Index: `idx_ws_arch_upd_desc` on `(archived_at, -updated_at)`

Why:

1. UI and lifecycle paths frequently split live vs archived workbooks.
2. Listing paths then sort by recency (`updated_at DESC`).
3. This reduces sort/filter cost when workbook count grows.

## Migration Files

1. `wb_workspaces/migrations/0005_workspace_idx_ws_arch_upd_desc.py`
2. `wb_runs/migrations/0002_investigationrun_idx_run_stat_queue_cr_and_more.py`

## Validation

1. `manage.py check` passed.
2. `manage.py test wb_workspaces wb_runs` passed (65 tests).

## Follow-up (Post-MVP UI phase)

1. Capture top `slow_query` signatures from production logs.
2. For repeated slow SQL, run `EXPLAIN ANALYZE` on Postgres.
3. Add/drop indexes based on measured query plans, not assumptions.
