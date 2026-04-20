# Action Cache (Slice 2)

Status: Implemented (v1 baseline)  
Last updated: 2026-04-20

## Why This Exists

Audit events are useful for coarse tracking, but support/debug workflows need replay-grade detail:

1. what query/options were used,
2. what changed before/after,
3. which entity was affected.

## Data Model

`wb_auditlog.ActionCacheEvent` stores:

1. scope: `workspace`, `user`
2. identity: `action_key`, `entity_type`, `entity_id`
3. replay payload:
   - `query_json`
   - `options_json`
   - `state_before_json`
   - `state_after_json`
   - `context_json`
4. `created_at`

## Logged Action Families (v1)

1. workbook create
2. membership add/update/remove
3. credential save/delete
4. report exclusion add/restore
5. investigation create/update
6. run queue/cancel-request
7. revision write + undo/redo/start-over/revert-reports cursor actions
8. share link create/update/revoke
9. share copy to workbook

## Inspection Tooling

Permissioned internal view:

1. `GET /workbooks/<id>/action-cache/`
2. (`/workspaces/<id>/action-cache/` alias)

Access:

1. superuser, or
2. workbook owner with member-management permission.

UI entry point:

1. workbook detail page link: `Inspect workbook action cache`.

## Notes

1. This is an internal transparency/debug surface, not end-user polished UI.
2. Payload shape can evolve; keep `action_key` stable where possible.
