# PFD Toolkit Workbench - User Function Source of Truth

Last updated: 2026-04-20
Scope: authoritative user/admin function inventory across v0.1 (`django_workbench`) and v0.2 (`django_workbench_v02`).

## Product Contract (Agreed)
- `Workbook` is the product term. (Current code still uses `Workspace` model/URLs; rename pending.)
- One workbook has exactly one investigation bundle (filter/themes/extract).
- Every workflow run is anchored to a workbook.
- Shared links are always read-only for viewers.
- Viewers can create their own editable copy.
- Only admin can assign owner role to additional users.
- All significant actions must be persistently logged (query text, options, run config, state changes).
- Excluded-report suite must exist in v0.2.
- Collections must exist in v0.2.

## Legend
- `[NEW]`: introduced in v0.2.
- Access levels:
  - `Public`: no login required.
  - `User`: authenticated non-owner member.
  - `Owner`: workbook owner (non-admin).
  - `Admin`: site admin (superuser).

## Role Boundary Matrix (Current Intended Policy)

| Capability | Public | User | Owner | Admin |
|---|---|---|---|---|
| View public share link (read-only) | Yes | Yes | Yes | Yes |
| Create workbook | No | Yes | Yes | Yes |
| Edit workbook contents | No | If edit access | Yes | Yes |
| Add/update/remove non-owner members | No | No | Yes (if owner has member management) | Yes |
| Grant owner role to another user | No | No | No | Yes |
| Create/revoke share links | No | No | Yes (if owner has share management) | Yes |
| Queue/cancel workflow runs | No | If run permission | Yes | Yes |
| Download artifacts (authorized scope) | Public share only | Yes (if can view workbook) | Yes | Yes |
| Django admin data management | No | No | No | Yes |

## Categorized Function Inventory

### 1) Access and Identity
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Landing/home page | Public | Yes | Yes | v0.1 `/`, `/home/`; v0.2 `/` |
| [NEW] Auth0 login/callback/logout | Public->User | No | Yes | `/auth/login/`, `/auth/callback/`, `/auth/logout/` |
| [NEW] Admin login proxy via Auth0 | Admin | No | Yes | `/admin/login/` |

### 2) Workbook Lifecycle
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Create workbook container | User | Session-only implicit | Yes | POST `/workspaces/` |
| List my workbooks | User | No | Yes | GET `/workspaces/` |
| Workbook detail page | Public/User/Owner/Admin (permissioned) | Explore page only | Yes | GET `/workspaces/<id>/` |
| [NEW] Public workbook listing | Public | No | Yes | GET `/workspaces/public/` |

### 3) Membership and Permissions
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Add member by email | Owner/Admin | No | Yes | POST `/workspaces/<id>/members/add/` |
| Update member flags and access mode | Owner/Admin | No | Yes | POST `/workspaces/<id>/members/<id>/update/` |
| Remove member | Owner/Admin | No | Yes | POST `/workspaces/<id>/members/<id>/remove/` |
| Grant owner role to another user | Admin only | No | [NEW] Enforced in service layer | `add_workspace_member`, `update_workspace_member` |
| Per-member read-only/edit mode | Owner/Admin | No | Yes | membership forms |
| Per-member run permission | Owner/Admin | No | Yes | membership forms |

### 4) Credentials (Per-user in Workbook Scope)
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Save encrypted OpenAI/OpenRouter key | User/Owner/Admin with run permission | Session key only | Yes | POST `/workspaces/<id>/credentials/save/` |
| Delete saved key | User/Owner/Admin with run permission | No | Yes | POST `/workspaces/<id>/credentials/remove/` |
| Optional custom base URL | User/Owner/Admin | No | Yes | credential form |

### 5) Investigation Bundle
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Create investigation | Owner/Admin or edit member | No | Yes | POST `/workspaces/<id>/investigations/` |
| Update investigation | Owner/Admin or edit member | No | Yes | POST `/workspaces/<id>/investigations/<id>/update/` |
| View investigation detail and runs | permissioned viewers | No | Yes | GET `/workspaces/<id>/investigations/<id>/` |
| Constraint target: exactly one investigation per workbook | n/a | n/a | Implemented | DB unique constraint + service guard |

### 6) Workflow Runs (Async)
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Queue filter run | User/Owner/Admin with run permission | SSE in-page | Yes | POST `.../runs/queue/` (`run_type=filter`) |
| Queue themes run | User/Owner/Admin with run permission | SSE in-page | Yes | POST `.../runs/queue/` (`run_type=themes`) |
| Queue extract run | User/Owner/Admin with run permission | SSE in-page | Yes | POST `.../runs/queue/` (`run_type=extract`) |
| [NEW] Queue export run | User/Owner/Admin with run permission | No dedicated run type | Yes | POST `.../runs/queue/` (`run_type=export`) |
| [NEW] Run status timeline/events | permissioned viewers | Partial/SSE only | Yes | GET `/workspaces/<id>/runs/<id>/` |
| [NEW] Cancel run | requester/authorized member/admin | No | Yes | POST `/workspaces/<id>/runs/<id>/cancel/` |

### 7) Artifacts and Exports
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Download run artifact | permissioned viewers | Bundle export | Yes | GET `/workspaces/<id>/runs/<id>/artifacts/<id>/download/` |
| Download multi-file dataset bundle | User/Owner | Yes | Partially (export run path) | v0.1 `download_bundle`; v0.2 export artifacts |
| [NEW] Artifact lifecycle states and expiry | system + viewers | No | Yes | artifact model + lifecycle job |

### 8) Sharing and Read-Only Access
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Create share link | Owner/Admin with share permission | Yes | Yes | POST `/workspaces/<id>/shares/create/` |
| Update/revoke share link | Owner/Admin with share permission | Limited | Yes | POST `/shares/<id>/update/`, `/revoke/` |
| Public share viewing | Public (if share public) | Yes | Yes | v0.1 workbook public URL; v0.2 `/s/<share_id>/` |
| Share link expiry and active toggle | Owner/Admin | No | Yes | share forms |
| Share mode live/snapshot setting | Owner/Admin | No | Yes (exists) |
| Policy target: all shared views read-only | Public/User | Yes | Enforce as core rule |
| Editable copy from shared view | User/Owner/Admin | Yes | Implemented | v0.2 share copy endpoint + audit log |

### 9) Collections and Browsing
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Browse curated collections | Public/User | Yes | Implemented (baseline) | v0.2 `/collections/` |
| Browse collection detail + dashboard | Public/User | Yes | Implemented (baseline detail) | v0.2 `/collections/<slug>/` |
| Custom search collection | Public/User | Yes | Implemented (baseline lexical search) | v0.2 `custom-search` query flow |
| Clone collection into editable copy | User | Yes | Implemented | v0.2 `/collections/<slug>/copy/` |

### 10) Dataset Curation and State History
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Load dataset/date/report bounds | User | Yes | Partial via run config | v0.1 `load_reports` |
| Exclude report from active set | User/Owner | Yes | Implemented | workbook excluded-report endpoints + model |
| Restore excluded report | User/Owner | Yes | Implemented | workbook excluded-report restore endpoint |
| Revert/start-over/undo/redo | User/Owner | Yes | Implemented | `/workbooks/<id>/state/*` actions |
| Persisted revision history | system/user action logging | No | [NEW] Implemented (baseline UX) | `WorkspaceRevision` + `Workspace.current_revision` |
| Full action logging incl. queries/options | system | Partial | [NEW] Implemented (v1 baseline) | `ActionCacheEvent` + internal inspection view |

### 11) Notifications
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| [NEW] Request completion email at queue time | User/Owner/Admin | No | Yes | run queue form |
| [NEW] Trigger policy (`success`/`failure`/`any`) | User/Owner/Admin | No | Yes | `notify_on` |
| [NEW] Dispatcher sends terminal-state email | system | No | Yes | notification dispatcher command/service |

### 12) Keepalive and Expiry
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| [NEW] Human-view keepalive (bot-filtered) | system | No | Yes | view tracking services |
| [NEW] Auto-expire inactive artifacts | system | No | Yes | lifecycle maintenance |
| [NEW] Auto-archive inactive workbooks | system | No | Yes | lifecycle maintenance |

### 13) Admin-Only Operations
| Function | Access | v0.1 | v0.2 | Entry points |
|---|---|---|---|---|
| Manage users and permissions | Admin | Basic | [NEW] custom user admin | Django admin |
| Manage workbooks, memberships, investigations | Admin | No | Yes | Django admin |
| Manage runs, artifacts, shares, notifications | Admin | No | Yes | Django admin |
| Inspect audit events | Admin | No | [NEW] Yes | Django admin |
| Ensure admin account | Admin/ops | No | [NEW] Yes | `manage.py ensure_admin_user` |

## v0.1 Explicit Action Surface (for parity tracking)
- `load_reports`
- `set_active_action`
- `filter_reports`
- `discover_themes`
- `accept_themes`
- `discard_themes`
- `extract_features`
- `undo`
- `redo`
- `start_over`
- `revert_reports`
- `exclude_report`
- `restore_excluded_report`
- `refresh_collections_cache`
- `download_bundle`

SSE endpoints in v0.1:
- `POST /sse/filter/`
- `POST /sse/themes/`
- `POST /sse/extract/`

## Immediate Enforcement Notes (implemented now)
- Non-admin owners can manage members but cannot promote anyone to owner.
- Admin retains ability to assign owner role.

## Next Contract Locks to Implement
1. Rename `Workspace` -> `Workbook` in models/routes/UI.
2. Enforce read-only semantics in share-view endpoints/UI and complete editable-copy flow polish.
3. Expand collections from baseline parity to Claude mock-up UX/spec.
4. Expand action-cache coverage/UX beyond v1 baseline where support workflows require it.

## Reality Check (Code vs SoT) - 2026-04-20

This section tracks what is truly shipped in v0.2 right now (not just model scaffolding).

### Confirmed Strongly Implemented
- Auth spine (Auth0 login/callback/logout + admin proxy) and role boundary enforcement.
- Workbook lifecycle (create/list/detail), membership permissions, and admin-owner-only promotion rule.
- Async run engine for `filter`, `themes`, `extract`, `export` with queueing, worker processing, status transitions, cancellation, events, and artifact records.
- Completion notifications (request + dispatcher flow).
- Artifact download endpoints and storage abstraction.
- Share links (snapshot/live), public read-only view, and editable copy flow.
- Excluded reports (exclude + restore + permission enforcement).
- Stateful workbook revision controls (`undo`, `redo`, `start_over`, `revert_reports`) with revision cursor semantics.
- Action-cache v1 with query/options/state-before/state-after payloads and permissioned workbook inspection view.
- Collections browsing and copy flow, including thematic collections and theme slug mapping.

### Implemented but UI/UX is Still Baseline
- Investigation detail and run queue UX are functional but raw (`form.as_p`/JSON-first workflow controls).
- Run detail page exposes status/events/artifacts/notifications but lacks production-grade operator UX.
- Workbook detail page is functionally complete for members/shares/credentials/exclusions, but still admin-style markup.

### Gaps Blocking Full v0.1 Parity
- No critical parity blockers remain on settings/theme preferences because these are retired from v0.2 scope.

## v0.2 Execution Queue (Authoritative)

Order is set to reduce churn: lock behavior first, then complete UI.

### Stage 1 - Stateful Workbook Actions Parity (Critical)
Goal: restore core v0.1 manipulation semantics in v0.2.

Status: Completed (behavior complete; UX remains baseline pending full design pass).

Deliverables:
1. Workbook revision writer service (append-only snapshots with clear `change_type`).
2. User actions:
   - `undo`
   - `redo`
   - `start_over`
   - `revert_reports`
3. Guardrails:
   - per-workbook revision cursor semantics
   - immutable history
   - share snapshot compatibility with restored states
4. Tests:
   - service tests for branching and restore correctness
   - view tests for permission and edge cases

Acceptance check:
- A user can move backward/forward in workbook state without data loss.
- A "start over" reset is reversible via history.
- Share snapshots consistently reflect the chosen revision.

### Stage 2 - Investigation and Workflow UX Parity (Critical)
Goal: make existing run engine fully usable and understandable to non-technical users.

Deliverables:
1. Replace raw JSON-heavy run queue UX with guided controls for filter/themes/extract/export.
2. Investigation page improvements:
   - clear run-type intent
   - query date bounds controls
   - provider/model inputs with safer defaults
3. Run detail improvements:
   - clearer status timeline
   - artifact intent labels
   - explicit cancellation states/messages

Acceptance check:
- A user can configure and launch each run type without editing raw JSON.
- Run progress and artifacts are understandable without admin knowledge.

### Stage 3 - Collections and Exclusions Deep Parity (High)
Goal: finish cross-flow parity between collections, workbook state, and exclusions.

Deliverables:
1. Ensure collections/filters/exclusions interplay is deterministic in all run types.
2. Lock expected thematic/rule collection behavior against approved schema updates.
3. Preserve active filter context when copying collections.

Acceptance check:
- Repeated copy/filter/exclude cycles produce predictable workbook state and run scope.

### Stage 4 - Settings and Preference Surface (Retired)
Goal: retired for v0.2 to keep focus on workflow/investigation delivery.

Notes:
1. v0.1 `set_ui_theme` and `save_settings` are explicitly out of scope for v0.2.
2. Reintroduce only if future UX requirements make persistent preferences operationally necessary.

### Stage 5 - Action Cache / Transparency Layer (Medium)
Goal: operational transparency and replay confidence.

Status: Completed (v1 baseline).

Deliverables:
1. Formal action-cache schema for query/options/state deltas (beyond coarse audit events).
2. Hook all major user actions and run submissions into this log.
3. Internal inspection view/tooling for debug and support operations.

Acceptance check:
- For any workbook state, we can trace what changed, when, and by whom at option/query level.

### Stage 6 - Workbook Naming Migration + Full UI Pass (After Behavior Lock)
Goal: terminology and design polish after contracts are stable.

Deliverables:
1. User-facing terminology migration (`Workspace` label -> `Workbook` label).
2. Claude design system port on stable backend contracts.
3. Accessibility and responsive QA pass.

Acceptance check:
- UI improvements do not require behavior rewrites.
- Terminology is consistent for end users.
