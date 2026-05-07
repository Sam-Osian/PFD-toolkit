# PFD Toolkit — Ops Console · UI-to-backend mapping

This prototype is a **UI-only** Django-templated console mounted at `/ops/`. Each page below maps directly to the existing backend surface. Anything flagged 🔴 needs a backend change before that piece of UI can do real work; everything else should slot onto fields that already exist on the models reachable from the codebase.

Status legend
- ✅ already exists on the backend, just needs view/template plumbing
- 🟡 partially exists — small additions or a new service method
- 🔴 NEW — explicit backend work required

Auth: every view requires `staff_member_required`. Mutations write an `AuditEvent` row.

---

## `approvals.html` — Run review (landing page)

The default Ops landing page. Lists `InvestigationRun.objects.filter(approval_status="pending")` with the most recent first. Clicking a row expands an editor in place over both the run and its linked `Investigation`.

| UI feature | Backend hook | Status |
|---|---|---|
| List of pending runs | `InvestigationRun.objects.filter(approval_status="pending")` | ✅ |
| Investigation `title` edit | `Investigation.title` | ✅ |
| Investigation `question_text` edit | `Investigation.question_text` | ✅ |
| Scope: date_from / date_to / queries / areas / receivers | fields inside `Investigation.scope_json` | 🟡 — needs a typed serializer that round-trips named keys to/from the JSONField. UI never edits raw JSON. |
| Method: filter prompt / theme prompt / extract schema | fields inside `Investigation.method_json` | 🟡 — same as above; typed serializer. |
| Run provider / execution_mode / pipeline_index | keys inside `InvestigationRun.input_config_json` | 🟡 — typed serializer; the UI never edits raw JSON. Provider choices: `openai`, `our-server` (api · openai (user)`, `api · openrouter (user)`, `gifted · server key (OpenAI)`, `gifted · server key (Anthropic)`. |
| Run query_start_date / query_end_date | `InvestigationRun.query_start_date` / `.query_end_date` | ✅ |
| Internal note (not user-visible) | new `InvestigationRun.internal_note` | 🔴 — a separate text field from `approval_note`, never surfaced to the run owner. |
| `Approve & queue` | extend `wb_runs.services.approve_run_for_execution(actor, run, note, scheduled_for)` to also accept and persist edits to the run + investigation in one transaction | 🟡 |
| `Reject…` | `wb_runs.services.reject_run_for_execution(actor, run, reason)` | ✅ |
| `Save edits only` (without approving) | new `wb_runs.services.update_pending_run(actor, run, run_edits, investigation_edits)` | 🔴 |
| Filter chips: type, provider, status segment | trivial query-param filtering on the queryset | ✅ |

---

## `failures.html` — Failure inspection

Read-only triage for runs with `status="failed"` over a chosen window.

| UI feature | Backend hook | Status |
|---|---|---|
| Group tiles by `error_code` | `InvestigationRun.objects.filter(status="failed").values("error_code").annotate(count=Count("id"))` | ✅ |
| Failures table | filtered queryset of `InvestigationRun` with related `User`, `Workspace` | ✅ |
| Stage chip (e.g. `theme.batch[3/12]`) | last `RunEvent.kind` + payload for that run | ✅ |
| Inline RunEvent timeline | `RunEvent.objects.filter(run=run).order_by("created_at")` — render `kind`, `payload` JSON pretty-printed inline | ✅ |
| Run summary panel (provider, wall time, tokens, approved by) | `InvestigationRun` fields + tied `AuditEvent` for the approval | ✅ |
| "Similar in last 7 days" same-error list | filter by `error_code` + window | ✅ |
| `Open in Run review` | route to `/ops/approvals/?run=<id>` and auto-expand that row | ✅ |
| `Requeue with same config` | new `wb_runs.services.requeue_run(actor, run)` — clones the `InvestigationRun` row with the same `input_config_json`, sets `approval_status="pending"`, writes `AuditEvent`. | 🔴 |
| Tab pivots: by code / by user / by workspace | the same queryset, grouped client-side | ✅ |

---

## `workers.html` — Workers & queue

| UI feature | Backend hook | Status |
|---|---|---|
| Worker rows (heartbeat, state, capacity) | `Worker` model — assumed to exist as the backing table for the existing dashboard | ✅ |
| `worker-01` highlighted as **primary** | `Worker.is_primary` boolean (or similar tag) | 🟡 — surfaces the tag if it exists; otherwise add a `kind` / `label` column. |
| Now-running run | `InvestigationRun.objects.filter(worker=w, status="running")` | ✅ |
| Capacity bar | `Worker.in_flight / Worker.max_concurrency` | ✅ |
| Queue table (active + pending capacity) | `RunQueueState` plus filtered `InvestigationRun` | ✅ |

(No fleet-level actions: drain, add capacity, p50 charts have all been removed per the brief.)

---

## `users.html` — Users directory

List-first, sorted by most recent activity. Clicking a row expands a slim profile.

| UI feature | Backend hook | Status |
|---|---|---|
| Directory listing sorted by activity | `User.objects.annotate(last_activity=Max("auditevent__created_at")).order_by("-last_activity")` | ✅ |
| First name / last name / email edit | `User.first_name`, `.last_name`, `.email` | ✅ |
| Suspend account | `User.is_active = False` | ✅ |
| Send re-auth link | existing django-allauth (or whatever the project uses) password-reset / login-link mailer wired up to `auth.PasswordResetForm` | 🟡 — likely a service method already; add an Ops-callable wrapper that writes `AuditEvent`. |
| LLM access mode picker (3 options) | new field `UserLLMConfig.mode`: `user_key` / `gifted` / `local_only` | 🔴 |
| Mode = "User-supplied key" | existing `UserLLMConfig` row holding the user's encrypted key (verified-status indicator only — **the raw key is never displayed**) | 🟡 — guarantee the value is `write-only` from the Ops UI's perspective; show only `provider`, `last_verified`, status. |
| Mode = "Gift a server key" | new `ServerLLMKey` model: `pool_label`, encrypted `secret`, `provider`, `monthly_cap_usd`, `expires_at`, `is_active`. New `GiftedLLMAssignment(user, server_key, provider_default, monthly_cap_usd, expires_at)`. The runtime LLM adapter resolves a user's effective key by checking `GiftedLLMAssignment` first, then falling back to `UserLLMConfig`. The user's settings page surfaces the `provider_default` (which can be set to a friendly `"our server"` label) — **never the underlying key**. | 🔴 |
| Mode = "Local only" | `UserLLMConfig.mode = "local_only"` — the runtime adapter routes the user's runs through `our-server` only, regardless of any key on file. | 🔴 |
| Workspaces · 1 owned + Archive button | `Workspace.objects.filter(owner=user)`; `Workspace.is_archived = True` (set per workspace, not via a force-archive-all). | ✅ |
| Recent activity (24h) timeline | `AuditEvent.objects.filter(actor=user).order_by("-created_at")[:20]` | ✅ |

**Removed by request** (i.e. NOT in the UI): impersonate / sign-in-as, rotate API key, model selector, force-archive-all, `is_staff` / `is_superuser` toggles, password reset (replaced by the lighter "send re-auth link").

---

## `workspaces.html` — Workspace view editor

Detail page for one workspace. Four operations only.

| UI feature | Backend hook | Status |
|---|---|---|
| 1. Remove a report from the user's view | `WorkspaceReportExclusion.objects.create(workspace, report, by=actor)` | ✅ |
| 2. Restore a previously-excluded report | delete that `WorkspaceReportExclusion` row | ✅ |
| 3. Add a report from the global archive that wasn't in the dataset | new `WorkspaceReportInclusion(workspace, report, by, created_at)`; the workspace's report set becomes `(filtered_dataset \ exclusions) ∪ inclusions` everywhere it's read | 🔴 |
| 4. Edit a report's title / summary as it appears in *this* workspace only | new `WorkspaceReportOverride(workspace, report, title_override, summary_override, updated_by, updated_at)`; resolved at read time so the canonical archive record is unchanged | 🔴 |
| Right-rail Dataset summary | counts queried at request time | ✅ |
| "Recent edits" timeline | `AuditEvent.objects.filter(workspace=ws, kind__in=["report_excluded", "report_restored", "report_included", "report_overridden"])` | 🟡 — emit those `AuditEvent.kind` values from the four service methods above. |

---

## Cross-cutting · backend additions checklist

A condensed list of every 🔴 item, in suggested implementation order:

1. **`InvestigationRun.internal_note`** TextField — never surfaced to the run owner.
2. **Typed serializers** for `Investigation.scope_json`, `Investigation.method_json`, `InvestigationRun.input_config_json` — round-trip named keys to and from the JSONField so the Ops form never touches raw JSON.
3. **`wb_runs.services.update_pending_run(actor, run, run_edits, investigation_edits)`** — single-transaction edit of run + linked investigation. Must reject if `approval_status != "pending"`.
4. **Extend `approve_run_for_execution`** to accept the same `run_edits` / `investigation_edits` and apply them atomically before flipping `approval_status`.
5. **`wb_runs.services.requeue_run(actor, run)`** — clone a failed run with `approval_status="pending"`.
6. **`UserLLMConfig.mode`** enum: `user_key` | `gifted` | `local_only`.
7. **`ServerLLMKey`** model — pooled keys held server-side, encrypted at rest. Never serialised over the wire.
8. **`GiftedLLMAssignment(user → server_key)`** with `provider_default`, `monthly_cap_usd`, `expires_at`. UI presents `provider_default = "our server"` as an option (the runtime maps it to whichever provider the assigned key is for).
9. **LLM adapter resolution order**: `GiftedLLMAssignment` → `UserLLMConfig.user_key` → `local`. The user's settings page reads only `provider_default`; raw keys are never returned.
10. **`WorkspaceReportInclusion`** — companion to the existing `WorkspaceReportExclusion`. Resolved set = `(filtered ∪ inclusions) \ exclusions`. Update every place `workspace.reports` is materialised.
11. **`WorkspaceReportOverride(workspace, report, title_override, summary_override)`** — resolved at read time; canonical archive record never written.
12. **Audit emit** — every Ops mutation writes an `AuditEvent` with the `actor=request.user`, `kind`, `target` GenericForeignKey, and a `payload` JSON of the diff.

The frontend prototype is structured to make these changes drop-in: every form field carries a `data-key` that names the path it should write to (e.g. `data-key="scope.queries"`), so a small JS layer can post a typed JSON diff to a single `update_pending_run` endpoint without the templates needing to know how the JSONField is shaped.
