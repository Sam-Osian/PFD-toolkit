# Investigation Pipeline Gap Remediation (v0.2)

## Purpose
This document defines the remediation plan for investigation launch and pipeline execution gaps, with explicit requirements for:
- User-facing UI behavior
- Internal service and worker behavior
- Data/state contracts
- Acceptance checks

The goal is to ensure investigation launch is truthful, robust, and operationally predictable.

---

## First-Principles Contract
An investigation pipeline is only valid if all of the following are true:
1. User has permission to run workflows in the target workspace.
2. A pipeline plan exists and includes at least one stage.
3. Required prerequisites are satisfied before launch (especially API credential for chosen provider).
4. The system can execute each stage asynchronously server-side and report status correctly.
5. Failure/cancel behavior is explicit and understandable.
6. Outputs and run history are auditable.

---

## Gap Summary (Observed)

## P0 Gaps (must fix first)
1. **Misleading credential messaging in wizard review**
   - Current modal review says API key is saved regardless of actual credential state.
   - This violates trust and creates guaranteed launch failures for users without credentials.

2. **No explicit preflight readiness gate in wizard UI**
   - Backend checks credential at launch, but UI does not clearly show readiness before launch.
   - Users can complete setup and only then discover launch cannot proceed.

## P1 Gaps (high priority)
3. **Dual wizard implementations drifting**
   - Modal wizard and server wizard page contain overlapping logic and can diverge.
   - Risk: one path validates behavior that the other path does not.

4. **Pipeline continue-on-fail semantics are under-specified in UI**
   - Stages can continue after upstream failure, then fail from missing upstream artifact.
   - Technically valid but user-unfriendly unless represented clearly.

5. **Provider/model validation is too soft pre-launch**
   - Invalid or unsupported model/provider combinations are mostly discovered at runtime.

## P2 Gaps (quality and clarity)
6. **Credential management is disconnected from launch context**
   - Users are not guided in-context to resolve missing key from the wizard itself.

7. **Pending workspace state detail is not yet explicit enough**
   - Pipeline stage status should be easy to inspect from workspace listing/detail.

---

## Target UX Specification

## A) Investigation Wizard (Launch Readiness)
On the wizard review step, show a **Launch readiness panel** with hard checks:
- `Credential`: `Ready` / `Missing`
- `Pipeline`: `Ready` / `No stages selected`
- `Permissions`: `Ready` / `Insufficient permissions`

Behavior:
- If any required check fails, disable `Launch` button.
- Show actionable inline guidance with a direct action.
  - Example: `Missing OpenAI API key for this workspace. [Add key]`
- Replace any static text implying a key exists.

UI states:
- `Ready` state: green indicator and enabled Launch.
- `Blocked` state: red/amber indicator, disabled Launch, explicit reason.
- `Checking` state optional only if async checks are added; otherwise compute server-side on GET.

## B) In-Context Credential Flow
- Add `Add key` action in the wizard review panel.
- Open an in-modal credential form (provider, API key, optional base URL) OR deep-link to credential section and return.
- After save, readiness panel refreshes and Launch becomes available if all checks pass.

## C) Pending Workspace and Stage Visibility
- After launch, workspace row shows:
  - `Pending` badge
  - `Current stage` (Filtering / Themes / Extract)
  - `Last update time`
- Workspace detail shows timeline:
  - queued -> starting -> running -> succeeded/failed/cancelled
  - continue-on-fail note when applicable.

---

## Internal Implementation Specification

## 1) Unified Preflight Function (single source of truth)
Create a service-level preflight function used by all launch paths:

`evaluate_investigation_launch_readiness(actor, investigation, wizard_state) -> readiness`

Return shape:
- `can_launch: bool`
- `checks:`
  - `permission_check`
  - `pipeline_check`
  - `credential_check`
  - optional `provider_model_check`
- `blocking_errors: list[str]`
- `actions: list[{"type": "...", "label": "...", "target": "..."}]`

Rules:
- If pipeline has real stages requiring LLM and no credential for selected provider, block.
- If permission missing, block.
- If no stages selected, block.

The existing launch service still re-validates all constraints (defense in depth).

## 2) Replace Misleading Credential Copy
Remove static claim that key is saved.
Replace with dynamic text from readiness result:
- `OpenAI key present (••••1234)` if present (optional last4)
- `No OpenAI key saved for this workspace` if missing

## 3) Collapse to One Wizard Execution Contract
Keep one authoritative wizard flow contract:
- canonical stage sequence construction
- canonical state serialization
- canonical launch preflight
- canonical launch service call

If two render paths remain, both must call the same service functions for:
- stage resolution
- validation
- preflight
- launch

## 4) Continue-on-fail Stage Policy
Make behavior explicit and deterministic:
- Default policy remains continue-on-fail (as requested).
- For each queued next stage after upstream failure:
  - mark event: `continued_after_failed_upstream=true`
  - if upstream artifact required and missing, fail fast with clear `error_code` (`MISSING_UPSTREAM_ARTIFACT`).

UI timeline must display:
- `Continued after upstream failure`
- `Skipped` (if you later choose skip semantics)

## 5) Provider/Model Guardrails
At preflight (before launch):
- validate provider in allowed enum
- validate model_name non-empty
- optional provider-specific allowlist (recommended for beta)

At runtime:
- keep adapter-level defensive validation and clear error mapping.

## 6) Notification Semantics
Keep current completion notification behavior:
- trigger on terminal status only (`success`, `failure`, or `any`)

Wizard review copy must state this explicitly:
- `Emails are sent when the pipeline reaches terminal status (not after each stage).`

---

## Data/State Contracts

## Run config contract (launch payload)
Must include:
- `execution_mode=real`
- `provider`
- `model_name`
- `pipeline_plan`
- `pipeline_index=0`
- `pipeline_continue_on_fail=true`
- stage-specific configs (filter/themes/extract)

## Workspace status contract
Workspace pending/active status should derive from latest investigation pipeline state:
- pending: any non-terminal run in active pipeline
- failed-warning: latest terminal in pipeline is failed/timed-out/cancelled and pipeline not complete
- complete: final stage terminal success (or terminal completion by policy)

---

## Acceptance Checks

## P0 Acceptance
1. Wizard review never claims API key exists unless verified.
2. Launch button is disabled when key is missing for selected provider.
3. User can save credential from launch context and launch without leaving investigation flow (or with explicit return path).
4. Launch attempt with missing key fails with clear, user-safe message and no silent failure.

## P1 Acceptance
5. Both wizard entry routes produce identical validation and launch behavior.
6. Timeline clearly indicates continue-on-fail transitions.
7. Missing-upstream-artifact failure uses explicit error code and explanatory text.
8. Invalid provider/model is blocked at preflight.

## P2 Acceptance
9. Workspace list and workspace detail both show pending stage status.
10. Review copy clearly explains completion-email trigger semantics.

---

## Suggested Implementation Order
1. P0: readiness contract + truthful review panel + launch gating.
2. P0: in-context credential save path.
3. P1: unify wizard contracts across modal/page routes.
4. P1: continue-on-fail timeline + explicit upstream artifact error semantics.
5. P1: provider/model preflight guardrails.
6. P2: pending workspace status UX polish and final messaging cleanup.

---

## Notes
- This plan intentionally prioritizes operational truth over visual fidelity.
- Visual matching to Claude Design should continue, but must not override preflight correctness and launch reliability.
