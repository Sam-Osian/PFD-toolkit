# Investigation-to-Workspace Contract (v0.2)

Status: Canonical contract (Slice 0)  
Last updated: 2026-04-22

## 1. Purpose

Define the canonical product and data contract for how an investigation wizard request becomes a new workspace and pipeline run.

This is the source of truth for:

1. Product behavior
2. Backend implementation
3. QA acceptance criteria
4. UX copy alignment

## 2. Canonical Definitions

1. `Investigation wizard`: the creation-and-launch flow for a new workspace pipeline.
2. `Workspace`: persisted container linked to a user/membership model.
3. `Investigation`: one investigation record per workspace (enforced).
4. `Pipeline`: ordered run plan composed of `filter`, `themes`, `extract` (and export as a separate run path).

## 3. Core Contract

1. Starting the investigation wizard creates a **new workspace**.
2. Wizard `Title` becomes workspace title and investigation title.
3. Wizard `Description` becomes investigation description (`question_text`) and workspace description where specified by flow policy.
4. The wizard launch queues real runs through the async worker path.
5. Runs continue server-side; browser/session closure does not stop execution.
6. One workspace has exactly one investigation.

## 4. Pipeline Ordering Contract

Stage order is deterministic and only includes selected stages in this order:

1. `filter` (if selected)
2. `themes` (if selected)
3. `extract` (if selected)

Rules:

1. At least one stage must be selected.
2. `pipeline_continue_on_fail=true` is default policy unless explicitly changed by product decision.
3. Stages requiring upstream artifact (`themes`, `extract` in chained mode) fail fast with explicit missing-upstream error semantics.

## 5. Status Semantics Contract

Run statuses:

1. `queued`
2. `starting`
3. `running`
4. `cancelling`
5. terminal: `succeeded`, `failed`, `timed_out`, `cancelled`

Workspace-level pipeline state projection:

1. `pending`: latest relevant run is non-terminal.
2. `failed-warning`: latest terminal state is failure-like and pipeline not complete by policy.
3. `complete`: pipeline reached terminal completion state by policy.

## 6. Data Mapping Contract (Wizard -> Persisted State)

1. Title/description -> investigation/workspace metadata.
2. Scope choice -> query dates and/or report limit.
3. Method toggles -> pipeline plan and run flags.
4. Filter config -> query, filter behavior, selected filters.
5. Themes config -> seed topics, bounds, optional guidance.
6. Extract config -> feature rows and extraction options.
7. Review config -> provider/model/concurrency + notification intent.

## 7. Security and Credential Contract

1. API keys are encrypted at rest.
2. Runtime credential resolution decrypts only for execution path.
3. Launch is blocked when required credential is unavailable for real execution.
4. Credential scope supports workspace/user-level fallback per implementation.

## 8. Non-Goals for Slice 0

Slice 0 does not implement:

1. Final workspace card visual parity to Claude design.
2. Pipeline-level notification trigger refactor.
3. Additional lifecycle policy changes.

## 9. Slice 0 Acceptance Criteria

1. Contract is documented and referenced by implementation docs.
2. Team language is aligned: investigation wizard is a new-workspace creation-and-launch flow.
3. No docs in current remediation path claim the wizard should launch into an existing workspace by default.

