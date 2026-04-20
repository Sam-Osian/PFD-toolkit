# Investigation Wizard Stage 2 Plan (Chained MVP)

Status: Approved for implementation  
Last updated: 2026-04-20

## 1. Objectives

1. Replace raw JSON-first investigation/run UX with a guided wizard.
2. Keep execution asynchronous and server-side.
3. Enforce sequential chaining for selected methods.
4. Support conditional configuration screens for selected methods only.

## 2. Locked Product Decisions

1. No LLM agentic defaulting for MVP.
2. Scope step is temporal only (no collection selector in wizard).
3. Network method is hidden for now.
4. Fixed execution order only.
5. Pipeline must be chained in MVP:
   - filter -> themes -> extract
6. Pipeline continuation policy:
   - continue-on-fail enabled.

## 3. Wizard Stages

## Stage 1 - Question

Compulsory: Yes

Inputs:

1. Investigation title (required).
2. Research question (required).

Persists to:

1. `Investigation.title`
2. `Investigation.question_text`

## Stage 2 - Scope (Temporal)

Compulsory: Yes

Options:

1. All reports
2. Last 3 years
3. Last year
4. Last 6 months
5. 100 most recent reports

Maps to run payload:

1. `query_start_date` / `query_end_date`
2. `input_config_json.report_limit=100` for "100 most recent reports"

## Stage 3 - Method Selection

Compulsory stage: Yes

Options:

1. Screen & filter (locked on)
2. Discover themes (default off)
3. Extract structured data (default off)

Removed:

1. Build network view (hidden)

## Stage 4A - Themes Config (Conditional)

Shown when: themes selected

Compulsory when shown: No (defaults allowed)

Inputs:

1. Seed topics (optional)
2. Min/max themes (optional)
3. Extra instructions (optional)

Maps to run payload:

1. `seed_topics`
2. `min_themes`, `max_themes`
3. `extra_theme_instructions`

## Stage 4B - Extract Config (Conditional)

Shown when: extract selected

Compulsory when shown: Yes (`feature_fields` required)

Inputs:

1. Feature schema builder (`feature_fields`)
2. Optional extract flags (allow_multiple/force_assign/etc.)

## Stage 5 - Review & Launch

Compulsory: Yes

Displays:

1. Question + scope summary
2. Method plan and ordering
3. Continue-on-fail policy
4. Optional completion notifications

Launch output:

1. First run queued (`filter`)
2. Pipeline metadata embedded in run config

## 4. Chained Pipeline Contract (Backend)

Execution sequence:

1. `filter` always runs first
2. if selected, `themes` runs next on `filter` output artifact
3. if selected, `extract` runs next on:
   - `themes` assignments artifact when themes selected
   - otherwise `filter` output artifact

Continuation behavior:

1. Terminal statuses that continue:
   - `succeeded`
   - `failed`
   - `timed_out`
2. `cancelled` stops pipeline.

Chaining safety:

1. Downstream stages can require upstream artifact.
2. If missing, downstream fails fast with explicit configuration error.

## 5. Current Backend Support Implemented

1. Worker auto-queues next pipeline stage based on config metadata.
2. Worker propagates upstream artifact id to next run.
3. Adapter supports `input_artifact_id` for themes/extract CSV loading.
4. Adapter supports `pipeline_require_upstream_artifact` fail-fast behavior.

## 6. UI Work Next (Stage 2 UX)

1. Replace `RunQueueForm`-heavy investigation page with step-driven wizard.
2. Build conditional route flow:
   - skip themes/extract config screens when methods not selected.
3. Surface pipeline timeline and per-stage status clearly.
4. Keep fallback/manual run queue endpoint during transition.

## 7. Testing Requirements

1. Wizard form validation tests per stage.
2. Integration tests for branching paths:
   - filter-only
   - filter+themes
   - filter+extract
   - filter+themes+extract
3. Pipeline tests:
   - success chain
   - fail-then-continue
   - cancel-stops-chain
