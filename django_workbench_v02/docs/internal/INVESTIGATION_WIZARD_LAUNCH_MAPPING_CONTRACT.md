# Investigation Wizard Launch Mapping Contract (Slice 1)

Status: Active  
Last updated: 2026-04-22

## 1. Purpose

Define the exact mapping from wizard inputs to:

1. persisted workspace/investigation state
2. run queue payload (`InvestigationRun.input_config_json`)

This is the implementation contract for Slice 1 (launch payload integrity).

## 2. Mapping Table

| Wizard input | Source field | Persisted destination | Run config destination | Behavior/default |
|---|---|---|---|---|
| Title | `title` | `Workspace.title`, `Investigation.title` | n/a | Required. |
| Description | `question_text` | `Workspace.description`, `Investigation.question_text` | `search_query` baseline fallback | Optional; empty string preserved. |
| Scope option | `scope_option` | `Investigation.scope_json.temporal_scope_option` | `query_start_date`/`query_end_date` or `report_limit` | Deterministic via `temporal_scope_parameters`. |
| Custom start date | `custom_start_date` | `Investigation.scope_json.query_start_date` | `query_start_date` | Required only when `scope_option=custom_range`. |
| Custom end date | `custom_end_date` | `Investigation.scope_json.query_end_date` | `query_end_date` | Required only when `scope_option=custom_range`. |
| Method: filter | `run_filter` | `Investigation.method_json.run_filter` | stage included in `pipeline_plan` | At least one method must be selected. |
| Method: themes | `run_themes` | `Investigation.method_json.run_themes` | stage included in `pipeline_plan` | Ordered after filter. |
| Method: extract | `run_extract` | `Investigation.method_json.run_extract` | stage included in `pipeline_plan` | Ordered after themes. |
| Filter query | `search_query` | n/a | `search_query` (filter stage) | Required when filter stage selected. |
| Filter keep matches only | `filter_df` | n/a | `filter_df` | Defaults true if omitted. |
| Filter include quotes | `include_supporting_quotes` | n/a | `produce_spans` | Mapped to boolean. |
| Filter selected filters | `coroner_filters/area_filters/receiver_filters` | n/a | `selected_filters` | CSV values normalized to lists. |
| Themes min | `min_themes` | n/a | `min_themes` | Optional int. |
| Themes max | `max_themes` | n/a | `max_themes` | Optional int; validated with min<=max. |
| Themes seed topics | `seed_topics` | n/a | `seed_topics` | Optional; passed through. |
| Themes guidance | `extra_theme_instructions` | n/a | `extra_theme_instructions` | Optional; passed through. |
| Extract feature rows | `feature_fields` | n/a | `feature_fields` | Required when extract stage selected. |
| Extract allow multiple | `allow_multiple` | n/a | `allow_multiple` | Optional boolean. |
| Extract force assign | `force_assign` | n/a | `force_assign` | Optional boolean. |
| Extract skip if present | `skip_if_present` | n/a | `skip_if_present` | Optional boolean (default true). |
| Extract include quotes | `extract_include_supporting_quotes` | n/a | `produce_spans` | Optional boolean. |
| Provider | `provider` | n/a | `provider` | Guardrailed enum/allowlist at preflight. |
| Model | `model_name` | n/a | `model_name` | Guardrailed at preflight. |
| Concurrency | `max_parallel_workers` | n/a | `max_parallel_workers` | Clamped 1..32. |
| API key | `api_key` | `WorkspaceCredential.encrypted_api_key` (when provided) | n/a | Encrypted at rest. |
| Notify on completion | `request_completion_email` + `notify_on` | `NotificationRequest` record | n/a | Created at launch when requested. |

## 3. Run-Level Contract Fields (always set)

For wizard launch, run config includes:

1. `execution_mode=real`
2. `provider`
3. `model_name`
4. `max_parallel_workers`
5. `pipeline_plan`
6. `pipeline_index=0`
7. `pipeline_continue_on_fail=true`
8. `pipeline_require_upstream_artifact=false` (first stage)

## 4. Intent-Preservation Rules

1. Optional blank description remains blank (no placeholder substitution).
2. Explicitly provided values must not be silently overridden.
3. Stage plan order is fixed and deterministic (`filter -> themes -> extract`).
4. Scope resolution uses one deterministic function (`temporal_scope_parameters`).

## 5. Validation and Tests

Primary tests:

1. `wb_investigations/tests.py::InvestigationModalWizardLaunchTests`
2. form validation tests for scope/method/filter/themes/extract
3. readiness guardrail tests for provider/model/credential checks

## 6. Known Related Follow-up (outside Slice 1)

1. Pipeline-level completion notification semantics (Slice 3).
2. Workspace card/dashboard design parity (Slice 4/5).
