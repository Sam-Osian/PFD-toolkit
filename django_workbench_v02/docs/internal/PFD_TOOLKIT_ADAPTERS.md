# PFD Toolkit Run Adapters (v0.2)

Status: Implemented  
Last updated: 2026-04-19

## 1. Purpose

`wb_runs/pfd_toolkit_adapter.py` is the boundary between Django run orchestration and `pfd_toolkit` execution logic.

This keeps:

1. Workflow orchestration in `wb_runs/worker.py`
2. Toolkit invocation details in one adapter module

## 2. Implemented Adapters

1. `execute_filter_workflow(...)`
2. `execute_themes_workflow(...)`
3. `execute_extract_workflow(...)`
4. `execute_export_workflow(...)`

Each adapter:

1. Reads `run.input_config_json`
2. Resolves date range and report limits
3. Loads reports with `pfd_toolkit.load_reports(...)`
4. Builds `LLM(...)` client from provider config
5. Patches LLM generation callback for worker progress reporting
6. Checks cancellation before/after expensive stages
7. Writes CSV/JSON outputs under runtime artifact path
8. Returns serializable metadata dict consumed by worker artifact creation

The export adapter is file-bundling focused and does not call LLM APIs.

## 3. Configuration Model

### 3.1 Shared

1. `provider`: `openai` or `openrouter`
2. `model_name`
3. `max_parallel_workers`
4. `llm_timeout_seconds`
5. `report_limit`
6. `refresh`
7. `start_date` / `end_date`
8. `artifact_dir`

Environment keys:

1. `OPENAI_API_KEY` when `provider=openai`
2. `OPENROUTER_API_KEY` when `provider=openrouter`
3. Optional `OPENAI_BASE_URL` / `OPENROUTER_BASE_URL` overrides

### 3.2 Filter

1. `search_query` (falls back to investigation question)
2. `filter_df`
3. `produce_spans`
4. `drop_spans`

### 3.3 Themes

1. `trim_approach`
2. `summarise_intensity`
3. `max_tokens` or `max_words`
4. `extra_theme_instructions`
5. `warning_threshold` / `error_threshold`
6. `min_themes` / `max_themes`
7. `seed_topics`

### 3.4 Extract

1. `feature_fields` (required)
2. `produce_spans`
3. `drop_spans`
4. `force_assign`
5. `allow_multiple`
6. `skip_if_present`
7. `extra_instructions`

### 3.5 Export

1. `include_run_types` (optional list)
2. `latest_per_artifact_type`
3. `max_artifacts`
4. `bundle_name`

## 4. Output Contract

All adapters return a dict containing at least `output_path`.

Worker behavior:

1. Uses `output_path` as primary artifact file URI.
2. Persists additional adapter return values into artifact metadata.
3. For theme runs, creates an additional `theme_assignments` artifact when present.
4. For export runs, stores a `bundle_export` artifact pointing to the generated zip.
5. Applies configured artifact storage backend (`file` or `object_storage`) before persisting artifact records.

## 5. Error and Cancellation Contract

Adapter exceptions:

1. `AdapterConfigurationError`: invalid/missing config or credentials.
2. `AdapterCancelledError`: run cancellation observed during adapter execution.

Worker maps these to run statuses:

1. configuration error -> `failed` with `error_code=ADAPTER_CONFIGURATION`
2. cancellation -> `cancelled`
3. uncaught runtime errors -> `failed` with run-type-specific error code

## 6. Why This Boundary Matters

This separation allows the team to evolve:

1. Worker lifecycle behavior (queueing/status/audit/retries)
2. Toolkit prompts/models/provider config
3. Artifact packaging

without tightly coupling those concerns in one module.
