# Open LLM Experiment Pipeline Spec

## 1. Scope
This document specifies the operational pipeline for:
- Automated Ollama model discovery and eligibility filtering
- Pull/run/delete execution orchestration
- Incremental results persistence
- Resume-safe long-running evaluation
- Incremental interactive plotting

It complements `protocol.md` (scientific method) and focuses on implementation behaviour.

## 2. Directory Layout
- `open_llm_ons_experiment/ons_replication/` copied source replication assets
- `open_llm_ons_experiment/protocol.md` study protocol
- `open_llm_ons_experiment/pipeline_spec.md` this spec
- `open_llm_ons_experiment/artifacts/` generated outputs
- `open_llm_ons_experiment/artifacts/model_manifest.csv`
- `open_llm_ons_experiment/artifacts/results.csv`
- `open_llm_ons_experiment/artifacts/exclusions.csv`
- `open_llm_ons_experiment/artifacts/run_state.json`
- `open_llm_ons_experiment/artifacts/plots/results_interactive.html`
- `open_llm_ons_experiment/artifacts/plots/results_static.png`

## 3. Model Discovery (Automated)

### 3.1 Source Strategy
- Primary automated source: scrape `https://ollama.com/library` plus model detail pages.
- Optional metadata enrichment source: Ollama API model details (`/api/show`) after pull.
- Local installed inventory source: `http://localhost:11434/api/tags`.

### 3.2 Why Scraping
Library pages expose key inclusion metadata (updated date, task tags, sizes, cloud/vision/embedding labels) needed for systematic filtering.

### 3.3 Snapshot
- At start, generate a dated snapshot manifest with all discovered candidates and raw metadata.
- Manifest is frozen for that run to ensure reproducibility.

## 4. Eligibility Filter
- Time filter: updated within selected window (24-month primary; 12-month sensitivity run).
- Exclude labels/families indicating embeddings.
- Exclude clearly specialised families (code-only, medical-specialised, safety-specialised, etc.).
- Include general multimodal models, exclude vision-specialised VLMs by default unless explicitly overridden.
- Apply parameter constraints:
- Minimum: `min_params_b = 5`
- Dense max: `dense_max_params_b = 80`
- MoE active max: `moe_max_active_params_b = 40`
- MoE total max: `moe_max_total_params_b = 200` where metadata is available
- Model artefact size max: `max_model_size_gb = 60`

All include/exclude decisions are written to `model_manifest.csv` and `exclusions.csv` with explicit reason codes.

## 5. Pydantic Compliance Gate (Mandatory)
- Before full evaluation, run a short schema-validation preflight for each model.
- Pass criterion: model returns parseable JSON that validates against required Pydantic schema within a fixed retry budget of 3 attempts.
- Fail criterion: repeated invalid/unsupported structured outputs.
- Failing models are excluded with reason code `pydantic_gate_failed`.

## 6. Pull / Run / Delete Lifecycle

### 6.1 Install Behaviour
- If model is not installed locally: pull model (`ollama pull` or `/api/pull`).
- If model is already installed: mark as preexisting.

### 6.2 Deletion Behaviour
- After model evaluation:
- Delete only models pulled by this run.
- Never delete models that were preexisting at run start.
- Use `ollama rm` or `/api/delete`.

### 6.3 Failure Safety
- If run exits unexpectedly, restart logic checks installed state and manifest state before any deletion.
- Deletion is idempotent and guarded by ownership flag (`pulled_by_run=true`).

## 7. Evaluation Execution
- Reuse the replication screening task and fixed input columns.
- Use deterministic settings where supported (temperature/seed).
- For reasoning models, parse and validate only final output payload.
- Ignore intermediate reasoning tokens/content.

## 8. Incremental Persistence
- After each model completes:
- Append row to `results.csv`
- Flush/sync to disk immediately
- Regenerate plots immediately
- Update run state

- Required per-row fields:
- `model`, `tag`, `family`, `is_moe`, `params_total_b`, `params_active_b`
- `agreement_with_clinical_adjudication`, `sensitivity`, `specificity`
- `local`, `installed_preexisting`, `pulled_by_run`
- `started_at`, `finished_at`, `status`, `error_reason`

## 9. Interruption/Resume Robustness
- `run_state.json` tracks current model and phase.
- Model status states: `pending`, `in_progress`, `completed`, `failed`, `excluded`.
- On restart:
- Load manifest/results/state
- Reconcile partial `in_progress` entries back to `pending` unless completion row exists
- Skip `completed`
- Continue from next eligible pending model

## 10. Plotting
- Generate interactive scatter using Plotly:
- X: parameters (B)
- Y: agreement with clinical adjudication
- Hover: model id, family, update date, agreement with clinical adjudication/sensitivity/specificity, dense vs MoE, preexisting/pulled
- Include GPT-4.1 reference line (horizontal dotted line)

- Also generate static PNG for manuscript drafts.

## 11. Optional Web Publishing
- Optional route candidate: `/open-llm-experiment`
- Preferred behaviour:
- Serve generated interactive artefact from `artifacts/plots/results_interactive.html`
- Add access controls if privacy is required
- Treat unlinked URL alone as obscurity, not security

## 12. Logging and Auditability
- Persist:
- Raw discovery snapshot
- Filter decisions with reason codes
- Runtime errors per model
- Pull/delete actions
- Tool/version metadata

This ensures peer-review traceability and exact rerun capability.
