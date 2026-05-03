# Discovery Script

## Purpose
`discover_eligible_ollama_models.py` discovers Ollama library tags and applies the current protocol filters to produce:
- discovered tags
- eligible tags
- excluded tags (with reason codes)

## Run
From repo root:

```bash
.venv/bin/python open_llm_ons_experiment/scripts/discover_eligible_ollama_models.py
```

Apply manual exclusions afterwards:

```bash
.venv/bin/python open_llm_ons_experiment/scripts/apply_manual_exclusions.py
```

Run the ONS agreement pipeline across eligible models:

```bash
.venv/bin/python open_llm_ons_experiment/scripts/run_experiment.py
```

Reconcile analysis-stage failures from `results.csv` into exclusion counts:

```bash
.venv/bin/python open_llm_ons_experiment/scripts/reconcile_analysis_failures.py
```

Optional overrides:

```bash
.venv/bin/python open_llm_ons_experiment/scripts/discover_eligible_ollama_models.py \
  --start-date 2024-05-01 \
  --end-date 2026-05-01 \
  --manual-exclusions open_llm_ons_experiment/config/manual_exclusions.csv \
  --out-dir open_llm_ons_experiment/artifacts/discovery
```

## Manual Exclusions Template
Template file:
- `open_llm_ons_experiment/config/manual_exclusions.csv`

Columns:
- `family`: exact family slug (optional)
- `tag`: exact tag name (optional)
- `stage`: `manual_exclusion` or `preprocessing`
- `reason_code`: short machine-readable manual exclusion code
- `rationale`: free-text justification

Matching rule:
- `family` + `tag`: both must match
- `family` only: excludes all tags in that family
- `tag` only: excludes that specific tag

When a row matches, the model is excluded with reason code `manual_exclusion`.

Current manual policy includes paired-variant pruning:
- when instruction-tuned and text/base variants are both present for the same stratum, retain instruction-tuned and manually exclude text/base variants.

## Reason Code Definitions
- `latest_alias_tag`: tag is a `latest` alias, not a pinned variant.
- `cloud_model_excluded`: tag is cloud-only, not a local Ollama model.
- `embedding_model_excluded`: embedding model (not generative chat/instruction).
- `specialised_model_excluded`: task-specialised model (for example coding-only or safety-only).
- `vision_specialised_model_excluded`: vision-language model judged vision-specialised.
- `outside_date_window`: model update date is outside the study window.
- `unknown_update_date`: update date could not be determined.
- `exceeds_max_model_size_gb`: model artefact size exceeds the GB cap.
- `below_min_params_b`: parameter count is below the minimum threshold.
- `exceeds_dense_max_params_b`: dense model parameter count exceeds the dense cap.
- `exceeds_moe_max_total_params_b`: MoE total parameter count exceeds the MoE total cap.
- `exceeds_moe_max_active_params_b`: MoE active parameter count exceeds the MoE active cap.
- `unknown_parameter_size`: parameter count could not be determined.
- `manual_exclusion`: excluded manually via `manual_exclusions.csv`.

In `exclusion_reason_counts_latest.csv`, `stage` is:
- `preprocessing` for pre-filter/metadata reasons (e.g. `latest_alias_tag`)
- `manual_exclusion` for manual typed exclusions (for example `manual_exclusion:specialised_model_excluded`)
- `eligibility` for non-manual threshold/task-fit exclusions
