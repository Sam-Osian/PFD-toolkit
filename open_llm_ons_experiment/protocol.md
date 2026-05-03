# Open LLM vs ONS Replication Protocol

## 1. Objective
Evaluate whether open models available through Ollama can match or exceed the previously reported GPT-4.1 agreement with clinical adjudication on the ONS child-suicide replication task.

## 2. Primary Question
On the fixed ONS replication sample, what agreement with clinical adjudication do eligible open Ollama models achieve relative to the GPT-4.1 benchmark?

## 3. Data and Ground Truth
- Fixed dataset: `open_llm_ons_experiment/ons_replication/PFD Toolkit--Consensus Comparison.xlsx`
- Ground truth label: post-consensus child-suicide verdict (`Yes`/`No`)
- Input text fields: investigation, circumstances of death, matters of concern

## 4. Cohort Windows
- Primary cohort window: **May 1, 2024 to May 1, 2026** (24 months)
- Sensitivity cohort window: **May 1, 2025 to May 1, 2026** (12 months)

These windows are fixed a priori for reproducibility.

## 5. Model Eligibility
- Include only Ollama models that are:
- Published/updated within the cohort window
- Non-embedding
- General-purpose text models, plus general-purpose multimodal models
- Technically runnable in the local setup
- Able to produce valid schema-constrained output that passes Pydantic validation

- Exclude:
- Embedding models
- Clearly task-specialised models (for example: code-only, medical-specialised, safety/guardrail-specialised)
- Cloud-only models (this benchmark is local-only by design)
- Alias tags such as `:latest` (non-pinned variants)
- Models below 5B effective parameter size

### 5.1 Variant Selection Policy (Protocol-Level)
The unit of execution is the model tag/variant.

- Evaluate at tag level, not family level.
- Include all eligible tags within the declared caps.
- No canonical tag de-duplication is applied.
- If both instruction-tuned and text/base variants are available for the same model family/size stratum, retain the instruction-tuned variant and exclude the paired text/base variant.
- Record variant metadata (including quantisation and tag) for each run so analyses can be stratified if required.

## 6. Parameter Policy
- Fixed dense minimum size: 5
- Fixed dense cap: 120
- Fixed MoE active cap: 40
- Fixed MoE total cap: 400 (where available)
- Fixed model artefact size cap: 100
- For MoE models, report both total and active/effective parameters when available.

Models exceeding any active threshold are excluded and logged with explicit reason codes.

Rationale for thresholds:
- Dense minimum size of 5 excludes very small dense models that are unlikely to be competitive on this complex screening task.
- Dense cap of 120 retains high-capability dense open models (including 120B-class models) while still constraining extreme outliers.
- MoE active cap of 40 permits sparse MoE inclusion while constraining effective inference-time compute to a practical range.
- MoE total cap of 400 permits larger sparse MoE families while still excluding extreme total-parameter outliers.
- Model artefact size cap of 100 limits operational burden from very large downloads and loading, while still admitting widely used large quantised models.

## 7. Evaluation Procedure
- Use the same task prompt template and same report fields for every model.
- Deterministic settings where possible (temperature, seed, fixed prompt).
- For each model, classify all reports, then compute:
- Agreement with clinical adjudication
- Sensitivity (recall for positive class)
- Specificity (recall for negative class)

## 8. Structured Output and Reasoning Handling
- Output must validate against the required schema (Pydantic gate).
- If a model is a reasoning/thinking model, only the final answer payload is evaluated.
- No constraints or scoring are applied to intermediate reasoning traces/tokens.

## 9. Statistical/Reporting Plan
- Primary report: model-wise performance table and scatterplot (x: parameters in billions, y: agreement with clinical adjudication).
- Reference line: GPT-4.1 benchmark agreement with clinical adjudication from the prior run.
- Sensitivity report: repeat with 12-month cohort and compare direction/magnitude of conclusions.
- All exclusions are logged with explicit reason codes.

## 10. Reproducibility
- Freeze model manifest at run start (with timestamp and eligibility decisions).
- Log model version/tag and runtime metadata.
- Persist incremental results per model and support exact resume after interruption.

## 11. Practical Justification
The 24-month window is used to capture the current generation of open models while maintaining a manageable and reproducible benchmark scope. A 12-month sensitivity analysis is pre-specified to test whether conclusions are robust to a narrower recency definition.
