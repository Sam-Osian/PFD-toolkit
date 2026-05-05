# Pipeline Spec

## 1. Inputs

Per report:

1. `report_id`
2. `date`
3. `url`
4. `circumstances`
5. `concerns`

Text used for extraction:

`source_text = "Circumstances:\n{circumstances}\n\nConcerns:\n{concerns}"`

## 2. Step Graph

1. **Extract issue sentences** (Gemma 4 via Ollama; deterministic settings; no reasoning text output).
2. **Normalize issues** (trim, dedupe within report, basic text cleaning).
3. **Embed issue sentences** (sentence embedding model).
4. **Reduce dimensions** (UMAP or equivalent).
5. **Cluster** (HDBSCAN or equivalent density clustering).
6. **Assign cluster labels** (Gemma summarization over representative issue sentences).
7. **Persist mappings**:
   1. report -> issue sentences
   2. issue sentence -> cluster (or unclustered)
8. **Aggregate recurrence metrics** (counts over time, per area/coroner/receiver where needed).

## 3. Runtime Pattern

Use two job families:

1. `issue_extract_job`
   1. Trigger: report dataset refresh or forced rerun.
   2. Unit: batch of report rows.
   3. Idempotency key: `report_id + extractor_version`.
2. `issue_cluster_job`
   1. Trigger: extraction completion or clustering config change.
   2. Unit: full-corpus clustering run snapshot.
   3. Idempotency key: `corpus_snapshot_id + embedding_model + cluster_config_hash`.

## 4. Versioning

Record versions on every generated row:

1. `extractor_version` (prompt/schema/model bundle)
2. `embedding_version`
3. `cluster_version`
4. `labeler_version`

This enables:

1. reproducibility
2. rollback
3. A/B comparisons

## 5. Failure Handling

1. Extraction failure on one report should not fail whole batch.
2. Persist per-report error records and retry budget.
3. Cluster run should fail fast on invalid config but keep last successful published snapshot active.

## 6. Publish Model

Use draft/published snapshots:

1. Build full result into a draft `issue_snapshot`.
2. Run validation checks.
3. Publish atomically by flipping active snapshot pointer.
