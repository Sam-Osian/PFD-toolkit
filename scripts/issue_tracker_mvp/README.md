# Issue Tracker Pipeline

## Recommended v3 pipeline

`build_issue_index.py` is the current recommended pipeline. It is designed for
one expensive overnight pass and repeatable, model-free downstream analysis:

1. A concerns-first structured Ollama request extracts and normalises all issues.
   Reports that reach the issue cap receive one continuation request.
2. Every completed report is appended immediately to a JSONL checkpoint.
3. Restarting with the same `--run-dir` skips successful reports and retries failures.
   A run lock prevents two extractors from using the same run directory.
4. Embeddings and grouping run only after extraction completes.
5. Similarity links use evidence from different reports.
6. Average-linkage splitting breaks broad connected components and reduces graph chaining.
7. A sub-issue is `recurring` only at three distinct reports, `emerging` at two,
   and `isolated` at one.
8. `03_issue_report_presence.csv` contains one row per report/sub-issue and is
   the intended input for instant UI scope queries.

The controlled extraction contract is `issue_schema_v3.json`. It separates four
different kinds of classification that v2 conflated: broad service sectors,
cross-cutting issue themes, concrete service contexts, and populations at risk.
It replaces the loose `failure_mode` label with a compact `failure_state`, and
requires the operational `process_stage` separately. `evidence_quote` is the only
model-extracted evidence field; `evidence_section` is derived by the validator.
Report metadata, including the URL, is stored once in `01_reports.csv` and linked
to occurrences by `report_key`.

The difficult-set results and the evidence behind the final vocabulary and
prompt rules are recorded in
`django_workbench_v02/docs/internal/ISSUE_TRACKER_MVP/SCHEMA_V3_EVALUATION.md`.

### Actor/object normalization

`normalize_issue_occurrences.py` enriches an existing v3 occurrence file without
re-extracting report text. It preserves the original evidence, issue identity,
and facets; stores the former sentence as `canonical_issue_original`; adds the
neutral free-text fields `responsible_actor_role` and `issue_object`; and replaces
`canonical_issue` with an actor- and object-explicit formulation. `issue_object`
can describe an action, information item, service, policy, system, equipment,
environment, decision, or duty. It deliberately does not assume that an issue is
medical or that every object is a safeguard.

Normalization is batched, resumable, and auditable. The normalized actor and
object are included once through the canonical sentence used for embedding; they
are neither repeated as separately weighted embedding facets nor used as hard
clustering gates. The prompt specifically preserves the difference
between a person's voluntary disengagement and a service failing to contact,
follow up, respond, escalate, or make reasonable adjustments.

Run the difficult set first:

```bash
.venv/bin/python scripts/issue_tracker_mvp/normalize_issue_occurrences.py \
  --input-csv artifacts/issue_index_v3_difficult_runs/run_20260720_125016/01_issue_occurrences.csv \
  --batch-size 12 \
  --workers 2 \
  --model gemma4:26b
```

The default output is `01_issue_occurrences_normalized.csv` beside the input,
with a JSONL checkpoint, metrics file, and a deterministic
`01_issue_occurrences_normalized_review_queue.csv`. Use that normalized CSV with
`build_issue_index.py --stage index --occurrences-csv ...`; the indexer will
automatically use the revised canonical sentence, actor role, and issue object.
For a smaller pilot, add `--limit 24`; `--selection-csv` also accepts a CSV of
`issue_id` or `report_key` values.

### Normalize and dual-index an existing extraction

An expensive report extraction does not need to be repeated to adopt the new
representation. Normalize its occurrence CSV, then index the normalized output.
`--embedding-mode auto` detects `canonical_issue_original` and creates normalized,
original, and weighted blended embedding caches in one model load. The default
blend is 44% original and 56% normalized. The blended array
remains `02_issue_embeddings.npy`, so tuning, repair, and prototype assignment
continue to work unchanged.

```bash
RUN_DIR="artifacts/issue_index_v3_1500/run_YYYYMMDD_HHMMSS"

.venv/bin/python scripts/issue_tracker_mvp/normalize_issue_occurrences.py \
  --input-csv "$RUN_DIR/01_issue_occurrences.csv" \
  --batch-size 12 \
  --workers 2 \
  --model gemma4:26b

.venv/bin/python scripts/issue_tracker_mvp/build_issue_index.py \
  --stage index \
  --run-dir "$RUN_DIR" \
  --occurrences-csv "$RUN_DIR/01_issue_occurrences_normalized.csv" \
  --embedding-mode auto \
  --original-view-weight 0.44 \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --no-label-subissues \
  --top-k 40 \
  --edge-similarity 0.84 \
  --split-similarity 0.872 \
  --min-centroid-similarity 0.83 \
  --min-recurring-reports 3
```

The normalization checkpoint makes the first command resumable. The three
embedding caches make subsequent threshold tuning model-free:
`02_issue_embeddings_original.npy`, `02_issue_embeddings_normalized.npy`, and the
blended `02_issue_embeddings.npy`. `02_embedding_occurrences.csv` records the
exact normalized rows aligned to those arrays; tuning, review repair, and
prototype assignment prefer this snapshot automatically.

### Mechanical smoke run

Start Ollama and run a small batch before committing to the complete archive:

```bash
.venv/bin/python scripts/issue_tracker_mvp/build_issue_index.py \
  --stage extract \
  --subset-size 25 \
  --output-dir artifacts/issue_index_v3_smoke \
  --model gemma4:26b \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --no-label-subissues
```

Check `01_extraction_metrics.json`, `01_extraction_failures.csv`, and a sample
of `01_issue_occurrences.csv`. Evidence-quote validity is checked automatically.

### Complete overnight run

```bash
.venv/bin/python scripts/issue_tracker_mvp/build_issue_index.py \
  --subset-size 0 \
  --output-dir artifacts/issue_index_v3 \
  --model gemma4:26b \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --min-recurring-reports 3 \
  --top-k 40 \
  --edge-similarity 0.84 \
  --split-similarity 0.872 \
  --min-centroid-similarity 0.83
```

The command prints its timestamped run directory at start-up. To resume an
interrupted run, pass that exact directory:

```bash
.venv/bin/python scripts/issue_tracker_mvp/build_issue_index.py \
  --run-dir artifacts/issue_index_v3/run_YYYYMMDD_HHMMSS \
  --subset-size 0 \
  --model gemma4:26b \
  --embedding-model Qwen/Qwen3-Embedding-8B
```

To rebuild grouping later without any Ollama extraction calls:

```bash
.venv/bin/python scripts/issue_tracker_mvp/build_issue_index.py \
  --stage index \
  --run-dir artifacts/issue_index_v3/run_YYYYMMDD_HHMMSS \
  --no-label-subissues \
  --edge-similarity 0.90 \
  --split-similarity 0.92
```

If `02_issue_embeddings.npy` remains in the run directory, the index stage also
reuses embeddings.

### Compare grouping configurations

Use the deterministic tuning command before changing production defaults. It
reuses `01_issue_occurrences.csv` and `02_issue_embeddings.npy`, makes no Ollama
calls, and does not overwrite the existing index:

```bash
.venv/bin/python scripts/issue_tracker_mvp/tune_issue_index.py \
  --run-dir artifacts/issue_index_v1/run_YYYYMMDD_HHMMSS
```

The default comparison keeps recurrence at three distinct reports and evaluates:

1. `baseline`: `top_k=12`, edge `0.88`, split `0.90`, centroid `0.86`.
2. `recommended`: `top_k=40`, edge `0.84`, split `0.872`, centroid `0.83`,
   with a 44% original / 56% normalized dual-view blend.
3. `high_recall`: `top_k=40`, edge `0.82`, split `0.86`, centroid `0.82`.

Outputs are written to `<run-dir>/05_tuning/`. Each configuration gets group,
assignment, recurring-group, near-duplicate, and human-review CSVs. The review
queue samples the largest groups, lowest-cohesion groups, and groups on the
recurrence boundary, with blank assessment columns for reviewers.

Pass `--config NAME:TOP_K:EDGE:SPLIT:CENTROID` one or more times to compare
different settings. Production defaults should only change after the candidate
review queue satisfies the coherence and over-merge gates.

### Audit recurrence quality and missed links

Build a deterministic, stratified audit from the exact indexed rows and blended
embedding snapshot:

```bash
.venv/bin/python scripts/issue_tracker_mvp/build_recurring_issue_audit.py \
  --run-dir artifacts/issue_index_v3/run_YYYYMMDD_HHMMSS
```

The default audit writes 80 recurring groups (large, recurrence-boundary,
facet-conflict, and random strata) plus 70 non-recurring nearest-neighbour pairs
to `<run-dir>/07_quality_audit/`. It distinguishes already-correct two-report
emerging groups from genuine missed links. After recording complete decisions in
`group_decisions.json` and `missed_link_decisions.json`, validate and score them:

```bash
.venv/bin/python scripts/issue_tracker_mvp/score_recurring_issue_audit.py \
  --audit-dir artifacts/issue_index_v3/run_YYYYMMDD_HHMMSS/07_quality_audit
```

The scorer rejects incomplete or duplicate decision coverage and reports
coherent-as-is, correctable membership, overmerge, false-recurrence, and
missed-link results. Because the sample deliberately over-represents risky
groups and near-threshold pairs, its percentages are diagnostic rather than
population-weighted estimates.

### Apply reviewed group repairs

After every recurring candidate has been reviewed, apply its explicit merge and
member-level split constraints without model calls:

```bash
.venv/bin/python scripts/issue_tracker_mvp/repair_reviewed_issue_groups.py \
  --run-dir artifacts/issue_index_v1/run_YYYYMMDD_HHMMSS
```

The default paths read `05_tuning/recommended_full_review.csv` and
`05_tuning/recommended_split_constraints.json`, then write isolated outputs to
`06_review_repair/`. The command validates complete review coverage, unknown
duplicate targets, duplicate or missing split members, and reports constraint
satisfaction. This is an auditable human-curation step, not a replacement for
evaluation of general grouping behaviour on future snapshots.

### Assign ungrouped occurrences to reviewed prototypes

After review repair, a conservative second pass can recover occurrences that
the mutual-neighbour graph left ungrouped:

```bash
.venv/bin/python scripts/issue_tracker_mvp/assign_issue_prototypes.py \
  --run-dir artifacts/issue_index_v3/run_YYYYMMDD_HHMMSS
```

The defaults require a shared issue theme, cosine similarity of at least
`0.845`, a `0.03` lead over the second-best prototype, and evidence from a new
report. In the v1 manual audit this retained 16 promotions, 14 of which were
correct (87.5% precision); one correct assignment duplicated an existing recurring
type. Treat promoted groups as a review queue, not automatically accepted
production types.

### v3 extraction fields

- `canonical_issue`: the reusable problem statement used for grouping.
- `evidence_quote`: a short, exact excerpt that supports that issue.
- `failure_state`: how the relevant action or safeguard failed, such as
  `omitted`, `delayed`, `inadequate`, `ambiguous`, or `unverified`.
- `process_stage`: the operational activity that failed, such as assessment,
  handover, maintenance, manufacture, or regulation.
- `service_sectors`, `issue_themes`, `service_contexts`, and
  `populations_at_risk`: independent multi-valued facets rather than one
  overloaded domain label.
- `communication_direction`, `responsible_actor_type`, and
  `responsible_actor_text`: communication and accountability facets.
- `concern_status`: whether the concern describes a historical failure, current
  system gap, prospective risk, or a recommendation whose underlying gap had to
  be inferred conservatively.

`issue_statement_original`, `source_span`, and `essential_qualifier_text` are
retired. Their useful roles are covered by `evidence_quote` and a sufficiently
specific `canonical_issue`; retaining all three created ambiguous, duplicated
text fields.

### v3 outputs

1. `manifest.json`: versioned configuration and schema/prompt hashes.
2. `01_extraction_checkpoint.jsonl`: append-only resumable raw extraction records.
3. `01_reports.csv`: report catalogue and stable identities.
4. `01_issue_occurrences.csv`: validated structured occurrence index.
5. `01_extraction_failures.csv`: reports that exhausted their retry budget.
6. `01_extraction_metrics.json`: extraction and evidence-quote diagnostics.
7. `02_issue_embeddings.npy`: cached normalised embeddings.
8. `02_mutual_neighbour_edges.csv`: cross-report similarity evidence.
9. `03_label_checkpoint.jsonl`: resumable labels for recurring sub-issues.
10. `03_subissues.csv`: labels, facets, recurrence status and counts.
11. `03_issue_assignments.csv`: occurrence-to-sub-issue assignments.
12. `03_issue_report_presence.csv`: unique report/sub-issue bridge for UI queries.
13. `03_occurrences_indexed.csv`: display-ready occurrences with assignments.
14. `03_parent_groups.csv`: broad subject-domain navigation.
15. `03_subissue_timeseries_month.csv`: distinct-report monthly time series.
16. `04_index_metrics.json`: summary diagnostics.

Validation warnings in `01_issue_occurrences.csv` belong only to that occurrence;
they are not copied from other issues in the same report. Evidence quotes must be
exact report text. The validator can conservatively recover an exact matching
sentence when the model uses an ellipsis or makes a minor transcription error,
but it does not accept loose semantic paraphrases as evidence.

## Previous experimental script

This script builds an issue-level snapshot from report text:

1. Gemma 4 extraction (`think: false`) -> concise issue sentences per report.
2. Embedding + analysis mode:
   - `similarity` (default): cosine-threshold grouping with LLM labels.
   - `cluster`: BERTopic-style UMAP + HDBSCAN with strict post-filtering.
3. CSV/JSON outputs designed for fast hyperparameter iteration.

## Script

- `scripts/issue_tracker_mvp/build_issue_snapshot.py`

## Install dependencies

```bash
uv add scikit-learn umap-learn sentence-transformers ollama
```

## First run (500 report subset)

```bash
UV_LINK_MODE=copy uv run python scripts/issue_tracker_mvp/build_issue_snapshot.py \
  --subset-size 500 \
  --gemma-model gemma4:26b \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --analysis-mode similarity \
  --keep-runs 0
```

## Faster reruns for tuning (skip extraction)

After first run, reuse `01_report_issues.csv`:

```bash
UV_LINK_MODE=copy uv run python scripts/issue_tracker_mvp/build_issue_snapshot.py \
  --issues-csv-input artifacts/issue_tracker_mvp/run_YYYYMMDD_HHMMSS/01_report_issues.csv \
  --analysis-mode similarity \
  --similarity-threshold 0.82 \
  --similarity-top-k 12 \
  --similarity-min-group-size 3 \
  --keep-runs 0 \
  --no-label-clusters
```

Use `--no-label-clusters` during tuning sweeps to save time, then run once more with labels enabled.
When `--no-label-clusters` is set, group labels will be generic (`Group 0`, `Group 1`, ...).

## Similarity controls

These are active in `--analysis-mode similarity`:

1. `--similarity-threshold` (default `0.82`)
   - edge threshold for linking issue sentences.
2. `--similarity-top-k` (default `12`)
   - number of nearest neighbors retained per issue.
3. `--similarity-min-group-size` (default `3`)
   - connected components smaller than this become ungrouped.

## Cluster controls (fallback mode)

These are active in `--analysis-mode cluster`:

1. `--min-assignment-similarity` (default `0.72`)
2. `--min-cluster-median-similarity` (default `0.76`)
3. `--min-cluster-size-final` (default `8`)

## Core output files

Each run writes to: `artifacts/issue_tracker_mvp/run_<timestamp>/`

1. `01_report_issues.csv`
   - `report_url` + one generated issue sentence per row.
2. `01_report_issue_lists.csv`
   - report-level sentence bundles for quick manual review.
3. `02_similarity_neighbors.csv` (similarity mode)
   - top cosine-neighbor rows per issue above threshold.
4. `02_similarity_grouped_issues.csv` (similarity mode)
   - issue rows with `similarity_group_id` and optional LLM label.
5. `03_similarity_group_summary.csv` (similarity mode)
   - one row per similarity group with label, size, date span, sample issues.
6. `03_similarity_group_examples.csv` (similarity mode)
   - representative sentences per group with centroid similarity.
7. `03_similarity_ungrouped_issues.csv` (similarity mode)
   - issue rows not assigned to a group.
8. `04_similarity_metrics.csv` and `04_similarity_metrics.json` (similarity mode)
9. `05_similarity_recommendations.txt` (similarity mode)
10. `02_clustered_issues.csv`, `03_cluster_summary.csv`, `03_cluster_examples.csv`, `03_unclustered_issues.csv`, `04_tuning_metrics.*` (cluster mode)

## Notes

- The extractor is strict JSON and deterministic (`temperature=0.0`).
- Ollama reasoning is explicitly disabled with `"think": false`.
- If `gemma4:26b` is not your local tag, pass your exact model name via `--gemma-model`.
- Cleanup behavior:
  - Full extraction runs (no `--issues-csv-input`) prune old `run_*` directories using `--keep-runs`.
  - Analysis-only runs (`--issues-csv-input`) keep all `01_*` issue sentence files and remove older analysis outputs (`02_*`, `03_*`, `04_*`, `05_*_recommendations.txt`).
