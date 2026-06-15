# Issue Tracker MVP Script

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
