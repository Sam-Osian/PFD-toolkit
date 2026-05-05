# Clustering and Tuning Plan

## 1. Recommended Baseline Stack

1. Sentence embeddings (local model).
2. UMAP dimension reduction.
3. HDBSCAN clustering.
4. BERTopic-style topic representation over clusters.

## 2. Initial Bias

Start with **higher precision / smaller clusters**:

1. Prefer unclustered noise over over-merged clusters.
2. Add optional merge pass later if needed.

## 3. Tuning Axes

Primary knobs:

1. Embedding model choice.
2. UMAP `n_neighbors`, `min_dist`, `n_components`.
3. HDBSCAN `min_cluster_size`, `min_samples`.
4. Similarity threshold for interface-level user grouping.

## 4. Practical Starting Ranges

1. `min_cluster_size`: 8 to 20
2. `min_samples`: 3 to 10
3. `n_neighbors`: 10 to 40
4. `n_components`: 5 to 15

Use narrow sweeps first, then expand only if needed.

## 5. Evaluation Signals

1. Cluster coherence (manual rating on sampled clusters).
2. Purity by nearest-neighbor inspection.
3. Stability across reruns (same config + minor data updates).
4. Fraction of noise points (should be meaningful but not extreme).
5. Over-merge flags (cluster examples show distinct issues mixed together).

## 6. Decision Rule for "small vs broad"

1. If users complain about fragmented issue groups: reduce strictness slightly.
2. If users complain cluster labels are vague/mixed: increase strictness.
3. Treat over-merging as a more severe failure than under-clustering.

## 7. Interface-Level Dynamic Grouping

For user-defined grouping:

1. Let users select a seed issue cluster or issue sentence.
2. Expand by cosine similarity threshold slider.
3. Show preview counts before applying.
4. Keep this ephemeral unless user saves a named grouping.
