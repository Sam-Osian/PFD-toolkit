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

## 8. July 2026 cached-embedding comparison

The 1,500-report run at `run_20260714_171206` extracted 6,996 occurrences. A
deterministic comparison produced:

| Configuration | Recurring types | Recurring occurrences | All assigned | Median recurring cohesion |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`12 / .88 / .90 / .86`) | 41 | 164 | 967 (13.82%) | .9736 |
| Recommended (`40 / .84 / .87 / .83`) | 99 | 432 | 2,379 (34.01%) | .9624 |
| High recall (`40 / .82 / .86 / .82`) | 137 | 600 | 3,211 (45.90%) | .9586 |

The values in parentheses are `top_k / edge / split / centroid`. The baseline's
very high cohesion and 6,029 ungrouped occurrences indicate under-grouping. The
recommended configuration is the current review candidate; it is not yet the
production default.

Review `05_tuning/recommended_review_queue.csv` and record `coherent`,
`over_merged`, `near_duplicate_of`, and notes. Apply the existing launch gates:
at least 80% coherent and no more than 10% obvious over-merges. Also inspect
`recommended_near_duplicates.csv` before accepting the configuration.

The completed 60-group review found 53 coherent groups (88.33%), seven obvious
over-merges (11.67%), and 19 groups with credible duplicate links. The candidate
passes coherence but narrowly fails the over-merge gate, so it must not replace
the production defaults yet. The simultaneous over-merging and fragmentation
supports an explicit constrained merge/split pass rather than another global
threshold reduction.

The review was subsequently extended to all 99 recurring candidates. It found
87 coherent groups (87.88%), 12 over-merges (12.12%), and 25 groups with credible
duplicate links. The review-guided repair satisfied all 16 confirmed merge pairs
and all 12 member-level split constraints. Sixty-four unchanged groups preserved
their membership, and all 38 changed outputs passed a second review for coherence,
over-merging, over-splitting, and repair appropriateness.

The repaired candidate contains 76 recurring, eight emerging, and 18 isolated
groups. The reduced recurring count is expected: several apparent three-report
types were mixtures of distinct one-report failures. This repaired candidate is
the input to the next recall experiment; the raw 99-group configuration remains
unsuitable as a production default.

## 9. Follow-up algorithm experiments

After selecting a tuned baseline, evaluate these as separate changes so their
effects remain measurable:

1. Assign currently ungrouped occurrences to accepted group centroids above a
   calibrated threshold.
2. Compare strict mutual-neighbour linkage with a radius or shared-neighbour
   fallback for dense issue families.
3. Calibrate thresholds by subject domain when a global threshold creates
   systematically different recall.
4. Test embedding text without noisy free-text facets and strengthen extraction
   canonicalisation.
