# Evaluation and Rollout

## 1. Quality Gates Before User-Facing Launch

1. Extraction validity:
   1. >= 99% JSON parse success after retries.
   2. <= 2% empty output on reports known to contain explicit concerns.
2. Cluster quality (sampled manual review):
   1. >= 80% clusters rated coherent.
   2. <= 10% clusters flagged as obvious over-merge.
3. Label quality:
   1. >= 85% labels judged understandable without opening examples.
4. Traceability:
   1. 100% issue rows link back to source report.

## 2. Operational Metrics

1. Extraction throughput (reports/hour).
2. Clustering run duration and memory.
3. Percent unclustered issue sentences.
4. Snapshot publish success rate.
5. Drift between consecutive snapshots:
   1. cluster count delta
   2. top cluster stability

## 3. Human-in-the-Loop Workflow (lightweight)

1. Weekly review of:
   1. top 50 clusters by size
   2. top 50 unclustered by recurrence signals
2. Allow manual actions:
   1. relabel cluster
   2. mark cluster as split candidate
   3. mark two clusters as merge candidate

## 4. Rollout Sequence

1. Internal only (`ISSUE_TRACKER_ENABLED=false` for users).
2. Beta read-only for selected users.
3. Broad read-only.
4. Default navigation switch.
5. Theme/collection deprecation decision after 2–4 weeks of stable usage.

## 5. Fallback Strategy

If issue pipeline quality degrades:

1. Pin to last good published snapshot.
2. Keep existing collections UX available.
3. Disable new snapshot publish until validation passes.
