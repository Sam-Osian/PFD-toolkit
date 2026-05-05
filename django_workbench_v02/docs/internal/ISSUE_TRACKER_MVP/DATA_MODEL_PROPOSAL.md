# Data Model Proposal

## 1. New Entities (MVP)

1. `IssueSnapshot`
   1. `id`
   2. `status` (`draft`, `published`, `archived`)
   3. `source_dataset_signature`
   4. `extractor_version`
   5. `embedding_version`
   6. `cluster_version`
   7. `labeler_version`
   8. `created_at`, `published_at`

2. `IssueSentence`
   1. `id`
   2. `snapshot_id`
   3. `report_id`
   4. `issue_text` (single sentence)
   5. `source_span_hint` (optional; source excerpt reference)
   6. `embedding_vector_ref` (blob/object pointer, not inline text column)
   7. `created_at`

3. `IssueCluster`
   1. `id`
   2. `snapshot_id`
   3. `cluster_key` (stable within snapshot)
   4. `label`
   5. `description` (optional 1–2 lines)
   6. `size`
   7. `is_noise` (false for normal clusters)

4. `IssueAssignment`
   1. `id`
   2. `snapshot_id`
   3. `issue_sentence_id`
   4. `cluster_id` (nullable for unclustered/noise)
   5. `assignment_score` (optional confidence/probability)

5. `IssueClusterTimeSeries` (materialized aggregate)
   1. `id`
   2. `snapshot_id`
   3. `cluster_id`
   4. `period` (`week`/`month`/`year`)
   5. `period_start`
   6. `count`

## 2. Key Indexes

1. `IssueSentence(snapshot_id, report_id)`
2. `IssueAssignment(snapshot_id, cluster_id)`
3. `IssueCluster(snapshot_id, size desc)`
4. `IssueClusterTimeSeries(snapshot_id, cluster_id, period, period_start)`

## 3. Relationship Notes

1. One report can produce many issue sentences.
2. One cluster can contain issue sentences from many reports.
3. Unclustered issue sentences remain queryable as first-class records.

## 4. Backward Compatibility

Keep existing theme/collection tables untouched during MVP.

Use feature flag to toggle "Issues (Beta)" views:

1. `ISSUE_TRACKER_ENABLED`
2. `ISSUE_TRACKER_SNAPSHOT_ID` (optional forced pin)
