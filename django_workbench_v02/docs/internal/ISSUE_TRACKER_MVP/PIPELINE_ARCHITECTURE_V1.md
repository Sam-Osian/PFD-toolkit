# Issue Index Pipeline Architecture v1

Status: Implemented offline pipeline contract  
Last updated: 2026-07-14

## 1. Product invariant

The issue tracker is a precomputed global report index. Selecting reports in
Explore or a workspace must not invoke an LLM. A request intersects stable report
identities with the published `IssueReportPresence` index and counts distinct reports.

## 2. Expensive boundary

The only report-level LLM stage is structured extraction. One call per report returns
all issue occurrences. New extractions use
`scripts/issue_tracker_mvp/issue_schema_v2.json`; the original v1 artefacts remain
immutable for comparison and reproducibility.

The checkpoint is append-only and stores the raw response, parsed issues, warnings,
failure state and completion time. Successful reports are not repeated on resume.

## 3. Deterministic stages

After extraction:

1. Compose embedding text from canonical issue plus selected controlled facets.
2. Generate normalised embeddings once and cache them.
3. Build a nearest-neighbour graph using only cross-report links.
4. Retain mutual links above the configured similarity threshold.
5. Find connected components.
6. Split each component using average-linkage clustering to resist chaining.
7. Reject members below the minimum centroid similarity.
8. Count distinct reports and classify candidate sub-issues:
   - `recurring`: at least three reports
   - `emerging`: two reports
   - `isolated`: one report
9. Generate a stable sub-issue identifier from member issue identifiers.
10. Build one `IssueReportPresence` row per sub-issue/report pair.
11. Aggregate monthly time series using distinct reports.

## 4. Offline entity contract

### IssueOccurrence

One source-grounded issue in one report. It contains original and canonical text,
an exact source span with validation status, controlled facets, report metadata and
version provenance.

### SubIssue

A fine-grained semantic group. It contains recurrence status, distinct-report count,
occurrence count, representative issue, cohesion, dominant facets and date bounds.

### IssueGroup

A broad navigation category derived from `subject_domain`. It is not the primary
recurrence unit.

### IssueReportPresence

The unique bridge `(snapshot, sub_issue, report)`. It prevents multiple issue
statements in one report from inflating recurrence and supports scope queries.

## 5. UI query contract

Given a selected set of report identities:

1. Filter `IssueReportPresence` to those identities.
2. Group by sub-issue.
3. Count distinct reports.
4. Show sub-issues at three or more reports by default.
5. Recalculate scope-specific date bounds, time series and facet distributions.
6. Join the representative occurrence to explain why each report was recalled.

No report extraction, normalisation, cluster labelling or generative model call occurs
in the request path.

## 6. Deliberate exclusions

Commitment, uptake, evidence, blocker, repair and closure motifs are excluded. They
belong to a future response-correspondence or communication-episode model, not the
PFD issue occurrence contract.

## 7. Production integration boundary

The v1 script emits CSV/JSON/NumPy artefacts. Django integration should import a
validated snapshot atomically into indexed database tables. The existing Collections
feature remains untouched until an Issues view reaches equivalent report filtering,
workspace-copy and export coverage.
