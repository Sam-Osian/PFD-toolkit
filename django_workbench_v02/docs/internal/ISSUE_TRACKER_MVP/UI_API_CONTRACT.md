# UI and API Contract (MVP)

## 1. Core Screens

1. **Issue Cluster List**
   1. Label
   2. Cluster size
   3. Last seen date
   4. Trend sparkline

2. **Issue Cluster Detail**
   1. Label + description
   2. Time series (week/month/year)
   3. Representative issue sentences
   4. Linked reports table

3. **Unclustered Issues**
   1. Issue sentence list with report links
   2. Filter by date range/area/receiver

4. **Similarity Explorer (Beta)**
   1. Pick seed issue
   2. Similarity threshold control
   3. Preview included issues and report count

## 2. API Endpoints (example shape)

1. `GET /issues/clusters/`
   1. query: `snapshot_id`, filters, pagination
   2. returns: cluster summaries

2. `GET /issues/clusters/{cluster_id}/`
   1. returns: cluster metadata + sample issues

3. `GET /issues/clusters/{cluster_id}/timeseries/`
   1. query: `period=week|month|year`
   2. returns: `{period_start, count}`

4. `GET /issues/clusters/{cluster_id}/reports/`
   1. returns report list + issue sentence links

5. `GET /issues/unclustered/`
   1. returns unclustered issue sentences with report links

6. `POST /issues/similarity/preview/`
   1. input: seed issue id + threshold
   2. returns matching issue ids/counts

## 3. Response Guarantees

1. All issue rows include traceability:
   1. `report_id`
   2. `report_url`
   3. source issue sentence
2. All cluster rows include:
   1. `snapshot_id`
   2. version fields

## 4. Permissions

Follow existing workspace/public visibility model:

1. Public context: only public-safe issue aggregates.
2. Workspace context: issue views filtered to workspace-scoped report set.
3. Internal ops/admin: full snapshot diagnostics and version metadata.
