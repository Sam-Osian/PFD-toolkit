# Performance Log Review Checklist (Weekly)

Status: Active  
Last updated: 2026-04-20

## Purpose

Keep MVP performance healthy using log-based review, without introducing full APM yet.

## Scope

Review web service logs for:

1. `slow_request ...`
2. `slow_query ...`

## Weekly Routine (15-20 minutes)

## 1) Confirm instrumentation is enabled

Check Railway `web` variables:

1. `PERF_REQUEST_LOGGING_ENABLED=True`
2. `PERF_SLOW_REQUEST_MS=800`
3. `PERF_SLOW_QUERY_MS=200`

## 2) Pull recent logs

Collect at least the last 7 days (or since last deploy).

## 3) Identify top slow request paths

From `slow_request` lines, list:

1. path
2. count
3. p95-ish duration (or highest observed if no percentile tooling)
4. average query_count

Goal: find repeated hotspots, not one-off spikes.

## 4) Identify repeated slow SQL patterns

From `slow_query` lines, group by normalized SQL shape and path.

Record:

1. SQL pattern
2. affected endpoint/path
3. frequency
4. worst duration

## 5) Classify each hotspot

For each hotspot, classify primary cause:

1. DB lookup/index issue
2. N+1 query pattern
3. heavy response serialization/template rendering
4. external I/O in request path
5. expected one-off operational event

## 6) Create a tiny action list

Limit to max 3 items per week:

1. one high-impact fix (do now),
2. one medium fix (next sprint),
3. one watch item (no change yet).

## 7) Re-check after changes

After deploying a fix:

1. compare same endpoint before/after over at least 24h,
2. verify no regression on unrelated endpoints.

## Logging Threshold Tuning Rules

Only tune thresholds if needed:

1. If too noisy, raise gradually (e.g., request 800->1000ms; query 200->300ms).
2. If too quiet, lower gradually (e.g., request 800->600ms; query 200->150ms).
3. Avoid changing both in the same week unless logs are unusable.

## Escalation Triggers

Escalate to deeper work (or APM) if any are true:

1. Same endpoint appears in top slow list for 2+ consecutive weeks.
2. Slow logs spike after each deploy with no clear root cause.
3. User-visible lag is reported but logs are inconclusive.

## Change Log Template

Keep a short note each week in your ops notes:

1. Date range reviewed
2. Top 3 slow endpoints
3. Top 3 SQL patterns
4. Actions taken
5. Result after deploy
