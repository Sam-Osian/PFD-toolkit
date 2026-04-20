# Performance MVP Backlog

Status: Active (MVP-first)  
Last updated: 2026-04-20

## Goal

Ship feature/UI parity first, while adding only the minimum performance guardrails needed to detect regressions early.

## What We Implemented Now

1. Request timing instrumentation middleware.
2. Slow-query instrumentation per web request.
3. Environment-driven thresholds (no code change needed to tune in Railway).

## Environment Variables (Web Service)

Add these to Railway `web` service:

1. `PERF_REQUEST_LOGGING_ENABLED=True`
2. `PERF_SLOW_REQUEST_MS=800`
3. `PERF_SLOW_QUERY_MS=200`
4. `PERF_ADD_RESPONSE_TIMING_HEADER=False`

Notes:

1. Set `PERF_ADD_RESPONSE_TIMING_HEADER=True` temporarily for live debugging.
2. Keep it `False` by default in production to avoid extra response metadata.

## How To Read Logs

Middleware writes warnings with:

1. `slow_request ... duration_ms=... query_count=... slow_query_count=...`
2. `slow_query ... duration_ms=... sql=...`

Use these as your primary signal for where latency is coming from (view path vs database query).

## MVP Deferrals (Do After UI Phase)

1. Cache strategy (Redis, per-view + fragment cache map).
2. Gunicorn tuning pass (`workers`, `threads`, `max-requests`, `timeout`) with measured load.
3. Full APM integration and alert thresholds.
4. Rate limiting / bot controls on public routes.
5. Deploy-time benchmark regression checks.

## Next Performance Decisions (After This Step)

1. Index plan:
   - identify highest-frequency filters/sorts from real slow-query logs;
   - add targeted DB indexes and verify with `EXPLAIN`.
2. Observability tier:
   - choose between log-only + periodic review, or APM product (e.g., Sentry Performance).
