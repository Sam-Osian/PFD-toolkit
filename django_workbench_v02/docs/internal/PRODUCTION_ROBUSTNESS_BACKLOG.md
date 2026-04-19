# Production Robustness Backlog

Status: Planned  
Last updated: 2026-04-19

This file captures architecture decisions to return to after immediate release tasks.

## 1. Queue Reliability Policy

Decide and implement:

1. Retry rules by failure class (network/transient vs validation/configuration).
2. Max retry count and backoff profile.
3. Idempotent artifact write strategy so retries cannot duplicate results.

## 2. Observability Baseline

Define minimum production telemetry:

1. Structured logs for web, worker, notification dispatcher.
2. Error tracking integration (for example Sentry).
3. Alerting thresholds for service health and run-failure spikes.

## 3. Artifact Durability Standard

Lock production storage policy:

1. Object storage as default backend for artifacts.
2. Retention and cleanup policy for expired artifacts.
3. Recovery process for missing/corrupted artifact pointers.

## 4. Credential Lifecycle Policy

Harden encrypted credential operations:

1. Credential revocation and rotation process.
2. App encryption key rotation runbook.
3. Incident response steps if credential exposure is suspected.

## 5. Operational Safety

Define operations controls:

1. Postgres backup cadence and restore drill.
2. Migration rollout and rollback procedure per release.
3. Service recovery SOP for failed deploy or bad env change.

## 6. Abuse and Cost Controls

Introduce guardrails:

1. Per-user and per-workspace run limits.
2. Concurrency caps per workspace.
3. Endpoint rate limits for queueing/auth-sensitive routes.
