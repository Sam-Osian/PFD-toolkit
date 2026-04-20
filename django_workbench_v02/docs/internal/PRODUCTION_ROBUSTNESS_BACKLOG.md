# Production Robustness Backlog

Status: In progress  
Last updated: 2026-04-20

This file captures architecture decisions to return to after immediate release tasks.

## 1. Queue Reliability Policy

Implemented baseline:

1. Retry rules by failure class (transient failures only).
2. Max retry count and backoff profile (env-configurable).
3. Requeue-on-retry uses the same run record, avoiding duplicate run records.

Remaining:

1. Optional dedicated run-attempt model for richer retry analytics.

## 2. Observability Baseline

Define minimum production telemetry:

1. Structured logs for web, worker, notification dispatcher.
2. Error tracking integration (for example Sentry).
3. Alerting thresholds for service health and run-failure spikes.

## 3. Artifact Durability Standard

Implemented baseline:

1. Object storage as default backend for artifacts.
2. Retention policy for run artifacts (`ARTIFACT_RETENTION_DAYS`).

Remaining:

1. Recovery process for missing/corrupted artifact pointers.
2. Optional checksum verification/enforcement pipeline.

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

Implemented baseline guardrails:

1. Per-user and per-workspace run limits.
2. Concurrency caps per user and global.
3. Endpoint rate limits for queueing routes (user and IP).
