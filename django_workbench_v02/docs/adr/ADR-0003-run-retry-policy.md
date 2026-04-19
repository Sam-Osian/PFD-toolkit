# ADR-0003: Run Retry Policy

Date: 2026-04-19  
Status: Accepted

## Context

AI/provider workflows can fail transiently (network faults, temporary upstream errors, rate limits).  
At the same time, automatic retries can create duplicate spend or repeated side effects if applied blindly.

Current model has one `InvestigationRun` record per execution request and clear terminal states.

## Decision

Use a conservative retry policy for v0.2 beta:

1. No automatic retries in worker execution path by default.
2. Retries are user-initiated by queueing a new run.
3. Failed/timed-out runs preserve diagnostics (`error_code`, `error_message`, events) for review.
4. Keep architecture ready for future retry-attempt modeling, but do not infer silent retries yet.

## Consequences

Positive:

1. Predictable billing and side-effect behavior.
2. Clear audit trail: one run record equals one execution attempt.
3. Lower risk of hidden repeat failures.

Tradeoffs:

1. Recovery from transient errors is manual in beta.
2. Operators may need follow-up tooling for bulk requeue if failure volume grows.
