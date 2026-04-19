# ADR-0001: Export Bundle Architecture

Date: 2026-04-19  
Status: Accepted

## Context

Users need a single downloadable export that can include outputs from multiple run types (`filter`, `themes`, `extract`) without coupling the web UI to local filesystem assumptions.

The system also needs:

1. Workspace-level permission enforcement.
2. Compatibility with multiple artifact storage backends.
3. Deterministic packaging behavior suitable for async server-side execution.

## Decision

Use a run-driven export workflow that materializes a zip bundle as a first-class `RunArtifact` (`artifact_type=bundle_export`):

1. Export is executed by the async run worker (`run_type=export`).
2. Export selection is performed from existing `RunArtifact` records, scoped to workspace/investigation.
3. Bundle creation uses the artifact storage abstraction (`open_artifact_for_download`) so inputs may come from local file or object storage.
4. The zip includes `manifest.json` describing included and skipped artifacts.
5. The resulting export bundle is persisted using the same artifact storage abstraction as other outputs.

## Consequences

Positive:

1. One architecture for all artifact backends.
2. Reproducible packaging semantics in worker context.
3. Strong auditability via run events and run artifacts.

Tradeoffs:

1. Export latency depends on number/size of selected artifacts.
2. Export failures must be handled through run lifecycle states, not synchronous UI feedback.
