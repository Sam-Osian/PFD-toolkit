# ADR-0002: Artifact Storage Strategy

Date: 2026-04-19  
Status: Accepted

## Context

Workbench produces run artifacts that must be downloadable, durable, and compatible with Railway deployment constraints.

The project needs:

1. A local-development path with minimal setup.
2. A production-safe path that does not rely on ephemeral local disk.
3. A migration path that avoids rewriting core run logic.

## Decision

Adopt a storage abstraction with two active backends in v0.2:

1. `file` backend for local/development use.
2. `object_storage` backend for production (S3-compatible).

Implementation details:

1. Persisted metadata on `RunArtifact` includes `storage_backend` and `storage_uri`.
2. Worker persistence writes through `store_artifact_file(...)`.
3. Downloads and export reads use `open_artifact_for_download(...)`.
4. Backend choice is controlled by environment (`ARTIFACT_STORAGE_BACKEND` and object-storage settings).

## Consequences

Positive:

1. Run execution logic stays backend-agnostic.
2. Production durability is possible without redesigning the run model.
3. Exports and downloads operate consistently across backends.

Tradeoffs:

1. Object storage introduces credential and bucket policy operations overhead.
2. Local-file behavior can diverge from production if backend parity is not tested.
