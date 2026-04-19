# Artifact Storage Backend (v0.2)

Status: Implemented  
Last updated: 2026-04-19

## 1. Purpose

Artifact binary storage is abstracted behind `wb_runs/artifact_storage.py` so run execution does not depend on local disk.

Supported backends:

1. `file` (development default)
2. `object_storage` (S3-compatible for production)

## 2. Settings

1. `ARTIFACT_STORAGE_BACKEND` (`file` or `object_storage`)
2. `ARTIFACT_OBJECT_STORAGE_BUCKET`
3. `ARTIFACT_OBJECT_STORAGE_REGION` (optional)
4. `ARTIFACT_OBJECT_STORAGE_ENDPOINT_URL` (optional)
5. `ARTIFACT_OBJECT_STORAGE_ACCESS_KEY_ID` (optional)
6. `ARTIFACT_OBJECT_STORAGE_SECRET_ACCESS_KEY` (optional)
7. `ARTIFACT_OBJECT_STORAGE_PREFIX` (default `workbench-artifacts`)
8. `ARTIFACT_STORAGE_DELETE_LOCAL_AFTER_UPLOAD` (default `true`)

## 3. Runtime Behavior

1. Adapters generate output files under runtime artifact directories.
2. Worker calls `store_artifact_file(...)` to persist output according to configured backend.
3. Stored location is written to `RunArtifact.storage_backend` and `RunArtifact.storage_uri`.
4. Download endpoint resolves the correct backend via `open_artifact_for_download(...)`.

## 4. URI Formats

1. Local file backend: absolute filesystem path.
2. Object storage backend: `s3://<bucket>/<key>`.

## 5. Current Limitations

1. Downloads are proxied through Django for object storage artifacts.
2. Pre-signed direct-download URLs are not implemented yet.
3. Retention cleanup jobs are not yet implemented (handled in next phase step).
