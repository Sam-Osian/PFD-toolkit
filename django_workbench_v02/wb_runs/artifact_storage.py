from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from django.conf import settings
from django.utils import timezone

from .models import ArtifactStorageBackend


class ArtifactStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredArtifactFile:
    storage_backend: str
    storage_uri: str
    size_bytes: int | None


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _configured_artifact_backend() -> str:
    raw = str(getattr(settings, "ARTIFACT_STORAGE_BACKEND", ArtifactStorageBackend.FILE)).strip().lower()
    if raw not in {
        ArtifactStorageBackend.FILE,
        ArtifactStorageBackend.OBJECT_STORAGE,
    }:
        raise ArtifactStorageError(f"Unsupported ARTIFACT_STORAGE_BACKEND={raw!r}.")
    enforce_object_storage = _as_bool(
        getattr(settings, "ARTIFACT_ENFORCE_OBJECT_STORAGE_IN_PRODUCTION", True),
        default=True,
    )
    if (
        enforce_object_storage
        and not bool(getattr(settings, "DEBUG", False))
        and not bool(getattr(settings, "TESTING", False))
    ):
        if raw != ArtifactStorageBackend.OBJECT_STORAGE:
            raise ArtifactStorageError(
                "Production requires ARTIFACT_STORAGE_BACKEND=object_storage."
            )
    return raw


def _object_storage_client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
        raise ArtifactStorageError(
            "boto3 is required for ARTIFACT_STORAGE_BACKEND=object_storage."
        ) from exc

    endpoint_url = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_ENDPOINT_URL", "") or "").strip() or None
    region_name = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_REGION", "") or "").strip() or None
    access_key = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_ACCESS_KEY_ID", "") or "").strip() or None
    secret_key = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_SECRET_ACCESS_KEY", "") or "").strip() or None
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _object_storage_bucket() -> str:
    bucket = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_BUCKET", "") or "").strip()
    if not bucket:
        raise ArtifactStorageError(
            "ARTIFACT_OBJECT_STORAGE_BUCKET must be configured for object storage backend."
        )
    return bucket


def _object_key_for_artifact(*, run, artifact_type: str, source_path: Path) -> str:
    prefix = str(getattr(settings, "ARTIFACT_OBJECT_STORAGE_PREFIX", "workbench-artifacts") or "").strip("/")
    stamp = timezone.now().strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{prefix}/workspace_{run.workspace_id}/investigation_{run.investigation_id}/"
        f"run_{run.id}/{artifact_type}/{stamp}_{uuid4().hex}_{source_path.name}"
    )


def store_artifact_file(*, source_path: Path, run, artifact_type: str) -> StoredArtifactFile:
    if not source_path.is_file():
        raise ArtifactStorageError(f"Artifact file does not exist: {source_path}")
    source_size = source_path.stat().st_size

    backend = _configured_artifact_backend()
    if backend == ArtifactStorageBackend.FILE:
        return StoredArtifactFile(
            storage_backend=ArtifactStorageBackend.FILE,
            storage_uri=str(source_path),
            size_bytes=source_size,
        )

    bucket = _object_storage_bucket()
    client = _object_storage_client()
    key = _object_key_for_artifact(run=run, artifact_type=artifact_type, source_path=source_path)
    extra_args = {}
    guessed_type, _ = mimetypes.guess_type(source_path.name)
    if guessed_type:
        extra_args["ContentType"] = guessed_type
    if extra_args:
        client.upload_file(str(source_path), bucket, key, ExtraArgs=extra_args)
    else:
        client.upload_file(str(source_path), bucket, key)

    delete_local = _as_bool(
        getattr(settings, "ARTIFACT_STORAGE_DELETE_LOCAL_AFTER_UPLOAD", True),
        default=True,
    )
    if delete_local:
        source_path.unlink(missing_ok=True)

    return StoredArtifactFile(
        storage_backend=ArtifactStorageBackend.OBJECT_STORAGE,
        storage_uri=f"s3://{bucket}/{key}",
        size_bytes=source_size,
    )


def _parse_s3_uri(storage_uri: str) -> tuple[str, str]:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ArtifactStorageError(f"Invalid S3 artifact URI: {storage_uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def open_artifact_for_download(artifact) -> tuple[object, str]:
    if artifact.storage_backend == ArtifactStorageBackend.FILE:
        if not artifact.storage_uri:
            raise ArtifactStorageError("File artifact has no storage URI.")
        path = Path(artifact.storage_uri)
        if not path.is_file():
            raise ArtifactStorageError("File artifact was not found on disk.")
        return path.open("rb"), path.name

    if artifact.storage_backend == ArtifactStorageBackend.OBJECT_STORAGE:
        if not artifact.storage_uri:
            raise ArtifactStorageError("Object-storage artifact has no storage URI.")
        bucket, key = _parse_s3_uri(artifact.storage_uri)
        client = _object_storage_client()
        response = client.get_object(Bucket=bucket, Key=key)
        filename = Path(key).name or f"{artifact.artifact_type}_{artifact.id}"
        return response["Body"], filename

    raise ArtifactStorageError(
        f"Artifact backend {artifact.storage_backend!r} is not downloadable."
    )
