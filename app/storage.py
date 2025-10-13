"""Storage backend abstractions for persisting generated assets."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

import boto3
from botocore.client import BaseClient

from .settings import settings


class Storage(Protocol):
    """Interface for persisting binary data and exposing access URLs."""

    def write_bytes(self, path: str, data: bytes) -> None:
        """Persist raw bytes to ``path`` relative to the backend's root."""

    def write_file(self, src_local: str, dest_path: str) -> None:
        """Persist a local file ``src_local`` to ``dest_path`` within the backend."""

    def url_for(self, path: str) -> str:
        """Return an externally accessible URL for ``path``."""


class LocalStorage:
    """Filesystem storage rooted at the provided base directory."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        relative = path.lstrip('/').lstrip('\\')
        return self.base_dir / relative

    def write_bytes(self, path: str, data: bytes) -> None:
        destination = self._resolve(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def write_file(self, src_local: str, dest_path: str) -> None:
        destination = self._resolve(dest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        src_path = Path(src_local).resolve()
        if src_path == destination:
            return
        shutil.copyfile(src_path, destination)

    def url_for(self, path: str) -> str:
        return str(self._resolve(path))


class S3Storage:
    """Amazon S3 backed storage using presigned URLs for access."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        url_expiration: int = 7 * 24 * 60 * 60,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket name is required for S3Storage")
        self.bucket = bucket
        self.prefix = prefix.strip('/').strip('\\')
        self._client: BaseClient = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._url_expiration = max(int(url_expiration), 60)

    def _object_key(self, path: str) -> str:
        normalized = path.lstrip('/').lstrip('\\')
        if self.prefix:
            return f"{self.prefix}/{normalized}"
        return normalized

    def write_bytes(self, path: str, data: bytes) -> None:
        key = self._object_key(path)
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def write_file(self, src_local: str, dest_path: str) -> None:
        key = self._object_key(dest_path)
        self._client.upload_file(src_local, self.bucket, key)

    def url_for(self, path: str) -> str:
        key = self._object_key(path)
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self._url_expiration,
        )


@lru_cache()
def get_storage(default_local_base_dir: Optional[str] = None) -> Storage:
    """Instantiate the configured storage backend."""

    backend = settings.storage_backend.lower()
    if backend == "local":
        base_dir = default_local_base_dir or str(Path("jobs").resolve())
        return LocalStorage(base_dir)
    if backend == "s3":
        return S3Storage(
            bucket=settings.s3_bucket or "",
            prefix=settings.s3_prefix,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
    raise RuntimeError(f"Unsupported storage backend: {settings.storage_backend}")
