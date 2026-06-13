"""File storage abstraction. Local disk for dev, S3 for production."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol

from echomind_core.config import get_settings


class StorageBackend(Protocol):
    def save(self, key: str, data: bytes | BinaryIO) -> str: ...
    def load(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def path(self, key: str) -> str | None: ...


class _LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"path traversal attempted: {key}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, key: str, data: bytes | BinaryIO) -> str:
        p = self._resolve(key)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            with p.open("wb") as fh:
                fh.write(data.read())
        return str(p)

    def load(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def path(self, key: str) -> str | None:
        p = self._resolve(key)
        return str(p) if p.exists() else None


class _S3Storage:
    def __init__(self, bucket: str, endpoint: str | None = None) -> None:
        import boto3  # lazy import

        self._bucket = bucket
        self._client = boto3.client("s3", endpoint_url=endpoint)

    def save(self, key: str, data: bytes | BinaryIO) -> str:
        body = data if isinstance(data, bytes) else data.read()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
        return f"s3://{self._bucket}/{key}"

    def load(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def path(self, key: str) -> str | None:
        return None  # remote — no local path


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is None:
        s = get_settings()
        if s.storage_backend == "s3" and s.s3_bucket:
            _backend = _S3Storage(s.s3_bucket, s.s3_endpoint)
        else:
            _backend = _LocalStorage(s.storage_path)
    return _backend
