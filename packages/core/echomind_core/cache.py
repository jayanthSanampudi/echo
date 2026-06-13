"""Pluggable cache backend: in-memory (diskcache) or Redis."""

from __future__ import annotations

import json
from typing import Any, Protocol

import diskcache

from echomind_core.config import get_settings


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...


class _DiskCache:
    def __init__(self, directory: str = "./.cache") -> None:
        self._cache = diskcache.Cache(directory)

    def get(self, key: str) -> Any | None:
        v = self._cache.get(key)
        return json.loads(v) if isinstance(v, str) else v

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        v = json.dumps(value) if not isinstance(value, (str, bytes, int, float)) else value
        self._cache.set(key, v, expire=ttl)

    def delete(self, key: str) -> None:
        self._cache.delete(key)


class _RedisCache:
    def __init__(self, url: str) -> None:
        import redis  # imported lazily so redis is optional

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        v = self._client.get(key)
        if v is None:
            return None
        try:
            return json.loads(v)
        except (TypeError, ValueError):
            return v

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        v = value if isinstance(value, str) else json.dumps(value)
        if ttl:
            self._client.setex(key, ttl, v)
        else:
            self._client.set(key, v)

    def delete(self, key: str) -> None:
        self._client.delete(key)


_backend: CacheBackend | None = None


def get_cache() -> CacheBackend:
    global _backend
    if _backend is None:
        s = get_settings()
        _backend = _RedisCache(s.redis_url) if s.cache_backend == "redis" else _DiskCache()
    return _backend
