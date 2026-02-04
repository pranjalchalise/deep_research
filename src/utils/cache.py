"""
Disk-based cache with TTL for search results and fetched pages.

Saves us from re-hitting Tavily or re-downloading the same URLs on every
run. Each entry is a JSON file keyed by a hash of the query/URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T")

DEFAULT_SEARCH_TTL = 3600 * 24      # 24 hours -- search results go stale fast
DEFAULT_PAGE_TTL = 3600 * 24 * 7    # 7 days -- page content is more stable


def _hash_key(key: str) -> str:
    """Hash a cache key down to a safe, fixed-length filename."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _safe_filename(key: str, prefix: str = "") -> str:
    """Produce a collision-resistant .json filename from a cache key."""
    hashed = _hash_key(key)
    if prefix:
        return f"{prefix}_{hashed}.json"
    return f"{hashed}.json"


class FileCache:
    """
    One-JSON-file-per-entry cache with TTL expiration.

    Each entry stores its creation time and TTL in metadata, so stale
    entries are silently ignored (and cleaned up on next access).
    """

    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / _safe_filename(key)

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        ttl = metadata.get("ttl", 0)
        if ttl == 0:
            return False  # No expiration
        created = metadata.get("created", 0)
        return time.time() > created + ttl

    def get(self, key: str) -> Optional[Any]:
        """Return cached value, or None if missing/expired."""
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            if self._is_expired(entry.get("metadata", {})):
                path.unlink(missing_ok=True)
                return None

            return entry.get("data")
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Write a JSON-serializable value to cache. ttl=0 means no expiration."""
        if ttl is None:
            ttl = self.default_ttl

        path = self._get_path(key)
        entry = {
            "metadata": {
                "key": key,
                "created": time.time(),
                "ttl": ttl,
            },
            "data": value,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        except (IOError, TypeError) as e:
            # Caching is best-effort; don't let it crash the pipeline
            print(f"[cache] Warning: Failed to cache key '{key[:50]}...': {e}")

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """Return cached value or call factory(), cache the result, and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached

        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except IOError:
                pass
        return count

    def clear_expired(self) -> int:
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if self._is_expired(entry.get("metadata", {})):
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, IOError, KeyError):
                try:
                    path.unlink()
                    count += 1
                except IOError:
                    pass
        return count

    def stats(self) -> Dict[str, Any]:
        total = 0
        expired = 0
        total_size = 0

        for path in self.cache_dir.glob("*.json"):
            total += 1
            total_size += path.stat().st_size
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if self._is_expired(entry.get("metadata", {})):
                    expired += 1
            except (json.JSONDecodeError, IOError, KeyError):
                expired += 1

        return {
            "total_entries": total,
            "expired_entries": expired,
            "valid_entries": total - expired,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


_search_cache: Optional[FileCache] = None
_page_cache: Optional[FileCache] = None


def get_search_cache(cache_dir: str = ".cache_v7/search") -> FileCache:
    """Lazily create and return the singleton search cache."""
    global _search_cache
    if _search_cache is None or str(_search_cache.cache_dir) != cache_dir:
        _search_cache = FileCache(cache_dir, default_ttl=DEFAULT_SEARCH_TTL)
    return _search_cache


def get_page_cache(cache_dir: str = ".cache_v7/pages") -> FileCache:
    """Lazily create and return the singleton page cache."""
    global _page_cache
    if _page_cache is None or str(_page_cache.cache_dir) != cache_dir:
        _page_cache = FileCache(cache_dir, default_ttl=DEFAULT_PAGE_TTL)
    return _page_cache


def make_search_key(query: str, lane: str = "general", max_results: int = 5) -> str:
    return f"search:{lane}:{max_results}:{query}"


def make_page_key(url: str) -> str:
    return f"page:{url}"


__all__ = [
    "FileCache",
    "get_search_cache",
    "get_page_cache",
    "make_search_key",
    "make_page_key",
    "DEFAULT_SEARCH_TTL",
    "DEFAULT_PAGE_TTL",
]
