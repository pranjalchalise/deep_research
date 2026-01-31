# src/utils/cache.py
"""
File-based caching utilities for search results and fetched pages.

Provides disk-based caching with TTL support to avoid redundant API calls
during development and repeated research queries.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T")

# Default TTL values (in seconds)
DEFAULT_SEARCH_TTL = 3600 * 24  # 24 hours for search results
DEFAULT_PAGE_TTL = 3600 * 24 * 7  # 7 days for fetched pages


def _hash_key(key: str) -> str:
    """Create a safe filename from a cache key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _safe_filename(key: str, prefix: str = "") -> str:
    """Create a safe filename from a cache key with optional prefix."""
    hashed = _hash_key(key)
    if prefix:
        return f"{prefix}_{hashed}.json"
    return f"{hashed}.json"


class FileCache:
    """
    Simple file-based cache with TTL support.

    Usage:
        cache = FileCache(".cache/search")

        # Simple get/set
        cache.set("my_key", {"data": "value"}, ttl=3600)
        result = cache.get("my_key")  # Returns None if expired or missing

        # With default factory
        result = cache.get_or_set("my_key", lambda: expensive_call(), ttl=3600)
    """

    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (0 = no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / _safe_filename(key)

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if a cache entry is expired."""
        ttl = metadata.get("ttl", 0)
        if ttl == 0:
            return False  # No expiration
        created = metadata.get("created", 0)
        return time.time() > created + ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Returns None if key doesn't exist or is expired.
        """
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            if self._is_expired(entry.get("metadata", {})):
                # Clean up expired entry
                path.unlink(missing_ok=True)
                return None

            return entry.get("data")
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = use default, 0 = no expiration)
        """
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
            # Log but don't fail if caching fails
            print(f"[cache] Warning: Failed to cache key '{key[:50]}...': {e}")

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """
        Get value from cache, or compute and cache it.

        Args:
            key: Cache key
            factory: Function to call if cache miss
            ttl: Time-to-live in seconds

        Returns:
            Cached or freshly computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if entry existed."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns count of deleted entries."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except IOError:
                pass
        return count

    def clear_expired(self) -> int:
        """Clear only expired entries. Returns count of deleted entries."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if self._is_expired(entry.get("metadata", {})):
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, IOError, KeyError):
                # Invalid entry, delete it
                try:
                    path.unlink()
                    count += 1
                except IOError:
                    pass
        return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
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


# Global cache instances (lazily initialized)
_search_cache: Optional[FileCache] = None
_page_cache: Optional[FileCache] = None


def get_search_cache(cache_dir: str = ".cache_v7/search") -> FileCache:
    """Get or create the global search results cache."""
    global _search_cache
    if _search_cache is None or str(_search_cache.cache_dir) != cache_dir:
        _search_cache = FileCache(cache_dir, default_ttl=DEFAULT_SEARCH_TTL)
    return _search_cache


def get_page_cache(cache_dir: str = ".cache_v7/pages") -> FileCache:
    """Get or create the global page content cache."""
    global _page_cache
    if _page_cache is None or str(_page_cache.cache_dir) != cache_dir:
        _page_cache = FileCache(cache_dir, default_ttl=DEFAULT_PAGE_TTL)
    return _page_cache


def make_search_key(query: str, lane: str = "general", max_results: int = 5) -> str:
    """Create a cache key for a search query."""
    return f"search:{lane}:{max_results}:{query}"


def make_page_key(url: str) -> str:
    """Create a cache key for a fetched page."""
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
