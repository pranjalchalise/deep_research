# src/utils/ids.py
from __future__ import annotations

import hashlib
from typing import Dict, List, TypedDict
from urllib.parse import urlparse, urlunparse


def stable_hash(s: str) -> str:
    """Create a stable hash for a string (useful for cache keys)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class RawSource(TypedDict):
    url: str
    title: str
    snippet: str


class Source(TypedDict):
    sid: str  # S1
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict):
    url: str
    title: str
    section: str
    text: str


class Evidence(TypedDict):
    eid: str  # E1
    sid: str  # S1
    url: str
    title: str
    section: str
    text: str


def normalize_url(url: str) -> str:
    """
    Normalize URL for deduplication:
    - Strip trailing slashes
    - Lowercase scheme and host
    - Remove default ports
    - Remove www. prefix
    - Remove fragments
    """
    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = (parsed.scheme or "https").lower()
        host = (parsed.hostname or "").lower()

        # Remove www. prefix
        if host.startswith("www."):
            host = host[4:]

        # Remove default ports
        port = parsed.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        # Rebuild netloc
        netloc = host
        if port:
            netloc = f"{host}:{port}"
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo = f"{userinfo}:{parsed.password}"
            netloc = f"{userinfo}@{netloc}"

        # Strip trailing slash from path (but keep "/" for root)
        path = parsed.path.rstrip("/") or "/"

        # Remove fragment, keep query
        normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        return normalized

    except Exception:
        return url


def dedup_sources(raw_sources: List[RawSource]) -> List[RawSource]:
    """
    Deduplicate sources by normalized URL.
    Keeps the longest snippet when duplicates are found.
    """
    # Map: normalized_url -> (original_url, RawSource)
    seen: Dict[str, RawSource] = {}
    url_map: Dict[str, str] = {}  # normalized -> original (first seen)

    for s in raw_sources or []:
        url = (s.get("url") or "").strip()
        if not url:
            continue

        norm_url = normalize_url(url)
        title = (s.get("title") or "").strip() or url
        snippet = (s.get("snippet") or "").strip()

        if norm_url in seen:
            # Keep longer snippet
            existing = seen[norm_url]
            if len(snippet) > len(existing.get("snippet") or ""):
                seen[norm_url]["snippet"] = snippet
            # Keep longer title if current is just URL
            if existing.get("title") == existing.get("url") and title != url:
                seen[norm_url]["title"] = title
        else:
            url_map[norm_url] = url
            seen[norm_url] = {"url": url, "title": title, "snippet": snippet}

    return list(seen.values())


def assign_source_ids(sources: List[RawSource]) -> List[Source]:
    """
    Assign stable sequential ids: S1, S2, ...
    """
    out: List[Source] = []
    for i, s in enumerate(sources or [], start=1):
        out.append(
            {
                "sid": f"S{i}",
                "url": s["url"],
                "title": s.get("title", "") or s["url"],
                "snippet": s.get("snippet", "") or "",
            }
        )
    return out


def _text_fingerprint(text: str) -> str:
    """
    Create a rough fingerprint for text deduplication.
    Normalizes whitespace and lowercases for comparison.
    """
    return " ".join(text.lower().split())


def assign_evidence_ids(raw_evidence: List[RawEvidence], url_to_sid: Dict[str, str]) -> List[Evidence]:
    """
    Convert RawEvidence -> Evidence with ids (E1, E2, ...) and attach sid by url.
    - Drops evidence whose url doesn't exist in url_to_sid
    - Deduplicates evidence with identical/very similar text from same source
    - Also tries to match evidence URLs via normalization
    """
    # Build normalized URL lookup
    norm_to_sid: Dict[str, str] = {}
    for url, sid in url_to_sid.items():
        norm_to_sid[normalize_url(url)] = sid

    out: List[Evidence] = []
    # Track (sid, text_fingerprint) to avoid duplicates
    seen_evidence: Dict[tuple, Evidence] = {}

    for e in raw_evidence or []:
        url = (e.get("url") or "").strip()
        if not url:
            continue

        # Try exact match first, then normalized
        sid = url_to_sid.get(url)
        if not sid:
            sid = norm_to_sid.get(normalize_url(url))
        if not sid:
            # evidence from a URL we didn't keep as a source
            continue

        title = (e.get("title") or "").strip() or url
        section = (e.get("section") or "").strip() or "General"
        text = (e.get("text") or "").strip()
        if not text:
            continue

        # Check for duplicate evidence (same source, similar text)
        fingerprint = _text_fingerprint(text)
        key = (sid, fingerprint)

        if key in seen_evidence:
            # Keep the longer version
            existing = seen_evidence[key]
            if len(text) > len(existing.get("text") or ""):
                existing["text"] = text[:1200]
            continue

        evidence: Evidence = {
            "eid": "",  # assigned below
            "sid": sid,
            "url": url,
            "title": title,
            "section": section,
            "text": text[:1200],
        }
        seen_evidence[key] = evidence

    # Assign sequential IDs
    for j, ev in enumerate(seen_evidence.values(), start=1):
        ev["eid"] = f"E{j}"
        out.append(ev)

    return out


__all__ = ["normalize_url", "dedup_sources", "assign_source_ids", "assign_evidence_ids"]
