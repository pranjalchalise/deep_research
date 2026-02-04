"""
Source and evidence identity management: URL normalization, deduplication, and ID assignment.

Every source gets a stable ID (S1, S2, ...) and every evidence chunk gets an
ID (E1, E2, ...) linked back to its source. Dedup uses normalized URLs so
http://www.example.com/ and https://example.com are treated as the same source.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, TypedDict
from urllib.parse import urlparse, urlunparse


def stable_hash(s: str) -> str:
    """Deterministic short hash for cache keys and fingerprinting."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class RawSource(TypedDict):
    url: str
    title: str
    snippet: str


class Source(TypedDict):
    sid: str       # e.g. "S1"
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict):
    url: str
    title: str
    section: str
    text: str


class Evidence(TypedDict):
    eid: str       # e.g. "E1"
    sid: str       # cross-ref to parent Source
    url: str
    title: str
    section: str
    text: str


def normalize_url(url: str) -> str:
    """Canonicalize a URL so that trivial variants (www, trailing slash, fragments) collapse."""
    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)

        scheme = (parsed.scheme or "https").lower()
        host = (parsed.hostname or "").lower()

        if host.startswith("www."):
            host = host[4:]

        port = parsed.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        netloc = host
        if port:
            netloc = f"{host}:{port}"
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo = f"{userinfo}:{parsed.password}"
            netloc = f"{userinfo}@{netloc}"

        path = parsed.path.rstrip("/") or "/"
        normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        return normalized

    except Exception:
        return url


def dedup_sources(raw_sources: List[RawSource]) -> List[RawSource]:
    """Collapse duplicates by normalized URL, keeping the longest snippet of each."""
    seen: Dict[str, RawSource] = {}
    url_map: Dict[str, str] = {}

    for s in raw_sources or []:
        url = (s.get("url") or "").strip()
        if not url:
            continue

        norm_url = normalize_url(url)
        title = (s.get("title") or "").strip() or url
        snippet = (s.get("snippet") or "").strip()

        if norm_url in seen:
            existing = seen[norm_url]
            if len(snippet) > len(existing.get("snippet") or ""):
                seen[norm_url]["snippet"] = snippet
            if existing.get("title") == existing.get("url") and title != url:
                seen[norm_url]["title"] = title
        else:
            url_map[norm_url] = url
            seen[norm_url] = {"url": url, "title": title, "snippet": snippet}

    return list(seen.values())


def assign_source_ids(sources: List[RawSource]) -> List[Source]:
    """Tag each source with a sequential ID (S1, S2, ...)."""
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
    """Normalize whitespace and case so near-identical texts produce the same key."""
    return " ".join(text.lower().split())


def assign_evidence_ids(raw_evidence: List[RawEvidence], url_to_sid: Dict[str, str]) -> List[Evidence]:
    """
    Assign E1, E2, ... IDs and link each evidence chunk to its parent source.

    Drops evidence from unknown URLs, deduplicates near-identical text from
    the same source, and tries both exact and normalized URL matching.
    """
    norm_to_sid: Dict[str, str] = {}
    for url, sid in url_to_sid.items():
        norm_to_sid[normalize_url(url)] = sid

    out: List[Evidence] = []
    seen_evidence: Dict[tuple, Evidence] = {}  # keyed by (sid, fingerprint)

    for e in raw_evidence or []:
        url = (e.get("url") or "").strip()
        if not url:
            continue

        sid = url_to_sid.get(url)
        if not sid:
            sid = norm_to_sid.get(normalize_url(url))
        if not sid:
            continue

        title = (e.get("title") or "").strip() or url
        section = (e.get("section") or "").strip() or "General"
        text = (e.get("text") or "").strip()
        if not text:
            continue

        fingerprint = _text_fingerprint(text)
        key = (sid, fingerprint)

        if key in seen_evidence:
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

    for j, ev in enumerate(seen_evidence.values(), start=1):
        ev["eid"] = f"E{j}"
        out.append(ev)

    return out


__all__ = ["normalize_url", "dedup_sources", "assign_source_ids", "assign_evidence_ids"]
