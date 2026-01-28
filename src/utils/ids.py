from __future__ import annotations

from typing import Dict, List, Tuple

from src.core.state import Evidence, RawEvidence, RawSource, Source


def dedup_sources(raw_sources: List[RawSource]) -> List[Tuple[str, str, str]]:
    """Return unique (url,title,snippet) by exact URL."""
    seen: Dict[str, Tuple[str, str, str]] = {}
    for s in raw_sources:
        url = (s.get("url") or "").strip()
        if not url:
            continue
        if url not in seen:
            title = (s.get("title") or url).strip()
            snippet = (s.get("snippet") or "").strip()
            seen[url] = (url, title, snippet)
    return list(seen.values())


def assign_source_ids(items: List[Tuple[str, str, str]]) -> List[Source]:
    sources: List[Source] = []
    for i, (url, title, snippet) in enumerate(items, start=1):
        sources.append({"sid": f"S{i}", "url": url, "title": title, "snippet": snippet})
    return sources


def assign_evidence_ids(raw_evidence: List[RawEvidence], url_to_sid: Dict[str, str]) -> List[Evidence]:
    evidence: List[Evidence] = []
    idx = 1
    for ev in raw_evidence:
        url = (ev.get("url") or "").strip()
        if not url or url not in url_to_sid:
            continue
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        evidence.append({
            "eid": f"E{idx}",
            "sid": url_to_sid[url],
            "url": url,
            "title": (ev.get("title") or url).strip(),
            "section": (ev.get("section") or "General").strip(),
            "text": text[:1200],
        })
        idx += 1
    return evidence
