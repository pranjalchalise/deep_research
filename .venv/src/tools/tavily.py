from __future__ import annotations

from typing import Any, Dict, List

from langchain_tavily import TavilySearch


def make_search(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results, topic="general")


def normalize(raw: Any) -> List[Dict[str, str]]:
    """
    Normalize Tavily output into list[{url,title,snippet}]
    """
    if raw is None:
        return []
    results = raw.get("results") if isinstance(raw, dict) else raw
    if not isinstance(results, list):
        return []

    out: List[Dict[str, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url", "")).strip()
        title = str(r.get("title", "")).strip() or url
        # Tavily typically uses "content"
        snippet = str(r.get("content", "") or r.get("snippet", "") or "").strip()
        if url:
            out.append({"url": url, "title": title, "snippet": snippet[:800]})
    return out
