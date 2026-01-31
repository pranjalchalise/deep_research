# src/tools/tavily.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_tavily import TavilySearch

from src.utils.cache import get_search_cache, make_search_key


# Lane -> (topic, include_domains)
# topic can be "general" or "news" (Tavily native)
# include_domains filters to specific sites
LANE_CONFIG: Dict[str, Dict[str, Any]] = {
    "docs": {
        "topic": "general",
        "include_domains": [
            "docs.python.org",
            "developer.mozilla.org",
            "docs.microsoft.com",
            "learn.microsoft.com",
            "cloud.google.com",
            "docs.aws.amazon.com",
            "kubernetes.io",
            "docs.docker.com",
            "react.dev",
            "nextjs.org",
            "vuejs.org",
            "angular.io",
            "docs.rs",
            "pkg.go.dev",
            "readthedocs.io",
        ],
    },
    "papers": {
        "topic": "general",
        "include_domains": [
            "arxiv.org",
            "scholar.google.com",
            "semanticscholar.org",
            "acm.org",
            "ieee.org",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "researchgate.net",
            "pubmed.ncbi.nlm.nih.gov",
        ],
    },
    "news": {
        "topic": "news",  # Tavily has native news topic
        "include_domains": [],  # let Tavily handle news sources
    },
    "forums": {
        "topic": "general",
        "include_domains": [
            "reddit.com",
            "news.ycombinator.com",
            "stackoverflow.com",
            "stackexchange.com",
            "dev.to",
            "hashnode.com",
            "discourse.org",
            "lobste.rs",
        ],
    },
    "code": {
        "topic": "general",
        "include_domains": [
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "gist.github.com",
            "sourcegraph.com",
        ],
    },
    "general": {
        "topic": "general",
        "include_domains": [],  # no filtering
    },
}


def make_search(
    max_results: int = 5,
    lane: Optional[str] = None,
) -> TavilySearch:
    """
    Create a TavilySearch configured for the given lane.

    Lanes: docs, papers, news, forums, code, general (default)
    """
    config = LANE_CONFIG.get(lane or "general", LANE_CONFIG["general"])
    topic = config.get("topic", "general")
    include_domains = config.get("include_domains", [])

    kwargs: Dict[str, Any] = {
        "max_results": max_results,
        "topic": topic,
    }

    if include_domains:
        kwargs["include_domains"] = include_domains

    # search_depth="advanced" helps a lot for people queries.
    # Some versions of langchain_tavily support this kwarg; if yours doesn't,
    # just remove search_depth and it'll still work.
    try:
        return TavilySearch(**kwargs, search_depth="advanced")
    except TypeError:
        return TavilySearch(**kwargs)


def normalize(raw: Any) -> List[Dict[str, str]]:
    """
    Normalize Tavily output to:
    [{"url":..., "title":..., "snippet":...}, ...]
    """
    results = []
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    out: List[Dict[str, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        title = str(r.get("title", "")).strip() or url
        snippet = str(r.get("content") or r.get("snippet") or "").strip()
        out.append({"url": url, "title": title, "snippet": snippet})
    return out


def cached_search(
    query: str,
    max_results: int = 5,
    lane: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Execute a search with caching support.

    Args:
        query: Search query
        max_results: Maximum results to return
        lane: Search lane (docs, papers, news, forums, code, general)
        use_cache: Whether to use cache (default True)
        cache_dir: Custom cache directory (default: .cache_v7/search)

    Returns:
        Normalized list of search results
    """
    lane = lane or "general"
    cache_key = make_search_key(query, lane, max_results)

    # Try cache first
    if use_cache:
        cache = get_search_cache(cache_dir) if cache_dir else get_search_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # Execute search
    search = make_search(max_results=max_results, lane=lane)
    raw = search.invoke({"query": query})
    results = normalize(raw)

    # Cache results
    if use_cache:
        cache = get_search_cache(cache_dir) if cache_dir else get_search_cache()
        cache.set(cache_key, results)

    return results
