from __future__ import annotations

from typing import Any, Dict, List

from src.core.state import AgentState, Page, Source
from src.tools.http_fetch import read_cache, write_cache, fetch_url, html_to_text

def fetcher_node(state: AgentState) -> Dict[str, Any]:
    sources: List[Source] = state.get("sources") or []
    cfg_cache_dir = state.get("cache_dir") or ".cache_v7/pages"
    timeout_s = float(state.get("request_timeout_s") or 12.0)

    pages: List[Page] = []

    for s in sources:
        url = s["url"]
        title = s.get("title", "") or url
        sid = s["sid"]

        cached = read_cache(cfg_cache_dir, url)
        if cached:
            pages.append({"sid": sid, "url": url, "title": title, "text": cached, "fetched_from_cache": True})
            continue

        html = fetch_url(url, timeout_s=timeout_s)
        text = html_to_text(html)
        if text:
            write_cache(cfg_cache_dir, url, text)
        pages.append({"sid": sid, "url": url, "title": title, "text": text, "fetched_from_cache": False})

    return {"pages": pages}
