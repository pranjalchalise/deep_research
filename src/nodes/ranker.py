from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Source
from src.utils.scoring import quality_score
from src.utils.json_utils import parse_json_object

RELEVANCE_FILTER_SYSTEM = """You filter sources to only include those about the EXACT target entity.

Given:
- TARGET: The specific person/entity being researched
- CONTEXT: Identifying information
- SOURCES: List of source titles and snippets

Return ONLY sources that are clearly about the EXACT target entity.
Exclude sources about:
- Different people with similar names
- Places/organizations with similar names
- People at different institutions than the target

Return JSON:
{
  "relevant_urls": ["url1", "url2"],
  "excluded": [{"url": "...", "reason": "different person - X at Y instead of target"}]
}

Be strict - when uncertain, exclude.
"""

RERANK_SYSTEM = """You are a source reranker.

Given a list of candidate sources with url/title/snippet/lane/score, return ONLY JSON list of urls
ordered from best to worst.

Preferences:
- Prefer primary sources (official docs, original papers, official repos).
- Prefer diversity across lanes and domains (avoid 5 similar blogs).
- Prefer sources that directly answer the user question.
- Prefer recency when user asks "latest" or when topic changes fast.
"""

def _filter_sources_by_entity(
    sources: List[Dict],
    primary_anchor: str,
    context_terms: List[str],
    llm: Any,
) -> List[Dict]:
    """Second-pass filter to remove sources about wrong entities."""
    if not sources or not primary_anchor:
        return sources

    context_str = ", ".join(context_terms) if context_terms else "none"

    sources_text = "\n".join([
        f"- URL: {s.get('url', '')}\n  Title: {s.get('title', '')}\n  Snippet: {s.get('snippet', '')[:200]}"
        for s in sources[:20]  # Limit to avoid token issues
    ])

    resp = llm.invoke([
        SystemMessage(content=RELEVANCE_FILTER_SYSTEM),
        HumanMessage(content=f"TARGET: {primary_anchor}\nCONTEXT: {context_str}\n\nSOURCES:\n{sources_text}"),
    ])

    result = parse_json_object(resp.content, default={})
    relevant_urls = set(result.get("relevant_urls", []))

    # If LLM returned empty, fall back to all sources
    if not relevant_urls:
        return sources

    filtered = [s for s in sources if s.get("url") in relevant_urls]

    # Log what was excluded
    excluded = result.get("excluded", [])
    if excluded:
        print(f"[ranker] filtered {len(excluded)} irrelevant sources")

    return filtered if filtered else sources


def ranker_node(state: AgentState) -> Dict[str, Any]:
    """
    Rank and filter sources after reducer.

    For person queries, applies relevance filtering to remove
    sources about different entities with similar names.
    """
    print("[ranker] === RANKER NODE CALLED ===")

    user_q = state["messages"][-1].content
    sources = state.get("sources") or []
    evidence = state.get("evidence") or []

    print(f"[ranker] sources count: {len(sources)}, evidence count: {len(evidence)}")

    if not sources:
        print("[ranker] no sources, returning empty")
        return {}

    # Get entity info for filtering
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")
    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []

    print(f"[ranker] query_type={query_type}, primary_anchor={primary_anchor}, sources={len(sources)}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Second-pass relevance filter for person queries
    if query_type == "person" and primary_anchor:
        print(f"[ranker] applying entity filter for '{primary_anchor}'")
        original_count = len(sources)
        sources = _filter_sources_by_entity(sources, primary_anchor, anchor_terms, llm)
        filtered_count = original_count - len(sources)

        if filtered_count > 0:
            print(f"[ranker] filtered {filtered_count} irrelevant sources, {len(sources)} remaining")

            # Also filter evidence to only keep evidence from remaining sources
            valid_sids = {s["sid"] for s in sources}
            evidence = [e for e in evidence if e.get("sid") in valid_sids]
            print(f"[ranker] evidence after filter: {len(evidence)}")

    # Sort by quality score
    for s in sources:
        if "score" not in s:
            s["score"] = quality_score(
                s.get("url", ""),
                s.get("title", ""),
                s.get("snippet", ""),
                s.get("lane", "general"),
                s.get("published_date", ""),
            )

    sources_sorted = sorted(sources, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # Limit to top N
    top_n = state.get("rerank_top_n") or 15
    sources_final = sources_sorted[:top_n]

    # Filter evidence to only keep evidence from final sources
    valid_sids = {s["sid"] for s in sources_final}
    evidence_final = [e for e in evidence if e.get("sid") in valid_sids]

    return {"sources": sources_final, "evidence": evidence_final}
