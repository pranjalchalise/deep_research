# src/nodes/search_worker.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, PlanQuery, RawEvidence, RawSource
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk, should_skip_url
from src.utils.json_utils import parse_json_array, parse_json_object
from src.utils.llm import create_chat_model


EXTRACT_SYSTEM = """You extract concise, high-signal evidence from source content.

Given the content from a web page and the research query, extract 2-4 evidence items.

Return ONLY a JSON array of:
{"text": "1-3 sentences of factual evidence directly relevant to the query"}

Rules:
- Evidence text must be short, specific, and factual (no fluff)
- Only include information that directly answers or relates to the query
- Prefer specific numbers, dates, findings over vague statements
- Do not invent or hallucinate information
"""

EXTRACT_FROM_SNIPPET_SYSTEM = """You extract concise, high-signal evidence from search snippets.

Given search result snippets and a target section, return 2-4 evidence items.

Return ONLY a JSON array of:
{"url": "...", "title": "...", "text": "1-3 sentences of evidence"}

Rules:
- Evidence text must be short and factual
- Don't invent URLs - use only provided URLs
- Prefer results that directly answer the query
"""

RELEVANCE_CHECK_SYSTEM = """You check if search results are about the EXACT TARGET entity or a DIFFERENT entity.

Given:
- TARGET ENTITY: The specific person we're researching
- CONTEXT: Identifying information (school, role, location, etc.)
- SEARCH RESULTS: Titles and snippets

STRICT RULES:
1. Names must match EXACTLY - similar names are DIFFERENT people (e.g., "Kim" vs "Kim-Lee", "Alex" vs "Alexander")
2. Context must match - if target is at school X, results about someone at school Y are DIFFERENT
3. Places/organizations with similar names to people are DIFFERENT
4. When uncertain, EXCLUDE the result

Return ONLY valid JSON:
{
  "relevant_indices": [0, 2, 3],  // indices of results that ARE about the EXACT target
  "reasoning": "Brief explanation"
}

Be VERY strict: only include results you are CONFIDENT are about the exact same person with matching name AND context.
"""


def _strip_site(query: str) -> str:
    """Remove site:domain patterns from query."""
    return re.sub(r"\bsite:[^\s]+\b", "", query).strip()


def _filter_relevant_results(
    results: List[Dict[str, str]],
    primary_anchor: str,
    context_terms: List[str],
    llm: Any,
) -> List[Dict[str, str]]:
    """
    Filter search results to only include those actually about the target entity.

    This prevents using results about different entities with similar names.
    """
    if not results or not primary_anchor:
        return results

    # Build context string
    context_str = ", ".join(context_terms) if context_terms else "no additional context"

    # Format results for LLM
    results_text = "\n".join([
        f"[{i}] {r['title']}\n    Snippet: {r.get('snippet', '')[:300]}"
        for i, r in enumerate(results)
    ])

    prompt = f"""TARGET ENTITY: {primary_anchor}
CONTEXT: {context_str}

SEARCH RESULTS:
{results_text}

Which results are actually about the TARGET entity (not a different person/place with similar name)?"""

    resp = llm.invoke([
        SystemMessage(content=RELEVANCE_CHECK_SYSTEM),
        HumanMessage(content=prompt),
    ])

    analysis = parse_json_object(resp.content, default={})
    relevant_indices = analysis.get("relevant_indices", [])

    # Filter results
    filtered = []
    for i in relevant_indices:
        if isinstance(i, int) and 0 <= i < len(results):
            filtered.append(results[i])

    # If LLM filtered everything, do a simple keyword check as fallback
    if not filtered and results:
        # Check if primary anchor appears in title or snippet
        primary_lower = primary_anchor.lower()
        for r in results:
            title_lower = r.get("title", "").lower()
            snippet_lower = r.get("snippet", "").lower()
            combined = title_lower + " " + snippet_lower

            # Require EXACT primary anchor to appear (not partial match)
            # Split name into parts and check all parts appear
            name_parts = primary_lower.split()
            all_parts_present = all(part in combined for part in name_parts if len(part) > 1)

            if all_parts_present:
                # Also check for at least one context term
                has_context = not context_terms or any(
                    ctx.lower() in combined for ctx in context_terms
                )
                if has_context:
                    filtered.append(r)

    return filtered


def _fallback_evidence(results: List[Dict[str, str]], section: str) -> List[RawEvidence]:
    """Create evidence from snippets when LLM extraction fails."""
    ev: List[RawEvidence] = []
    for r in results[:3]:
        snip = (r.get("snippet") or "").strip()
        if not snip:
            continue
        ev.append({
            "url": r["url"],
            "title": r["title"],
            "section": section,
            "text": snip[:900],
        })
    return ev


def _extract_evidence_from_chunks(
    llm: Any,
    url: str,
    title: str,
    chunks: List[str],
    query: str,
    section: str,
    max_evidence: int = 3,
) -> List[RawEvidence]:
    """Extract evidence from page content chunks using LLM."""
    if not chunks:
        return []

    # Combine chunks (limit total size)
    combined = "\n\n---\n\n".join(chunks[:3])
    if len(combined) > 8000:
        combined = combined[:8000] + "..."

    resp = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=f"Query: {query}\nTarget section: {section}\n\nSource content:\n{combined}\n\nReturn JSON only.")
    ])

    txt = resp.content.strip()
    ev_raw = parse_json_array(txt, default=[])

    evidence: List[RawEvidence] = []
    for item in ev_raw[:max_evidence]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text or len(text) < 20:
            continue
        evidence.append({
            "url": url,
            "title": title,
            "section": section,
            "text": text[:900],
        })

    return evidence


def _extract_evidence_from_snippets(
    llm: Any,
    results: List[Dict[str, str]],
    query: str,
    section: str,
) -> List[RawEvidence]:
    """Extract evidence from search snippets (fallback when fetch fails)."""
    prompt_blob = "\n\n".join([
        f"- TITLE: {r['title']}\n  URL: {r['url']}\n  SNIPPET: {(r.get('snippet') or '')[:600]}"
        for r in results
    ])

    resp = llm.invoke([
        SystemMessage(content=EXTRACT_FROM_SNIPPET_SYSTEM),
        HumanMessage(content=f"Query: {query}\nTarget section: {section}\n\nSearch results:\n{prompt_blob}\n\nReturn JSON only.")
    ])

    txt = resp.content.strip()
    ev_raw = parse_json_array(txt, default=[])

    evidence: List[RawEvidence] = []
    for item in ev_raw[:4]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        if not url or not text:
            continue
        if not title:
            match = next((r for r in results if r["url"] == url), None)
            title = match["title"] if match else url
        evidence.append({
            "url": url,
            "title": title,
            "section": section,
            "text": text[:900],
        })

    return evidence


def worker_node(state: AgentState) -> Dict[str, Any]:
    """
    Search worker that:
    1. Searches via Tavily (with caching)
    2. FILTERS results to only include those about the target entity
    3. Fetches full page content for relevant results
    4. Extracts evidence from full content (or snippets as fallback)

    Invoked via Send with injected `query_item`.
    """
    qi: PlanQuery = state["query_item"]
    max_results = state.get("tavily_max_results") or state.get("max_results") or 6

    q = (qi.get("query") or "").strip()
    section = (qi.get("section") or "General").strip()
    lane = (qi.get("lane") or "general").strip()

    # Get primary anchor and context for relevance filtering
    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []

    # Get query type from discovery - only apply relevance filtering for person queries
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")

    # Config from state
    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir")
    timeout_s = state.get("request_timeout_s", 12.0)
    chunk_chars = state.get("chunk_chars", 3500)
    chunk_overlap = state.get("chunk_overlap", 350)
    max_chunks = state.get("max_chunks_per_source", 4)
    evidence_per_source = state.get("evidence_per_source", 3)
    fast_mode = state.get("fast_mode", True)

    # In fast mode, fetch fewer pages
    pages_to_fetch = 2 if fast_mode else 4

    # ---- Search with retries (using cache) ----
    results: List[Dict[str, str]] = []
    tried: List[str] = []

    def run_search(query: str, search_lane: str) -> List[Dict[str, str]]:
        return cached_search(
            query=query,
            max_results=max_results,
            lane=search_lane,
            use_cache=use_cache,
            cache_dir=f"{cache_dir}/search" if cache_dir else None,
        )

    # try 1: original query with lane filtering
    tried.append(f"{q} [lane={lane}]")
    results = run_search(q, lane)

    # try 2: same query but fall back to general (no domain filter)
    if not results and lane != "general":
        tried.append(f"{q} [lane=general]")
        results = run_search(q, "general")

    # try 3: remove site: operators and search general
    if not results:
        q2 = _strip_site(q)
        if q2 and q2 != q:
            tried.append(f"{q2} [lane=general]")
            results = run_search(q2, "general")

    # try 4: broaden for people queries
    if not results:
        q3 = f"{q} profile biography research publications"
        tried.append(f"{q3} [lane=general]")
        results = run_search(q3, "general")

    # If no results, return empty
    if not results:
        print(f"[worker] no results for query='{q}' tried={tried}")
        return {"raw_sources": [], "raw_evidence": [], "done_workers": 1}

    # ---- RELEVANCE FILTERING ----
    # Filter out results that are about different entities with similar names
    # ONLY apply for person queries - concept/technical queries don't need this
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    if primary_anchor and query_type == "person":
        original_count = len(results)
        results = _filter_relevant_results(results, primary_anchor, anchor_terms, llm)
        filtered_count = len(results)

        if filtered_count < original_count:
            print(f"[worker] filtered {original_count - filtered_count} irrelevant results for '{primary_anchor}'")

        if not results:
            print(f"[worker] no RELEVANT results for query='{q}' (all {original_count} were about different entities)")
            return {"raw_sources": [], "raw_evidence": [], "done_workers": 1}

    # Build raw sources from FILTERED results only
    raw_sources: List[RawSource] = [
        {"url": r["url"], "title": r["title"], "snippet": (r.get("snippet") or "")[:700]}
        for r in results
    ]

    # ---- Fetch full pages and extract evidence ----
    all_evidence: List[RawEvidence] = []
    fetched_count = 0

    # Try to fetch top pages
    for r in results[:pages_to_fetch]:
        url = r["url"]
        title = r["title"]

        # Skip unfetchable URLs
        if should_skip_url(url):
            continue

        # Fetch and chunk the page
        chunks = fetch_and_chunk(
            url=url,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            timeout_s=timeout_s,
            use_cache=use_cache,
            cache_dir=f"{cache_dir}/pages" if cache_dir else None,
        )

        if chunks:
            fetched_count += 1
            # Extract evidence from full content
            ev = _extract_evidence_from_chunks(
                llm=llm,
                url=url,
                title=title,
                chunks=chunks,
                query=q,
                section=section,
                max_evidence=evidence_per_source,
            )
            all_evidence.extend(ev)

    # If we couldn't fetch any pages, fall back to snippet-based extraction
    if fetched_count == 0 and results:
        print(f"[worker] fetch failed for all URLs, falling back to snippets for query='{q}'")
        all_evidence = _extract_evidence_from_snippets(llm, results, q, section)

    # Final fallback if LLM extraction failed
    if not all_evidence and results:
        all_evidence = _fallback_evidence(results, section)

    return {"raw_sources": raw_sources, "raw_evidence": all_evidence, "done_workers": 1}
