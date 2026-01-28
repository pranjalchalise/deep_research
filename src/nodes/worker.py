from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, PlanQuery, RawEvidence, RawSource
from src.tools.tavily import make_search, normalize

EXTRACT_SYSTEM = """You extract concise, high-signal evidence.

Given search results for a query and a target section, return 2-4 evidence items.
Each evidence item must be grounded in a specific URL provided.

Return ONLY JSON list of:
{"url": "...", "title": "...", "section": "...", "text": "1-3 sentences of evidence"}
Rules:
- evidence text must be short and factual (no fluff)
- don't invent urls
- prefer results that directly answer the query
"""

def worker_node(state: AgentState) -> Dict[str, Any]:
    """
    This node is invoked via Send with an injected field `query_item`.
    """
    qi: PlanQuery = state["query_item"]  # injected by Send
    max_results = state.get("max_results") or 5

    q = (qi.get("query") or "").strip()
    section = (qi.get("section") or "General").strip()

    search = make_search(max_results=max_results)
    raw = search.invoke({"query": q})
    results = normalize(raw)

    raw_sources: List[RawSource] = [{"url": r["url"], "title": r["title"], "snippet": r.get("snippet", "")} for r in results]

    # If Tavily returns nothing, still count worker as done.
    if not results:
        return {"raw_sources": raw_sources, "raw_evidence": [], "done_workers": 1}

    # Ask LLM to pick evidence from the snippets (v1, no full page fetch)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt_blob = "\n\n".join([f"- TITLE: {r['title']}\n  URL: {r['url']}\n  SNIPPET: {r.get('snippet','')[:600]}" for r in results])

    resp = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=f"Query: {q}\nTarget section: {section}\n\nSearch results:\n{prompt_blob}")
    ])

    txt = resp.content.strip()
    ev: List[RawEvidence] = []
    try:
        import json
        ev = json.loads(txt)
    except Exception:
        # fallback: nothing extracted
        ev = []

    # enforce section + trim
    cleaned: List[RawEvidence] = []
    for item in ev[:4]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip() or url
        text = str(item.get("text", "")).strip()
        if url and text:
            cleaned.append({"url": url, "title": title, "section": section, "text": text[:900]})

    return {"raw_sources": raw_sources, "raw_evidence": cleaned, "done_workers": 1}
