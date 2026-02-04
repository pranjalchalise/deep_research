"""
Node functions for the v10 research graph.

Every function follows ``(state: ResearchState) -> dict`` and returns only
the keys it wants to update.  Fields annotated with ``operator.add`` in
ResearchState (evidence, worker_results, done_workers) are accumulated
automatically by LangGraph.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk

from src.v10.state import ResearchState
from src.v10.prompts import (
    UNDERSTAND_PROMPT,
    PLAN_PROMPT,
    EXTRACT_PROMPT,
    GAP_DETECTION_PROMPT,
    VERIFY_PROMPT,
    WRITE_PROMPT,
    ORCHESTRATE_PROMPT,
    WORKER_EXTRACT_PROMPT,
    WORKER_SUMMARY_PROMPT,
    SYNTHESIZE_PROMPT,
    WRITE_MULTI_PROMPT,
)

# ── Lazy model singletons ─────────────────────────────────────────────
_llm = None
_fast_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_chat_model(model="gpt-4o", temperature=0.2)
    return _llm


def _get_fast_llm():
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)
    return _fast_llm


def _clarification_section(state: ResearchState) -> str:
    uc = state.get("user_clarification", "")
    return f"\nUSER CLARIFICATION:\n{uc}" if uc else ""


# ── 1. understand_node ─────────────────────────────────────────────────

def understand_node(state: ResearchState) -> Dict[str, Any]:
    """Analyze the query for ambiguity across seven dimensions."""
    query = state["query"]
    llm = _get_llm()

    response = llm.invoke([
        SystemMessage(content="Analyze the query. Output valid JSON only."),
        HumanMessage(content=UNDERSTAND_PROMPT.format(query=query)),
    ])

    result = parse_json_object(response.content, default={
        "understanding": query,
        "is_clear": True,
    })

    is_clear = result.get("is_clear", True)

    return {
        "understanding": result,
        "is_ambiguous": not is_clear,
        "clarification_question": result.get("clarification_question", ""),
        "clarification_options": result.get("clarification_options", []),
    }


# ── 2. clarify_node ───────────────────────────────────────────────────

def clarify_node(state: ResearchState) -> Dict[str, Any]:
    """HITL interrupt target.

    When ``interrupt_before=["clarify"]`` is set, the graph pauses *before*
    this node runs.  The caller injects ``user_clarification`` via
    ``graph.update_state()`` and resumes.  This node is a passthrough that
    ensures the clarification is visible in state for downstream nodes.
    """
    return {
        "user_clarification": state.get("user_clarification", ""),
    }


# ── 3. plan_node ──────────────────────────────────────────────────────

def plan_node(state: ResearchState) -> Dict[str, Any]:
    """Create a research plan with questions and initial search queries."""
    query = state["query"]
    llm = _get_llm()

    response = llm.invoke([
        SystemMessage(content="Create a research plan. Output valid JSON only."),
        HumanMessage(content=PLAN_PROMPT.format(
            query=query,
            clarification_section=_clarification_section(state),
        )),
    ])

    result = parse_json_object(response.content, default={
        "research_questions": [query],
        "aspects_to_cover": [],
        "initial_searches": [query],
    })

    return {
        "research_questions": result.get("research_questions", [query]),
        "aspects_to_cover": result.get("aspects_to_cover", []),
        "pending_searches": result.get("initial_searches", [query]),
        "iteration": 0,
        "evidence": [],
        "worker_results": [],
        "done_workers": 0,
        "sources": {},
        "searches_done": [],
    }


# ── 4. orchestrate_node (multi-agent) ─────────────────────────────────

def orchestrate_node(state: ResearchState) -> Dict[str, Any]:
    """Break the query into independent sub-questions for parallel workers."""
    query = state["query"]
    llm = _get_llm()

    # On loop-back the synthesis may suggest additional searches
    synthesis = state.get("synthesis", {})
    extra = ""
    if synthesis.get("additional_searches"):
        extra = f"\n\nPREVIOUS GAPS TO ADDRESS:\n"
        for s in synthesis["additional_searches"]:
            extra += f"- {s}\n"

    prompt_query = query + extra

    response = llm.invoke([
        SystemMessage(content="Break down research query. Output JSON."),
        HumanMessage(content=ORCHESTRATE_PROMPT.format(
            query=prompt_query,
            clarification_section=_clarification_section(state),
        )),
    ])

    result = parse_json_object(response.content, default={"sub_questions": []})
    sub_questions = result.get("sub_questions", [])[:5]

    return {
        "sub_questions": sub_questions,
        "total_workers": len(sub_questions),
    }


# ── 5. search_worker_node (multi-agent, via Send) ─────────────────────

def search_worker_node(state: ResearchState) -> Dict[str, Any]:
    """Research a single sub-question.  Invoked via Send() — receives
    ``sub_question``, ``query``, and ``user_clarification`` from the
    Send payload merged with the graph state.
    """
    sq = state.get("sub_question", {})
    if isinstance(sq, str):
        sq = {"question": sq, "search_queries": [sq]}

    question = sq.get("question", state.get("query", ""))
    search_queries = sq.get("search_queries", [question])

    fast_llm = _get_fast_llm()
    llm = _get_llm()

    worker_evidence: List[Dict] = []
    worker_sources: Dict[str, Dict] = {}

    for search_q in search_queries[:3]:
        results = cached_search(
            query=search_q,
            max_results=4,
            use_cache=True,
            cache_dir=".cache_v10/search",
        )
        if not results:
            continue

        for r in results[:2]:
            url = r["url"]
            title = r["title"]

            if url in worker_sources or len(worker_sources) >= 5:
                continue

            worker_sources[url] = {
                "url": url,
                "title": title,
                "source_id": len(worker_sources) + 1,
            }

            chunks = fetch_and_chunk(
                url=url,
                chunk_chars=3000,
                max_chunks=2,
                timeout_s=8,
                use_cache=True,
                cache_dir=".cache_v10/pages",
            )
            if not chunks:
                continue

            content = "\n".join(chunks[:2])

            response = fast_llm.invoke([
                SystemMessage(content="Extract facts as JSON array."),
                HumanMessage(content=WORKER_EXTRACT_PROMPT.format(
                    question=question,
                    url=url,
                    content=content[:5000],
                )),
            ])

            facts = parse_json_array(response.content, default=[])
            for fact in facts[:4]:
                if isinstance(fact, dict) and fact.get("fact"):
                    worker_evidence.append({
                        "fact": fact["fact"],
                        "quote": fact.get("quote", ""),
                        "source_url": url,
                        "source_title": title,
                        "confidence": fact.get("confidence", 0.8),
                    })

    # Summarize findings
    evidence_text = "\n".join(
        f"- {e['fact']}" for e in worker_evidence
    ) if worker_evidence else "No evidence found."

    response = llm.invoke([
        SystemMessage(content="Summarize research findings as JSON."),
        HumanMessage(content=WORKER_SUMMARY_PROMPT.format(
            question=question,
            evidence=evidence_text,
            sources_count=len(worker_sources),
        )),
    ])

    summary = parse_json_object(response.content, default={
        "answer": "Insufficient evidence",
        "key_findings": [],
        "confidence": 0.0,
        "gaps": ["Unable to find relevant information"],
    })

    worker_result = {
        "question": question,
        "answer": summary.get("answer", ""),
        "key_findings": summary.get("key_findings", []),
        "confidence": summary.get("confidence", 0.0),
        "gaps": summary.get("gaps", []),
        "evidence_count": len(worker_evidence),
        "sources": worker_sources,
    }

    return {
        "worker_results": [worker_result],
        "done_workers": 1,
        "evidence": worker_evidence,
    }


# ── 6. collect_node (multi-agent barrier) ─────────────────────────────

def collect_node(state: ResearchState) -> Dict[str, Any]:
    """Merge worker sources into the global deduplicated sources dict.

    LangGraph's Send() guarantees all workers finish before this node runs.
    """
    merged_sources = dict(state.get("sources", {}))
    source_id = max((s.get("source_id", 0) for s in merged_sources.values()), default=0)

    for wr in state.get("worker_results", []):
        for url, src in wr.get("sources", {}).items():
            if url not in merged_sources:
                source_id += 1
                src_copy = dict(src)
                src_copy["source_id"] = source_id
                merged_sources[url] = src_copy

    return {"sources": merged_sources}


# ── 7. synthesize_node (multi-agent) ──────────────────────────────────

def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesize findings from all parallel workers."""
    query = state["query"]
    llm = _get_llm()

    findings_text = ""
    for wr in state.get("worker_results", []):
        findings_text += f"\n### {wr.get('question', 'Unknown')}\n"
        findings_text += f"Answer: {wr.get('answer', 'N/A')}\n"
        findings_text += f"Key findings: {', '.join(wr.get('key_findings', []))}\n"
        findings_text += f"Confidence: {wr.get('confidence', 0):.0%}\n"
        findings_text += f"Gaps: {', '.join(wr.get('gaps', []))}\n"

    response = llm.invoke([
        SystemMessage(content="Synthesize research findings. Output JSON."),
        HumanMessage(content=SYNTHESIZE_PROMPT.format(
            query=query,
            clarification_section=_clarification_section(state),
            worker_findings=findings_text,
        )),
    ])

    result = parse_json_object(response.content, default={
        "combined_answer": "",
        "overall_confidence": 0.5,
        "needs_more_research": False,
    })

    return {
        "synthesis": result,
        "iteration": state.get("iteration", 0) + 1,
        "coverage": result.get("overall_confidence", 0.5),
        "ready_to_write": not result.get("needs_more_research", False),
    }


# ── 8. search_and_extract_node (single-agent) ─────────────────────────

def search_and_extract_node(state: ResearchState) -> Dict[str, Any]:
    """Execute pending searches and extract evidence (single-agent path)."""
    query = state["query"]
    pending = list(state.get("pending_searches", []))
    sources = dict(state.get("sources", {}))
    searches_done = list(state.get("searches_done", []))
    research_questions = state.get("research_questions", [])

    fast_llm = _get_fast_llm()
    new_evidence: List[Dict] = []
    max_sources = 20

    for search_q in pending:
        if search_q in searches_done:
            continue
        searches_done.append(search_q)

        results = cached_search(
            query=search_q,
            max_results=5,
            use_cache=True,
            cache_dir=".cache_v10/search",
        )
        if not results:
            continue

        for r in results[:3]:
            url = r["url"]
            title = r["title"]

            if url in sources or len(sources) >= max_sources:
                continue

            source_id = len(sources) + 1
            sources[url] = {
                "url": url,
                "title": title,
                "source_id": source_id,
                "evidence_count": 0,
            }

            chunks = fetch_and_chunk(
                url=url,
                chunk_chars=4000,
                max_chunks=2,
                timeout_s=10,
                use_cache=True,
                cache_dir=".cache_v10/pages",
            )
            if not chunks:
                continue

            context = f"Query: {query}\n"
            uc = state.get("user_clarification", "")
            if uc:
                context += f"User clarification: {uc}\n"
            if research_questions:
                context += "Research questions:\n"
                for q in research_questions:
                    context += f"- {q}\n"

            content = "\n\n".join(chunks[:2])

            response = fast_llm.invoke([
                SystemMessage(content="Extract facts as JSON array. Return [] if not relevant."),
                HumanMessage(content=EXTRACT_PROMPT.format(
                    context=context,
                    url=url,
                    content=content[:6000],
                )),
            ])

            facts = parse_json_array(response.content, default=[])
            count = 0
            for fact in facts[:5]:
                if isinstance(fact, dict) and fact.get("fact"):
                    new_evidence.append({
                        "fact": fact["fact"],
                        "quote": fact.get("quote", ""),
                        "source_url": url,
                        "source_title": title,
                        "confidence": fact.get("confidence", 0.8),
                    })
                    count += 1
            sources[url]["evidence_count"] = count

    return {
        "evidence": new_evidence,
        "sources": sources,
        "searches_done": searches_done,
    }


# ── 9. detect_gaps_node (single-agent) ────────────────────────────────

def detect_gaps_node(state: ResearchState) -> Dict[str, Any]:
    """Assess coverage and decide whether to loop or proceed to writing."""
    query = state["query"]
    llm = _get_llm()

    evidence_list = state.get("evidence", [])
    research_questions = state.get("research_questions", [])

    evidence_text = "\n".join(
        f"[{i+1}] {e.get('fact', '')} (from: {e.get('source_title', 'unknown')})"
        for i, e in enumerate(evidence_list)
    )
    questions_text = "\n".join(f"- {q}" for q in research_questions)

    response = llm.invoke([
        SystemMessage(content="Assess research coverage. Output valid JSON."),
        HumanMessage(content=GAP_DETECTION_PROMPT.format(
            query=query,
            clarification_section=_clarification_section(state),
            questions=questions_text,
            evidence=evidence_text,
        )),
    ])

    result = parse_json_object(response.content, default={
        "overall_coverage": 0.5,
        "gaps": [],
        "suggested_searches": [],
        "ready_to_write": False,
    })

    return {
        "coverage": result.get("overall_coverage", 0.5),
        "ready_to_write": result.get("ready_to_write", False),
        "pending_searches": result.get("suggested_searches", []),
        "gaps": result.get("gaps", []),
        "iteration": state.get("iteration", 0) + 1,
    }


# ── 10. verify_node ──────────────────────────────────────────────────

def verify_node(state: ResearchState) -> Dict[str, Any]:
    """Cross-check evidence items for internal consistency."""
    evidence_list = state.get("evidence", [])
    if not evidence_list:
        return {"verified_claims": [], "confidence": 0.0}

    llm = _get_llm()

    claims = [e.get("fact", "") for e in evidence_list[:20]]
    claims_text = "\n".join(f"- {c}" for c in claims)

    evidence_text = "\n".join(
        f"[{i+1}] {e.get('fact', '')} (source: {e.get('source_title', 'unknown')})"
        for i, e in enumerate(evidence_list)
    )

    response = llm.invoke([
        SystemMessage(content="Verify claims against evidence. Output valid JSON array."),
        HumanMessage(content=VERIFY_PROMPT.format(
            claims=claims_text,
            evidence=evidence_text,
        )),
    ])

    verifications = parse_json_array(response.content, default=[])

    verified = []
    for v in verifications:
        if isinstance(v, dict):
            verified.append({
                "claim": v.get("claim", ""),
                "supported": v.get("supported", False),
                "supporting_evidence": v.get("supporting_evidence", []),
                "confidence": v.get("confidence", 0.5),
                "notes": v.get("notes", ""),
            })

    supported_count = sum(1 for v in verified if v.get("supported"))
    confidence = supported_count / len(verified) if verified else 0.0

    return {
        "verified_claims": verified,
        "confidence": confidence,
    }


# ── 11. write_report_node ────────────────────────────────────────────

def write_report_node(state: ResearchState) -> Dict[str, Any]:
    """Write the final research report with citations."""
    query = state["query"]
    mode = state.get("mode", "single")
    llm = _get_llm()

    evidence_list = state.get("evidence", [])
    sources = state.get("sources", {})

    if not evidence_list:
        report = f"# Research Report\n\nInsufficient information found for: {query}\n"
        return {"report": report, "metadata": {"sources_count": 0, "evidence_count": 0}}

    # Assign stable IDs to evidence and sources
    evidence_text = "\n".join(
        f"[{i+1}] {e.get('fact', '')}\n   Quote: \"{e.get('quote', '')}\"\n   Source: {e.get('source_title', 'unknown')}"
        for i, e in enumerate(evidence_list)
    )

    sources_text = "\n".join(
        f"[{s.get('source_id', i+1)}] {s.get('title', 'Unknown')}\n   URL: {s.get('url', '')}"
        for i, (url, s) in enumerate(sources.items())
    )

    if mode == "multi":
        # Build worker summaries
        worker_summaries = ""
        for wr in state.get("worker_results", []):
            worker_summaries += f"\n### {wr.get('question', 'Unknown')}\n"
            worker_summaries += f"{wr.get('answer', 'N/A')}\n"
            worker_summaries += f"Findings: {', '.join(wr.get('key_findings', []))}\n"

        response = llm.invoke([
            SystemMessage(content="Write research report with citations."),
            HumanMessage(content=WRITE_MULTI_PROMPT.format(
                query=query,
                clarification_section=_clarification_section(state),
                worker_summaries=worker_summaries,
                evidence=evidence_text,
                sources=sources_text,
            )),
        ])
    else:
        response = llm.invoke([
            SystemMessage(content="Write a comprehensive research report. Every claim must have a citation."),
            HumanMessage(content=WRITE_PROMPT.format(
                query=query,
                clarification_section=_clarification_section(state),
                evidence=evidence_text,
                sources=sources_text,
            )),
        ])

    report = response.content

    # Append sources footer if not already present
    if "## Sources" not in report and "## References" not in report:
        report += "\n\n---\n\n## Sources\n\n"
        for url, s in sources.items():
            sid = s.get("source_id", "?")
            title = s.get("title", "Unknown")
            report += f"[{sid}] {title} - {url}\n"

    metadata = {
        "iterations": state.get("iteration", 0),
        "coverage": state.get("coverage", 0.0),
        "confidence": state.get("confidence", 0.0),
        "sources_count": len(sources),
        "evidence_count": len(evidence_list),
        "mode": mode,
    }

    return {"report": report, "metadata": metadata}
