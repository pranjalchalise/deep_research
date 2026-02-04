"""
Node functions for the research graph.

Every node takes the full state dict, reads what it needs, does its work
(LLM calls, search, extraction), and returns only the keys it wants to update.

Fields that use operator.add in the state (evidence, worker_results,
done_workers) get merged automatically, so parallel workers can append
without stepping on each other.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk

from src.pipeline.state import ResearchState, get_configuration
from src.pipeline.prompts import (
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
    get_structure_instruction,
)


# ---------------------------------------------------------------------------
# Model helpers -- read model names from RunnableConfig so they're
# configurable via config={"configurable": {...}}.
# Defaults: gpt-4o for planning/writing, gpt-4o-mini for extraction.
# ---------------------------------------------------------------------------

# Cache by model name so we don't create a new wrapper per node call
_model_cache: Dict[str, Any] = {}


def _get_llm(config: RunnableConfig):
    """Smart model for planning, synthesis, writing."""
    conf = get_configuration(config)
    model = conf["model"]
    if model not in _model_cache:
        _model_cache[model] = create_chat_model(model=model, temperature=0.2)
    return _model_cache[model]


def _get_fast_llm(config: RunnableConfig):
    """Fast/cheap model for bulk extraction."""
    conf = get_configuration(config)
    model = conf["fast_model"]
    if model not in _model_cache:
        _model_cache[model] = create_chat_model(model=model, temperature=0.1)
    return _model_cache[model]


def _clarification_section(state: ResearchState) -> str:
    """Build the clarification snippet we inject into prompts.
    Returns empty string if the user never clarified anything."""
    uc = state.get("user_clarification", "")
    return f"\nUSER CLARIFICATION:\n{uc}" if uc else ""


def _max_results(config: RunnableConfig) -> int:
    """How many Tavily results to fetch per query."""
    return get_configuration(config)["max_search_results"]


# ===================================================================
# 1. UNDERSTAND -- figure out what the user is actually asking
# ===================================================================

def understand_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Check the query for ambiguity (polysemy, scope, time, geography,
    etc.) and decide if we need to ask the user before proceeding."""
    query = state["query"]
    llm = _get_llm(config)

    response = llm.invoke([
        SystemMessage(content="Analyze the query. Output valid JSON only."),
        HumanMessage(content=UNDERSTAND_PROMPT.format(query=query)),
    ])

    result = parse_json_object(response.content, default={
        "understanding": query,
        "is_clear": True,
    })

    is_clear = result.get("is_clear", True)
    understanding = result.get("understanding", query)

    status = "clear" if is_clear else "ambiguous"
    msg = AIMessage(content=f"[understand] Query is {status}: {understanding}")

    return {
        "understanding": result,
        "is_ambiguous": not is_clear,
        "clarification_question": result.get("clarification_question", ""),
        "clarification_options": result.get("clarification_options", []),
        "messages": [msg],
    }


# ===================================================================
# 2. CLARIFY -- human-in-the-loop pause point
# ===================================================================

def clarify_node(state: ResearchState) -> Dict[str, Any]:
    """This node is basically a passthrough. The real magic happens
    *before* it runs: the graph pauses (interrupt_before), the caller
    injects the user's answer via update_state(), then resumes.
    We just make sure the clarification is visible in state."""
    return {
        "user_clarification": state.get("user_clarification", ""),
    }


# ===================================================================
# 3. PLAN -- turn the query into a research plan
# ===================================================================

def plan_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Ask the LLM to decide what questions to answer, what angles to
    cover, and what to search for first."""
    query = state["query"]
    llm = _get_llm(config)

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

    questions = result.get("research_questions", [query])
    searches = result.get("initial_searches", [query])
    msg = AIMessage(content=f"[plan] {len(questions)} questions, {len(searches)} initial searches planned")

    # We reset accumulators here so the research loop starts clean
    return {
        "research_questions": questions,
        "aspects_to_cover": result.get("aspects_to_cover", []),
        "pending_searches": searches,
        "iteration": 0,
        "evidence": [],
        "worker_results": [],
        "done_workers": 0,
        "sources": {},
        "searches_done": [],
        "messages": [msg],
    }


# ===================================================================
# 4. ORCHESTRATE -- split query into parallel sub-questions (multi)
# ===================================================================

def orchestrate_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Break the query into 3-5 independent sub-questions for parallel
    workers. If we're looping back after synthesis flagged gaps, we
    feed those gaps in so the orchestrator targets them specifically."""
    query = state["query"]
    llm = _get_llm(config)

    # On loop-back, tell the LLM what gaps remain from the last round
    synthesis = state.get("synthesis", {})
    extra = ""
    if synthesis.get("additional_searches"):
        extra = "\n\nPREVIOUS GAPS TO ADDRESS:\n"
        for s in synthesis["additional_searches"]:
            extra += f"- {s}\n"

    response = llm.invoke([
        SystemMessage(content="Break down research query. Output JSON."),
        HumanMessage(content=ORCHESTRATE_PROMPT.format(
            query=query + extra,
            clarification_section=_clarification_section(state),
        )),
    ])

    result = parse_json_object(response.content, default={"sub_questions": []})
    sub_questions = result.get("sub_questions", [])[:5]

    return {
        "sub_questions": sub_questions,
        "total_workers": len(sub_questions),
    }


# ===================================================================
# 5. SEARCH WORKER -- research one sub-question (invoked via Send)
# ===================================================================

def search_worker_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Handle a single sub-question end-to-end: search, fetch pages,
    extract facts, then summarize.

    Invoked via Send() so each worker gets its own sub_question in
    the state payload. Returns are merged via operator.add -- that's
    why worker_results is a list-of-one and done_workers is 1."""
    sq = state.get("sub_question", {})
    if isinstance(sq, str):
        sq = {"question": sq, "search_queries": [sq]}

    question = sq.get("question", state.get("query", ""))
    search_queries = sq.get("search_queries", [question])

    fast_llm = _get_fast_llm(config)
    llm = _get_llm(config)

    worker_evidence: List[Dict] = []
    worker_sources: Dict[str, Dict] = {}

    for search_q in search_queries[:3]:
        results = cached_search(
            query=search_q, max_results=_max_results(config),
            use_cache=True, cache_dir=".cache/search",
        )
        if not results:
            continue

        for r in results[:2]:
            url, title = r["url"], r["title"]

            # Cap at 5 sources per worker to keep costs sane
            if url in worker_sources or len(worker_sources) >= 5:
                continue

            worker_sources[url] = {
                "url": url, "title": title,
                "source_id": len(worker_sources) + 1,
            }

            chunks = fetch_and_chunk(
                url=url, chunk_chars=3000, max_chunks=2,
                timeout_s=8, use_cache=True, cache_dir=".cache/pages",
            )
            if not chunks:
                continue

            content = "\n".join(chunks[:2])
            response = fast_llm.invoke([
                SystemMessage(content="Extract facts as JSON array."),
                HumanMessage(content=WORKER_EXTRACT_PROMPT.format(
                    question=question, url=url, content=content[:5000],
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

    # Have the smart model summarize what this worker found
    evidence_text = "\n".join(
        f"- {e['fact']}" for e in worker_evidence
    ) if worker_evidence else "No evidence found."

    response = llm.invoke([
        SystemMessage(content="Summarize research findings as JSON."),
        HumanMessage(content=WORKER_SUMMARY_PROMPT.format(
            question=question, evidence=evidence_text,
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

    # Wrapped in lists / set to 1 because these fields use operator.add
    return {
        "worker_results": [worker_result],
        "done_workers": 1,
        "evidence": worker_evidence,
    }


# ===================================================================
# 6. COLLECT -- merge sources after all workers finish (multi)
# ===================================================================

def collect_node(state: ResearchState) -> Dict[str, Any]:
    """Deduplicate sources from all workers into one dict. LangGraph
    guarantees every Send() worker is done before we get here."""
    merged = dict(state.get("sources", {}))
    next_id = max((s.get("source_id", 0) for s in merged.values()), default=0)

    for wr in state.get("worker_results", []):
        for url, src in wr.get("sources", {}).items():
            if url not in merged:
                next_id += 1
                merged[url] = {**src, "source_id": next_id}

    return {"sources": merged}


# ===================================================================
# 7. SYNTHESIZE -- combine all worker findings (multi)
# ===================================================================

def synthesize_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Pull together what every worker found, resolve conflicts, spot
    remaining gaps, and decide if we need another research round."""
    query = state["query"]
    llm = _get_llm(config)

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


# ===================================================================
# 8. SEARCH & EXTRACT -- the single-agent search loop
# ===================================================================

def search_and_extract_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Run every pending search query, fetch the top pages, and pull
    out evidence. This is the single-agent path -- everything happens
    sequentially inside one node."""
    query = state["query"]
    pending = list(state.get("pending_searches", []))
    sources = dict(state.get("sources", {}))
    searches_done = list(state.get("searches_done", []))
    research_questions = state.get("research_questions", [])

    fast_llm = _get_fast_llm(config)
    new_evidence: List[Dict] = []
    max_sources = 20

    for search_q in pending:
        if search_q in searches_done:
            continue
        searches_done.append(search_q)

        results = cached_search(
            query=search_q, max_results=_max_results(config),
            use_cache=True, cache_dir=".cache/search",
        )
        if not results:
            continue

        for r in results[:3]:
            url, title = r["url"], r["title"]

            if url in sources or len(sources) >= max_sources:
                continue

            source_id = len(sources) + 1
            sources[url] = {
                "url": url, "title": title,
                "source_id": source_id, "evidence_count": 0,
            }

            chunks = fetch_and_chunk(
                url=url, chunk_chars=4000, max_chunks=2,
                timeout_s=10, use_cache=True, cache_dir=".cache/pages",
            )
            if not chunks:
                continue

            # Give the extraction model context so it knows what to look for
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
                    context=context, url=url, content=content[:6000],
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

    new_queries = len(pending) - sum(1 for q in pending if q in state.get("searches_done", []))
    msg = AIMessage(content=f"[search] Ran {new_queries} searches, found {len(new_evidence)} evidence items from {len(sources)} sources")

    return {
        "evidence": new_evidence,
        "sources": sources,
        "searches_done": searches_done,
        "messages": [msg],
    }


# ===================================================================
# 9. DETECT GAPS -- decide whether to loop or write (single-agent)
# ===================================================================

def detect_gaps_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Look at what we have so far and figure out what's missing.
    If coverage is good enough (or we've hit the iteration cap),
    we tell the graph to move on to writing. Otherwise we suggest
    more searches and loop back."""
    query = state["query"]
    llm = _get_llm(config)

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

    cov = result.get("overall_coverage", 0.5)
    gaps = result.get("gaps", [])
    iteration = state.get("iteration", 0) + 1
    msg = AIMessage(content=f"[gaps] Iteration {iteration}: coverage {cov:.0%}, {len(gaps)} gaps remaining")

    return {
        "coverage": cov,
        "ready_to_write": result.get("ready_to_write", False),
        "pending_searches": result.get("suggested_searches", []),
        "gaps": gaps,
        "iteration": iteration,
        "messages": [msg],
    }


# ===================================================================
# 10. VERIFY -- fact-check claims against evidence (both paths)
# ===================================================================

def verify_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Cross-check the top evidence items. Each claim gets compared
    against all available evidence to see if it holds up."""
    evidence_list = state.get("evidence", [])
    if not evidence_list:
        return {"verified_claims": [], "confidence": 0.0}

    llm = _get_llm(config)

    # Cap at 20 claims so we don't blow the budget
    claims = [e.get("fact", "") for e in evidence_list[:20]]
    claims_text = "\n".join(f"- {c}" for c in claims)

    evidence_text = "\n".join(
        f"[{i+1}] {e.get('fact', '')} (source: {e.get('source_title', 'unknown')})"
        for i, e in enumerate(evidence_list)
    )

    response = llm.invoke([
        SystemMessage(content="Verify claims against evidence. Output valid JSON array."),
        HumanMessage(content=VERIFY_PROMPT.format(
            claims=claims_text, evidence=evidence_text,
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

    msg = AIMessage(content=f"[verify] {supported_count}/{len(verified)} claims supported ({confidence:.0%} confidence)")

    return {
        "verified_claims": verified,
        "confidence": confidence,
        "messages": [msg],
    }


# ===================================================================
# 11. WRITE REPORT -- produce the final markdown (both paths)
# ===================================================================

def write_report_node(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the final report with inline citations. Uses a different
    prompt for multi-agent mode since we also have worker summaries
    to give the writer thematic context.

    Supports configurable report_structure and system_prompt from config."""
    query = state["query"]
    mode = state.get("mode", "single")
    llm = _get_llm(config)
    conf = get_configuration(config)

    evidence_list = state.get("evidence", [])
    sources = state.get("sources", {})

    if not evidence_list:
        report = f"# Research Report\n\nInsufficient information found for: {query}\n"
        return {
            "report": report,
            "messages": [AIMessage(content=report)],
            "metadata": {"sources_count": 0, "evidence_count": 0},
        }

    evidence_text = "\n".join(
        f"[{i+1}] {e.get('fact', '')}\n   Quote: \"{e.get('quote', '')}\"\n   Source: {e.get('source_title', 'unknown')}"
        for i, e in enumerate(evidence_list)
    )

    sources_text = "\n".join(
        f"[{s.get('source_id', i+1)}] {s.get('title', 'Unknown')}\n   URL: {s.get('url', '')}"
        for i, (url, s) in enumerate(sources.items())
    )

    # Build the system message: base instruction + optional user-provided prompt
    report_structure = conf["report_structure"]
    structure_instruction = get_structure_instruction(report_structure)
    custom_prompt = conf["system_prompt"]

    base_system = "Write a comprehensive research report. Every claim must have a citation."
    if custom_prompt:
        base_system += f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{custom_prompt}"

    if mode == "multi":
        # In multi-agent mode we also pass worker summaries so the writer
        # can organize the report by theme instead of by source
        worker_summaries = ""
        for wr in state.get("worker_results", []):
            worker_summaries += f"\n### {wr.get('question', 'Unknown')}\n"
            worker_summaries += f"{wr.get('answer', 'N/A')}\n"
            worker_summaries += f"Findings: {', '.join(wr.get('key_findings', []))}\n"

        response = llm.invoke([
            SystemMessage(content=base_system),
            HumanMessage(content=WRITE_MULTI_PROMPT.format(
                query=query,
                clarification_section=_clarification_section(state),
                worker_summaries=worker_summaries,
                evidence=evidence_text,
                sources=sources_text,
                structure_instruction=structure_instruction,
            )),
        ])
    else:
        response = llm.invoke([
            SystemMessage(content=base_system),
            HumanMessage(content=WRITE_PROMPT.format(
                query=query,
                clarification_section=_clarification_section(state),
                evidence=evidence_text,
                sources=sources_text,
                structure_instruction=structure_instruction,
            )),
        ])

    report = response.content

    # Tack on a sources footer if the LLM forgot to include one
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

    return {"report": report, "messages": [AIMessage(content=report)], "metadata": metadata}
