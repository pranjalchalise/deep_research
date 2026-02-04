"""
Orchestrator-worker pattern: a lead agent assigns research questions to
parallel subagents, each independently searches and extracts evidence,
then a synthesizer merges everything for gap detection.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import (
    AgentState,
    SubagentAssignment,
    SubagentFindings,
    OrchestratorState,
    RawEvidence,
    RawSource,
    DeadEnd,
    TrajectoryStep,
)
from src.core.config import V8Config
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk, should_skip_url
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model
from src.utils.optimization import deduplicate_queries, deduplicate_query_items


ORCHESTRATOR_SYSTEM = """You are a research orchestrator that coordinates a team of research subagents.

Given a research plan, assign specific research questions to subagents.
Each subagent will independently research their assigned question.

Guidelines:
1. Assign 3-5 questions to subagents (primary questions first)
2. Each question should be focused and answerable
3. Distribute work to maximize coverage
4. Prioritize questions that contribute most to overall understanding

Return JSON:
{
  "assignments": [
    {
      "subagent_id": "SA1",
      "question": "Specific research question",
      "queries": ["search query 1", "search query 2"],
      "target_sources": ["academic", "official", "news"]
    }
  ],
  "strategy": "Brief explanation of research strategy"
}
"""


def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """Assign research questions to subagents with query deduplication."""
    cfg = V8Config()
    plan = state.get("plan") or {}
    research_tree = plan.get("research_tree") or {}
    primary_anchor = state.get("primary_anchor", "")
    anchor_terms = state.get("anchor_terms") or []
    refinement_queries = state.get("refinement_queries") or []
    current_iteration = state.get("research_iteration", 0)

    if refinement_queries:
        # Create assignments from refinement queries
        assignments = _create_assignments_from_refinement(refinement_queries, primary_anchor)
    elif research_tree:
        assignments = _create_assignments_from_tree(research_tree, primary_anchor, anchor_terms)
    else:
        assignments = _create_assignments_from_plan(plan, primary_anchor, anchor_terms)

    total_queries_before = 0
    total_queries_after = 0

    if cfg.enable_query_dedup:
        for assignment in assignments:
            queries = assignment.get("queries", [])
            total_queries_before += len(queries)

            deduped, _ = deduplicate_queries(queries, cfg.query_similarity_threshold)
            assignment["queries"] = deduped
            total_queries_after += len(deduped)

        assignments = _deduplicate_assignments(assignments, cfg.query_similarity_threshold)

    max_subagents = state.get("max_subagents", cfg.max_subagents)
    assignments = assignments[:max_subagents]

    orchestrator_state: OrchestratorState = {
        "phase": "primary_research" if current_iteration == 0 else "refinement",
        "questions_assigned": len(assignments),
        "questions_completed": 0,
        "overall_confidence": state.get("overall_confidence", 0.0),
    }

    dedup_stats = {
        "queries_before": total_queries_before,
        "queries_after": total_queries_after,
        "queries_removed": total_queries_before - total_queries_after,
    }

    return {
        "subagent_assignments": assignments,
        "orchestrator_state": orchestrator_state,
        "total_workers": len(assignments),
        "done_workers": 0,
        "done_subagents": 0,
        "dedup_stats": dedup_stats,
    }


def _deduplicate_assignments(
    assignments: List[SubagentAssignment],
    threshold: float = 0.85
) -> List[SubagentAssignment]:
    """Remove near-duplicate assignments, merging their queries into the kept one."""
    if len(assignments) <= 1:
        return assignments

    unique_assignments: List[SubagentAssignment] = []

    for assignment in assignments:
        question = assignment.get("question", "")
        is_duplicate = False

        for existing in unique_assignments:
            existing_question = existing.get("question", "")
            # Use the same similarity function
            from src.utils.optimization import query_similarity
            sim = query_similarity(question, existing_question)

            if sim >= threshold:
                is_duplicate = True
                # Merge queries from duplicate into existing
                existing_queries = existing.get("queries", [])
                new_queries = assignment.get("queries", [])
                merged = existing_queries + [q for q in new_queries if q not in existing_queries]
                existing["queries"] = merged[:5]  # Cap at 5 queries
                break

        if not is_duplicate:
            unique_assignments.append(assignment)

    return unique_assignments


def _create_assignments_from_refinement(
    refinement_queries: List[Dict],
    primary_anchor: str
) -> List[SubagentAssignment]:
    """Create subagent assignments from refinement queries."""
    section_queries: Dict[str, List[str]] = {}
    for q in refinement_queries:
        section = q.get("section", "General")
        query = q.get("query", "")
        if query:
            if section not in section_queries:
                section_queries[section] = []
            section_queries[section].append(query)

    assignments = []
    for i, (section, queries) in enumerate(section_queries.items()):
        assignments.append({
            "subagent_id": f"SA{i+1}",
            "question": f"Find more information about: {section}",
            "queries": queries[:3],  # Max 3 queries per subagent
            "target_sources": ["general", "official"],
        })

    return assignments


def _create_assignments_from_tree(
    research_tree: Dict,
    primary_anchor: str,
    anchor_terms: List[str]
) -> List[SubagentAssignment]:
    """Create subagent assignments from research tree."""
    assignments = []

    primary_questions = research_tree.get("primary", [])
    for i, q in enumerate(primary_questions[:4]):
        queries = q.get("queries", [])

        for j, query in enumerate(queries):
            if primary_anchor and primary_anchor.lower() not in query.lower():
                queries[j] = f'"{primary_anchor}" {query}'

        assignments.append({
            "subagent_id": f"SA{i+1}",
            "question": q.get("question", ""),
            "queries": queries[:3],
            "target_sources": q.get("target_sources", ["general"]),
        })

    secondary_questions = research_tree.get("secondary", [])
    if len(assignments) < 5 and secondary_questions:
        q = secondary_questions[0]
        queries = q.get("queries", [])
        for j, query in enumerate(queries):
            if primary_anchor and primary_anchor.lower() not in query.lower():
                queries[j] = f'"{primary_anchor}" {query}'

        assignments.append({
            "subagent_id": f"SA{len(assignments)+1}",
            "question": q.get("question", ""),
            "queries": queries[:2],
            "target_sources": q.get("target_sources", ["general"]),
        })

    return assignments


def _create_assignments_from_plan(
    plan: Dict,
    primary_anchor: str,
    anchor_terms: List[str]
) -> List[SubagentAssignment]:
    """Create subagent assignments from flat plan queries."""
    queries = plan.get("queries", [])
    outline = plan.get("outline", [])

    section_queries: Dict[str, List[str]] = {}
    for q in queries:
        section = q.get("section", "General")
        query = q.get("query", "")
        if query:
            if section not in section_queries:
                section_queries[section] = []
            section_queries[section].append(query)

    assignments = []
    for i, (section, sq) in enumerate(section_queries.items()):
        if i >= 5:
            break
        assignments.append({
            "subagent_id": f"SA{i+1}",
            "question": f"Research: {section}",
            "queries": sq[:3],
            "target_sources": ["general"],
        })

    return assignments


def fanout_subagents(state: AgentState):
    """Fan out to parallel subagents based on assignments."""
    from langgraph.types import Send

    assignments = state.get("subagent_assignments") or []
    return [
        Send("subagent", {"subagent_assignment": assignment})
        for assignment in assignments
    ]


EXTRACT_SYSTEM = """You extract concise, high-signal evidence from source content.

Given the content from a web page and the research question, extract 2-4 evidence items.

Return ONLY a JSON array of:
{"text": "1-3 sentences of factual evidence directly relevant to the question"}

Rules:
- Evidence text must be short, specific, and factual (no fluff)
- Only include information that directly answers or relates to the question
- Prefer specific numbers, dates, findings over vague statements
- Do not invent or hallucinate information
"""

COMPRESS_SYSTEM = """Compress research findings into a concise summary.

Given evidence items collected for a research question, create a compressed summary
that captures the key facts and findings.

Rules:
1. Maximum 3-4 sentences
2. Include only verified facts from the evidence
3. Prioritize unique, specific information
4. Note any gaps or uncertainties

Return a plain text summary (not JSON).
"""


def subagent_node(state: AgentState) -> Dict[str, Any]:
    """Independently research an assigned question with iterative search and extraction."""
    assignment: SubagentAssignment = state["subagent_assignment"]
    subagent_id = assignment.get("subagent_id", "SA?")
    question = assignment.get("question", "")
    queries = assignment.get("queries", [])

    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir")
    timeout_s = state.get("request_timeout_s", 15.0)
    chunk_chars = state.get("chunk_chars", 3500)
    chunk_overlap = state.get("chunk_overlap", 350)
    max_chunks = state.get("max_chunks_per_source", 4)
    evidence_per_source = state.get("evidence_per_source", 3)
    max_iterations = state.get("subagent_max_iterations", 2)

    primary_anchor = state.get("primary_anchor", "")
    anchor_terms = state.get("anchor_terms") or []
    query_type = (state.get("discovery") or {}).get("query_type", "general")

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    all_evidence: List[RawEvidence] = []
    all_sources: List[RawSource] = []
    dead_ends: List[DeadEnd] = []
    confidence = 0.0

    for iteration in range(max_iterations):
        for query in queries:
            results = cached_search(
                query=query,
                max_results=6,
                lane="general",
                use_cache=use_cache,
                cache_dir=f"{cache_dir}/search" if cache_dir else None,
            )

            if not results:
                dead_ends.append({
                    "query": query,
                    "reason": "no_results",
                    "iteration": iteration,
                    "alternative_tried": False,
                })
                continue

            for r in results:
                all_sources.append({
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r.get("snippet", "")[:700],
                })

            for r in results[:2]:
                url = r["url"]
                title = r["title"]

                if should_skip_url(url):
                    continue

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
                    evidence = _extract_evidence(
                        llm=llm,
                        url=url,
                        title=title,
                        chunks=chunks,
                        question=question,
                        max_evidence=evidence_per_source,
                    )
                    all_evidence.extend(evidence)

        confidence = _assess_subagent_confidence(question, all_evidence)

        if confidence >= 0.8:
            break

        if iteration < max_iterations - 1 and confidence < 0.7:
            queries = _refine_subagent_queries(question, all_evidence, primary_anchor)

    compressed_findings = _compress_findings(llm, question, all_evidence)

    findings: SubagentFindings = {
        "subagent_id": subagent_id,
        "question": question,
        "findings": compressed_findings,
        "evidence_ids": [],  # Will be assigned by reducer
        "confidence": confidence,
        "iterations_used": iteration + 1,
        "dead_ends": dead_ends,
    }

    return {
        "subagent_findings": [findings],
        "raw_sources": all_sources,
        "raw_evidence": all_evidence,
        "done_workers": 1,
        "done_subagents": 1,
    }


def _extract_evidence(
    llm: Any,
    url: str,
    title: str,
    chunks: List[str],
    question: str,
    max_evidence: int = 3,
) -> List[RawEvidence]:
    """Extract evidence from page content chunks."""
    if not chunks:
        return []

    combined = "\n\n---\n\n".join(chunks[:3])
    if len(combined) > 8000:
        combined = combined[:8000] + "..."

    resp = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nSource content:\n{combined}\n\nReturn JSON only.")
    ])

    ev_raw = parse_json_array(resp.content.strip(), default=[])

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
            "section": "General",  # Will be updated later
            "text": text[:900],
        })

    return evidence


def _assess_subagent_confidence(question: str, evidence: List[RawEvidence]) -> float:
    """Assess how well the evidence answers the question."""
    if not evidence:
        return 0.0

    evidence_count = len(evidence)
    avg_length = sum(len(e.get("text", "")) for e in evidence) / max(evidence_count, 1)

    count_score = min(1.0, evidence_count / 4)
    length_score = min(1.0, avg_length / 200)

    return (count_score * 0.6 + length_score * 0.4)


def _refine_subagent_queries(
    question: str,
    evidence: List[RawEvidence],
    primary_anchor: str
) -> List[str]:
    """Generate refined queries based on what we learned."""
    if evidence:
        return [
            f'"{primary_anchor}" {question.split()[-2:]}',  # Last words of question
            f'"{primary_anchor}" details specific',
        ]
    else:
        return [
            f'{primary_anchor} {question}',
            f'"{primary_anchor}" profile',
        ]


def _compress_findings(llm: Any, question: str, evidence: List[RawEvidence]) -> str:
    """Compress all evidence into a summary."""
    if not evidence:
        return f"No relevant information found for: {question}"

    evidence_texts = [e.get("text", "") for e in evidence[:6]]
    evidence_str = "\n".join(f"- {t}" for t in evidence_texts)

    resp = llm.invoke([
        SystemMessage(content=COMPRESS_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nEvidence collected:\n{evidence_str}\n\nCreate a compressed summary.")
    ])

    return resp.content.strip()[:500]


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Merge subagent findings once all workers are done."""
    total_workers = state.get("total_workers", 0)
    done_workers = state.get("done_workers", 0)

    if done_workers < total_workers:
        return {}

    subagent_findings = state.get("subagent_findings") or []
    orchestrator_state = state.get("orchestrator_state") or {}

    if subagent_findings:
        confidences = [f.get("confidence", 0) for f in subagent_findings]
        avg_confidence = sum(confidences) / len(confidences)
    else:
        avg_confidence = 0.0

    all_dead_ends: List[DeadEnd] = state.get("dead_ends") or []
    for finding in subagent_findings:
        for de in finding.get("dead_ends", []):
            all_dead_ends.append(de)

    updated_orchestrator: OrchestratorState = {
        **orchestrator_state,
        "questions_completed": len(subagent_findings),
        "overall_confidence": avg_confidence,
    }

    trajectory = state.get("research_trajectory") or []
    trajectory_step: TrajectoryStep = {
        "iteration": state.get("research_iteration", 0),
        "action": "synthesize",
        "query": f"Synthesized {len(subagent_findings)} subagent findings",
        "result_summary": f"Avg confidence: {avg_confidence:.2f}",
        "confidence_delta": avg_confidence - orchestrator_state.get("overall_confidence", 0),
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "orchestrator_state": updated_orchestrator,
        "overall_confidence": avg_confidence,
        "dead_ends": all_dead_ends,
        "research_trajectory": trajectory + [trajectory_step],
    }


def route_after_synthesis(state: AgentState) -> str:
    """Route after synthesis - always go to gap detector."""
    return "gap_detector"
