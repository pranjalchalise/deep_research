"""
Iterative research loop: confidence checking, automatic refinement,
gap detection, backtracking on dead ends, and early termination.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.advanced.state import (
    AgentState,
    KnowledgeGap,
    DeadEnd,
    TrajectoryStep,
)
from src.advanced.config import ResearchConfig
from src.tools.tavily import cached_search
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model
from src.utils.optimization import should_terminate_early, CostTracker


def confidence_check_node(state: AgentState) -> Dict[str, Any]:
    """
    After discovery, assess if confidence is sufficient.

    Routes to:
    - "clarify" if ambiguous AND human-in-loop enabled
    - "auto_refine" if ambiguous but can auto-resolve
    - "planner" if confident
    """
    discovery = state.get("discovery") or {}
    confidence = discovery.get("confidence", 0)
    candidates = discovery.get("entity_candidates", [])
    query_type = discovery.get("query_type", "general")
    needs_clarification = discovery.get("needs_clarification", False)

    if query_type in ("concept", "technical", "comparison"):
        return {"_route": "planner"}

    if confidence >= 0.85:
        return {"_route": "planner"}

    if len(candidates) > 1:
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.get("confidence", 0),
            reverse=True
        )
        if len(sorted_candidates) >= 2:
            top_two = sorted_candidates[:2]
            confidence_gap = top_two[0].get("confidence", 0) - top_two[1].get("confidence", 0)
            if confidence_gap < 0.2:
                return {"_route": "clarify"}

    if confidence < 0.7:
        if len(candidates) == 1:
            refinement_queries = _generate_refinement_queries(candidates[0], state)
            return {
                "_route": "auto_refine",
                "refinement_queries": refinement_queries,
            }
        elif len(candidates) == 0:
            original_query = state.get("original_query", "")
            refinement_queries = [
                {"query": f"{original_query} profile", "purpose": "broaden"},
                {"query": f"{original_query} biography background", "purpose": "broaden"},
            ]
            return {
                "_route": "auto_refine",
                "refinement_queries": refinement_queries,
            }

    if needs_clarification:
        return {"_route": "clarify"}

    return {"_route": "planner"}


def _generate_refinement_queries(candidate: Dict, state: AgentState) -> List[Dict]:
    """Generate targeted queries to increase confidence about a candidate."""
    name = candidate.get("name", "")
    identifiers = candidate.get("identifiers", [])

    queries = []

    queries.append({
        "query": f'"{name}" LinkedIn profile',
        "purpose": "identity_verification",
    })

    queries.append({
        "query": f'"{name}" site:linkedin.com OR site:github.com',
        "purpose": "official_profile",
    })

    if identifiers:
        queries.append({
            "query": f'"{name}" {identifiers[0]} biography',
            "purpose": "contextual",
        })

    return queries


def route_after_confidence(state: AgentState) -> str:
    """Route based on _route field set by confidence_check_node."""
    return state.get("_route", "planner")


AUTO_REFINE_SYSTEM = """You analyze additional search results to refine entity identification.

Given:
- Original candidate information
- New search results

Determine:
1. Do these results confirm the candidate identity?
2. What new identifying information was found?
3. What is the updated confidence level?

Return JSON:
{
  "confirmed": true/false,
  "new_identifiers": ["identifier1", "identifier2"],
  "updated_confidence": 0.0-1.0,
  "reasoning": "explanation"
}
"""


def auto_refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Do additional discovery searches to increase confidence.

    Strategies:
    1. Search with different query formulations
    2. Look for unique identifiers (LinkedIn, personal site)
    3. Cross-reference with known affiliations
    """
    discovery = state.get("discovery") or {}
    candidates = discovery.get("entity_candidates", [])
    refinement_queries = state.get("refinement_queries", [])

    if candidates:
        candidate = max(candidates, key=lambda c: c.get("confidence", 0))
    else:
        candidate = {"name": "", "identifiers": [], "confidence": 0}

    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir")

    all_results = []
    for q in refinement_queries:
        query = q.get("query", "")
        if not query:
            continue

        results = cached_search(
            query=query,
            max_results=5,
            lane="general",
            use_cache=use_cache,
            cache_dir=f"{cache_dir}/search" if cache_dir else None,
        )
        all_results.extend(results)

    if not all_results:
        return {
            "_route": "planner",
            "discovery": discovery,
        }

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    results_text = "\n\n".join([
        f"[{i+1}] {r['title']}\nURL: {r['url']}\nSnippet: {r.get('snippet', '')[:400]}"
        for i, r in enumerate(all_results[:8])
    ])

    resp = llm.invoke([
        SystemMessage(content=AUTO_REFINE_SYSTEM),
        HumanMessage(content=f"""Original candidate:
Name: {candidate.get('name', 'Unknown')}
Description: {candidate.get('description', '')}
Identifiers: {candidate.get('identifiers', [])}
Current confidence: {candidate.get('confidence', 0)}

New search results:
{results_text}

Analyze whether these results confirm or refute the candidate identity."""),
    ])

    analysis = parse_json_object(resp.content, default={})

    new_confidence = float(analysis.get("updated_confidence", candidate.get("confidence", 0.5)))
    new_identifiers = list(candidate.get("identifiers", []))

    for ident in analysis.get("new_identifiers", []):
        if ident and ident not in new_identifiers:
            new_identifiers.append(ident)

    updated_candidate = {
        **candidate,
        "identifiers": new_identifiers,
        "confidence": new_confidence,
    }

    updated_discovery = {
        **discovery,
        "confidence": new_confidence,
        "entity_candidates": [updated_candidate] + [c for c in candidates if c != candidate],
        "needs_clarification": new_confidence < 0.7,
    }

    if new_confidence >= 0.7:
        return {
            "_route": "planner",
            "discovery": updated_discovery,
            "selected_entity": updated_candidate,
            "primary_anchor": updated_candidate.get("name", ""),
            "anchor_terms": new_identifiers,
        }
    else:
        return {
            "_route": "clarify",
            "discovery": updated_discovery,
        }


GAP_DETECTOR_SYSTEM = """Analyze research findings and identify knowledge gaps.

Given:
- Original question
- Research outline (target sections)
- Current findings by section
- Evidence collected

Identify:
1. Sections with low coverage (insufficient evidence)
2. Unanswered aspects of the question
3. Conflicting information needing resolution
4. Missing perspectives

Return JSON:
{
  "overall_confidence": 0.0-1.0,
  "section_confidence": {
    "Section Name": 0.0-1.0
  },
  "gaps": [
    {
      "section": "Section Name",
      "description": "What information is missing",
      "priority": 0.0-1.0,
      "suggested_queries": ["query1", "query2"]
    }
  ],
  "conflicts": [
    {
      "topic": "What is conflicting",
      "evidence_ids": ["E1", "E3"],
      "resolution_query": "query to resolve"
    }
  ],
  "recommendation": "continue" | "sufficient",
  "reasoning": "explanation"
}
"""


def gap_detector_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze current research state and identify gaps.

    Decides whether to:
    - Continue with another research iteration
    - Proceed to synthesis (sufficient coverage)
    """
    evidence = state.get("evidence") or []
    sources = state.get("sources") or []
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []
    original_query = state.get("original_query", "")
    current_iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_research_iterations", 3)

    section_evidence: Dict[str, List] = defaultdict(list)
    for e in evidence:
        section = e.get("section", "General")
        section_evidence[section].append(e)

    section_confidence = {}
    for section in outline:
        ev_count = len(section_evidence.get(section, []))
        # Heuristic: 3+ evidence items = high confidence
        section_confidence[section] = min(1.0, ev_count / 3.0)

    min_section_conf = min(section_confidence.values()) if section_confidence else 0
    avg_section_conf = sum(section_confidence.values()) / len(section_confidence) if section_confidence else 0

    if avg_section_conf >= 0.8 and min_section_conf >= 0.5:
        return {
            "knowledge_gaps": [],
            "overall_confidence": avg_section_conf,
            "section_confidence": section_confidence,
            "proceed_to_synthesis": True,
            "research_iteration": current_iteration,
        }

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    evidence_summary = []
    for section in outline:
        section_ev = section_evidence.get(section, [])
        if section_ev:
            texts = [e.get("text", "")[:200] for e in section_ev[:3]]
            evidence_summary.append(f"**{section}** ({len(section_ev)} items):\n" + "\n".join(f"- {t}" for t in texts))
        else:
            evidence_summary.append(f"**{section}**: No evidence found")

    resp = llm.invoke([
        SystemMessage(content=GAP_DETECTOR_SYSTEM),
        HumanMessage(content=f"""Original question: {original_query}

Target sections: {outline}

Current evidence by section:
{chr(10).join(evidence_summary)}

Total sources: {len(sources)}
Research iteration: {current_iteration + 1} of {max_iterations}

Analyze gaps and recommend whether to continue researching or proceed to synthesis."""),
    ])

    analysis = parse_json_object(resp.content, default={})

    overall_confidence = float(analysis.get("overall_confidence", avg_section_conf))
    recommendation = analysis.get("recommendation", "continue")

    gaps: List[KnowledgeGap] = []
    for gap in analysis.get("gaps", []):
        if isinstance(gap, dict):
            gaps.append({
                "section": gap.get("section", "General"),
                "description": gap.get("description", ""),
                "suggested_queries": gap.get("suggested_queries", []),
                "priority": float(gap.get("priority", 0.5)),
                "current_confidence": section_confidence.get(gap.get("section", ""), 0),
            })

    should_continue = (
        recommendation == "continue" and
        current_iteration < max_iterations and
        overall_confidence < 0.8 and
        len(gaps) > 0
    )

    if should_continue:
        refinement_queries = []
        for gap in sorted(gaps, key=lambda g: g.get("priority", 0), reverse=True)[:3]:
            for query in gap.get("suggested_queries", [])[:2]:
                primary_anchor = state.get("primary_anchor", "")
                if primary_anchor and primary_anchor.lower() not in query.lower():
                    query = f'"{primary_anchor}" {query}'
                refinement_queries.append({
                    "query": query,
                    "section": gap.get("section", "General"),
                    "priority": gap.get("priority", 0.5),
                })

        return {
            "knowledge_gaps": gaps,
            "overall_confidence": overall_confidence,
            "section_confidence": analysis.get("section_confidence", section_confidence),
            "proceed_to_synthesis": False,
            "refinement_queries": refinement_queries,
            "research_iteration": current_iteration + 1,
        }
    else:
        return {
            "knowledge_gaps": gaps,
            "overall_confidence": overall_confidence,
            "section_confidence": analysis.get("section_confidence", section_confidence),
            "proceed_to_synthesis": True,
            "research_iteration": current_iteration,
        }


def route_after_gaps(state: AgentState) -> str:
    """Route based on gap detection results."""
    proceed = state.get("proceed_to_synthesis", False)
    dead_ends = state.get("dead_ends") or []

    unhandled_dead_ends = [d for d in dead_ends if not d.get("alternative_tried", False)]

    if unhandled_dead_ends and state.get("enable_backtracking", True):
        return "backtrack"

    if proceed:
        return "reduce"
    else:
        return "orchestrator"


def backtrack_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    When a search path fails, generate alternative approaches.

    Failure types:
    1. No results → broaden query
    2. Irrelevant results → refine with context
    3. Paywall/blocked → try alternative domains
    4. Low credibility → find authoritative sources
    """
    dead_ends = state.get("dead_ends") or []
    research_trajectory = state.get("research_trajectory") or []
    primary_anchor = state.get("primary_anchor", "")
    anchor_terms = state.get("anchor_terms") or []
    current_iteration = state.get("research_iteration", 0)

    if not dead_ends:
        return {}

    unhandled = [d for d in dead_ends if not d.get("alternative_tried", False)]

    if not unhandled:
        return {}

    alternative_queries = []
    updated_dead_ends = list(dead_ends)

    for dead_end in unhandled[:3]:  # Handle up to 3 dead ends per iteration
        reason = dead_end.get("reason", "unknown")
        failed_query = dead_end.get("query", "")

        alternatives = []

        if reason == "no_results":
            # Broaden: remove quotes, add synonyms
            alternatives = [
                failed_query.replace('"', ''),  # Remove quotes
                f"{failed_query} OR background OR profile",
                f"{primary_anchor} {anchor_terms[0] if anchor_terms else ''}".strip(),
            ]

        elif reason == "irrelevant":
            # Add more context to filter
            if anchor_terms:
                alternatives = [
                    f'"{primary_anchor}" {anchor_terms[0]}',
                    f'"{primary_anchor}" "{anchor_terms[0]}"' if anchor_terms else "",
                ]
            else:
                alternatives = [
                    f'"{primary_anchor}" profile biography',
                ]

        elif reason == "paywall":
            # Try alternative domains
            alternatives = [
                f'{failed_query} site:wikipedia.org',
                f'{failed_query} site:github.com OR site:linkedin.com',
                f'{failed_query} -site:wsj.com -site:nytimes.com -site:ft.com',
            ]

        elif reason == "low_credibility":
            # Target authoritative sources
            alternatives = [
                f'{failed_query} site:.edu',
                f'{failed_query} site:.gov OR site:.org',
                f'{primary_anchor} official',
            ]

        else:
            # Generic fallback
            alternatives = [
                f'"{primary_anchor}" {failed_query.replace(primary_anchor, "").strip()}',
            ]

        alternatives = [a.strip() for a in alternatives if a.strip()]
        alternative_queries.extend(alternatives[:2])

        idx = dead_ends.index(dead_end)
        updated_dead_ends[idx] = {**dead_end, "alternative_tried": True}

    trajectory_step: TrajectoryStep = {
        "iteration": current_iteration,
        "action": "backtrack",
        "query": f"Handling {len(unhandled)} dead ends",
        "result_summary": f"Generated {len(alternative_queries)} alternative queries",
        "confidence_delta": 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "dead_ends": updated_dead_ends,
        "research_trajectory": research_trajectory + [trajectory_step],
        "refinement_queries": [
            {"query": q, "section": "General", "priority": 0.7}
            for q in alternative_queries
        ],
    }


def complexity_router_node(state: AgentState) -> Dict[str, Any]:
    """Set routing parameters based on query complexity (simple/medium/complex)."""
    cfg = ResearchConfig()
    original_query = state.get("original_query", "")

    if not cfg.enable_complexity_routing:
        return {"query_complexity": "medium"}

    complexity = cfg.get_query_complexity(original_query)

    if complexity == "simple":
        return {
            "query_complexity": "simple",
            "max_subagents": 1,
            "max_research_iterations": 1,
            "enable_cross_validation": False,
            "batch_trust_engine": True,
            "_fast_path": True,
        }

    elif complexity == "complex":
        return {
            "query_complexity": "complex",
            "max_subagents": cfg.max_subagents,
            "max_research_iterations": cfg.max_research_iterations,
            "enable_cross_validation": True,
            "batch_trust_engine": cfg.batch_trust_engine,
            "_fast_path": False,
        }

    else:  # medium
        return {
            "query_complexity": "medium",
            "max_subagents": min(2, cfg.max_subagents),
            "max_research_iterations": min(2, cfg.max_research_iterations),
            "enable_cross_validation": cfg.enable_cross_validation,
            "batch_trust_engine": True,
            "_fast_path": False,
        }


def route_by_complexity(state: AgentState) -> str:
    """Route to appropriate path based on complexity."""
    fast_path = state.get("_fast_path", False)
    if fast_path:
        return "fast_path"
    return "standard_path"


def early_termination_check_node(state: AgentState) -> Dict[str, Any]:
    """Check whether research should stop early due to diminishing returns or budget."""
    cfg = ResearchConfig()

    if not cfg.enable_early_termination:
        return {"should_terminate": False}

    current_confidence = state.get("overall_confidence", 0.0)
    previous_confidence = state.get("previous_confidence", 0.0)
    new_sources_found = len(state.get("raw_sources", [])) - state.get("previous_source_count", 0)
    iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_research_iterations", cfg.max_research_iterations)

    cost_tracker = None
    if cfg.cost_budget_usd > 0:
        cost_tracker = CostTracker(budget_usd=cfg.cost_budget_usd)
        cost_tracker.total_cost = iteration * 0.05  # ~$0.05 per iteration estimate

    should_terminate, reason = should_terminate_early(
        current_confidence=current_confidence,
        previous_confidence=previous_confidence,
        new_sources_found=new_sources_found,
        iteration=iteration,
        max_iterations=max_iterations,
        min_confidence_delta=cfg.min_confidence_delta,
        min_new_sources=cfg.min_new_sources_threshold,
        cost_tracker=cost_tracker,
    )

    return {
        "should_terminate": should_terminate,
        "termination_reason": reason,
        "previous_confidence": current_confidence,
        "previous_source_count": len(state.get("raw_sources", [])),
    }


def route_after_termination_check(state: AgentState) -> str:
    """Route based on early termination check."""
    if state.get("should_terminate", False):
        return "reduce"  # Skip to synthesis
    return "gap_detector"  # Continue normal flow
