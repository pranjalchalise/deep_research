# src/nodes/gap_detector.py
"""
V9 Knowledge Gap Detector - Iterative Research Loop Control

This is the KEY decision node that enables "deep" research:
1. Analyzes current research findings vs. the research plan
2. Identifies sections with insufficient coverage
3. Generates targeted refinement queries for gaps
4. Decides: LOOP BACK (more research) or CONTINUE (sufficient)

Inspired by:
- OpenAI: "backtrack when paths are unfruitful, pivot strategies"
- Anthropic: "Lead agent assesses if more research needed"
- Google: "iteratively plans, identifies gaps, searches again"

Flow:
    SYNTHESIZER → GAP_DETECTOR → [has gaps?] → ORCHESTRATOR (loop)
                                            → TRUST_ENGINE (continue)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import (
    AgentState,
    KnowledgeGap,
    TrajectoryStep,
)
from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# GAP DETECTION PROMPT
# ============================================================================

GAP_DETECTOR_PROMPT = """You are analyzing research findings to identify knowledge gaps.

## Research Plan

Topic: {topic}
Outline Sections: {outline}
Original Intent: {intent}

## Current Findings

We have collected {num_sources} sources and {num_evidence} evidence items.

### Evidence by Section:
{evidence_by_section}

## Your Task

Analyze the coverage and identify gaps:

1. **Section Coverage**: For each outline section, assess:
   - How well is it covered? (0.0 = no coverage, 1.0 = comprehensive)
   - What specific information is missing?

2. **Gap Identification**: List sections that need more research:
   - What's missing?
   - What queries would help fill the gap?

3. **Recommendation**: Should we continue researching or is coverage sufficient?

## Decision Criteria

CONTINUE RESEARCHING if:
- Overall confidence < 0.7 AND this is iteration 0-1
- Any critical section has confidence < 0.5
- Major questions from the plan are unanswered

STOP RESEARCHING if:
- Overall confidence >= 0.8
- All critical sections have confidence >= 0.6
- Improvement from last iteration < 5% (diminishing returns)
- Max iterations reached

## Output Format

Return ONLY valid JSON:
{{
  "section_analysis": [
    {{
      "section": "Section Name",
      "confidence": 0.0-1.0,
      "evidence_count": 5,
      "coverage_notes": "What's covered well",
      "missing": "What's missing or incomplete"
    }}
  ],
  "overall_confidence": 0.0-1.0,
  "gaps": [
    {{
      "section": "Section with gap",
      "severity": "high|medium|low",
      "description": "What information is missing",
      "suggested_queries": ["query to fill gap 1", "query to fill gap 2"]
    }}
  ],
  "conflicts": [
    {{
      "topic": "Conflicting topic",
      "sources": ["S1", "S3"],
      "description": "Nature of the conflict"
    }}
  ],
  "recommendation": "continue|sufficient",
  "reasoning": "Why this recommendation"
}}
"""


# ============================================================================
# GAP DETECTOR NODE
# ============================================================================

def gap_detector_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze research findings and identify knowledge gaps.

    This node:
    1. Groups evidence by outline section
    2. Calculates coverage confidence per section
    3. Identifies gaps that need more research
    4. Decides whether to iterate or proceed

    Returns:
        State updates including gaps, confidence, and refinement queries
    """
    # Get current state
    plan = state.get("plan") or {}
    evidence = state.get("evidence") or []
    sources = state.get("sources") or []
    query_analysis = state.get("query_analysis") or {}

    outline = plan.get("outline", [])
    topic = plan.get("topic", "Unknown")
    intent = query_analysis.get("intent", "")

    current_iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    previous_confidence = state.get("overall_confidence", 0.0)

    # Early exit if no evidence
    if not evidence:
        return {
            "knowledge_gaps": [{
                "section": "All",
                "description": "No evidence collected",
                "suggested_queries": [],
                "priority": 1.0,
                "current_confidence": 0.0,
            }],
            "overall_confidence": 0.0,
            "section_confidence": {},
            "refinement_queries": [],
            "previous_confidence": previous_confidence,
        }

    # Group evidence by section
    evidence_by_section = _group_evidence_by_section(evidence, outline)

    # Format for LLM
    evidence_summary = _format_evidence_summary(evidence_by_section)

    # Call LLM for analysis
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    prompt = GAP_DETECTOR_PROMPT.format(
        topic=topic,
        outline=", ".join(outline),
        intent=intent,
        num_sources=len(sources),
        num_evidence=len(evidence),
        evidence_by_section=evidence_summary,
    )

    response = llm.invoke([
        SystemMessage(content="You analyze research coverage and identify gaps. Return only valid JSON."),
        HumanMessage(content=prompt),
    ])

    result = parse_json_object(response.content, default={})

    # Extract results
    section_analysis = result.get("section_analysis", [])
    overall_confidence = float(result.get("overall_confidence", 0.5))
    gaps = result.get("gaps", [])
    conflicts = result.get("conflicts", [])
    recommendation = result.get("recommendation", "sufficient")
    reasoning = result.get("reasoning", "")

    # Build section confidence dict
    section_confidence: Dict[str, float] = {}
    for sa in section_analysis:
        if isinstance(sa, dict):
            section_confidence[sa.get("section", "")] = float(sa.get("confidence", 0.5))

    # Fill in missing sections with heuristic confidence
    for section in outline:
        if section not in section_confidence:
            # Heuristic: count evidence items mentioning this section
            count = len(evidence_by_section.get(section, []))
            section_confidence[section] = min(1.0, count / 3)  # 3+ items = full confidence

    # Build knowledge gaps list
    knowledge_gaps: List[KnowledgeGap] = []
    for gap in gaps:
        if isinstance(gap, dict):
            severity = gap.get("severity", "medium")
            priority = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(severity, 0.7)

            knowledge_gaps.append({
                "section": gap.get("section", "Unknown"),
                "description": gap.get("description", ""),
                "suggested_queries": gap.get("suggested_queries", []),
                "priority": priority,
                "current_confidence": section_confidence.get(gap.get("section", ""), 0.5),
            })

    # Build refinement queries from gaps
    refinement_queries: List[Dict[str, Any]] = []
    for gap in knowledge_gaps:
        for query in gap.get("suggested_queries", [])[:2]:  # Max 2 queries per gap
            refinement_queries.append({
                "query": query,
                "section": gap.get("section", "General"),
                "priority": gap.get("priority", 0.7),
            })

    # Override recommendation based on hard rules
    confidence_delta = overall_confidence - previous_confidence

    # Check stopping conditions
    should_stop = False
    stop_reason = ""

    if current_iteration >= max_iterations - 1:
        should_stop = True
        stop_reason = f"Max iterations ({max_iterations}) reached"
    elif overall_confidence >= 0.85:
        should_stop = True
        stop_reason = f"High confidence ({overall_confidence:.2f}) achieved"
    elif confidence_delta < 0.05 and current_iteration > 0:
        should_stop = True
        stop_reason = f"Diminishing returns (delta={confidence_delta:.2f})"
    elif not knowledge_gaps:
        should_stop = True
        stop_reason = "No gaps identified"

    if should_stop:
        recommendation = "sufficient"
        refinement_queries = []  # Clear queries if stopping

    # Add trajectory step
    trajectory = state.get("research_trajectory") or []
    step: TrajectoryStep = {
        "iteration": current_iteration,
        "action": "gap_detection",
        "query": f"Analyzed {len(evidence)} evidence items",
        "result_summary": f"Confidence: {overall_confidence:.2f}, Gaps: {len(knowledge_gaps)}, Recommendation: {recommendation}",
        "confidence_delta": confidence_delta,
        "timestamp": datetime.now().isoformat(),
    }

    # Log for debugging
    print(f"\n[Gap Detector] Iteration {current_iteration}")
    print(f"[Gap Detector] Evidence: {len(evidence)}, Sources: {len(sources)}")
    print(f"[Gap Detector] Confidence: {previous_confidence:.2f} → {overall_confidence:.2f} (Δ{confidence_delta:+.2f})")
    print(f"[Gap Detector] Gaps: {len(knowledge_gaps)}")
    print(f"[Gap Detector] Recommendation: {recommendation}")
    if stop_reason:
        print(f"[Gap Detector] Stop reason: {stop_reason}")

    return {
        "knowledge_gaps": knowledge_gaps,
        "section_confidence": section_confidence,
        "overall_confidence": overall_confidence,
        "previous_confidence": previous_confidence,  # Store for next iteration
        "refinement_queries": refinement_queries,
        "research_trajectory": trajectory + [step],
        # Flag for routing
        "proceed_to_synthesis": recommendation == "sufficient",
    }


def _group_evidence_by_section(
    evidence: List[Dict],
    outline: List[str]
) -> Dict[str, List[Dict]]:
    """Group evidence items by outline section."""
    grouped: Dict[str, List[Dict]] = {section: [] for section in outline}
    grouped["Unassigned"] = []

    for e in evidence:
        if not isinstance(e, dict):
            continue

        section = e.get("section", "")
        text = e.get("text", "")

        # Try to match to outline section
        matched = False
        for outline_section in outline:
            if section.lower() == outline_section.lower():
                grouped[outline_section].append(e)
                matched = True
                break
            # Also check for partial match
            elif outline_section.lower() in section.lower() or section.lower() in outline_section.lower():
                grouped[outline_section].append(e)
                matched = True
                break

        if not matched:
            # Try to infer from text content
            text_lower = text.lower()
            for outline_section in outline:
                section_words = outline_section.lower().split()
                if any(word in text_lower for word in section_words if len(word) > 3):
                    grouped[outline_section].append(e)
                    matched = True
                    break

        if not matched:
            grouped["Unassigned"].append(e)

    return grouped


def _format_evidence_summary(evidence_by_section: Dict[str, List[Dict]]) -> str:
    """Format evidence summary for LLM."""
    lines = []

    for section, items in evidence_by_section.items():
        if not items and section != "Unassigned":
            lines.append(f"\n### {section}\nNo evidence collected for this section.")
        elif items:
            lines.append(f"\n### {section} ({len(items)} items)")
            for item in items[:3]:  # Show max 3 per section
                text = item.get("text", "")[:200]
                lines.append(f"- {text}...")

    return "\n".join(lines)


# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def route_after_gap_detection(state: AgentState) -> str:
    """
    Route based on gap detection results.

    Returns:
        "orchestrator" - Has gaps, need more research (loop back)
        "trust_engine" - Sufficient coverage, proceed to verification
    """
    proceed = state.get("proceed_to_synthesis", False)
    gaps = state.get("knowledge_gaps") or []
    refinement_queries = state.get("refinement_queries") or []
    iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # If proceed flag is set, go to trust engine
    if proceed:
        print("[Gap Router] → trust_engine (sufficient coverage)")
        return "trust_engine"

    # If we have refinement queries and haven't hit max iterations
    if refinement_queries and iteration < max_iterations - 1:
        print(f"[Gap Router] → orchestrator (iteration {iteration + 1}, {len(refinement_queries)} queries)")
        return "orchestrator"

    # Default: proceed to trust engine
    print("[Gap Router] → trust_engine (default)")
    return "trust_engine"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "gap_detector_node",
    "route_after_gap_detection",
]
