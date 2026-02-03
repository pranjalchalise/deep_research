# src/nodes/planner_v9.py
"""
V9 Research Planner - LLM-Driven Plan Generation

This module generates research plans based on the query analysis,
WITHOUT using hardcoded templates.

Key Principles (from industry research):
- OpenAI: Let the model decide search strategies dynamically
- Anthropic: Structured decomposition into parallel subtasks
- Google: Show plan to user, allow modification
- Perplexity: Semantic understanding drives query generation

The planner:
1. Takes query_analysis from the analyzer
2. Uses enriched_context if clarification was provided
3. Generates research questions tailored to the ACTUAL intent
4. Creates an outline that matches the topic, not a template
5. Generates optimized search queries for each question
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Plan, ResearchQuestion, ResearchTree, PlanQuery
from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# RESEARCH PLANNER PROMPT
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are an expert research planner. Your task is to create a comprehensive research plan based on the query analysis.

## Your Goal

Create a research plan that will thoroughly answer the user's question. The plan should:
1. Break down the research into specific, answerable questions
2. Generate optimized search queries for each question
3. Create a logical outline for the final report
4. Prioritize the most important aspects

## Query Analysis

Intent: {intent}
Query Class: {query_class}
Primary Subject: {primary_subject}
Topic Focus: {topic_focus}
Temporal Scope: {temporal_scope}
Domain: {domain}
Complexity: {complexity}

{enriched_context_section}

## Pre-generated Questions (if available)
{suggested_questions}

## Guidelines for Different Query Types

### For current_events / policy_analysis:
- Include recent time qualifiers (2024, 2025, "latest", "recent")
- Search for official announcements, news coverage, expert analysis
- Include multiple perspectives (supporters, critics, experts)

### For person_profile / person_work:
- Use quoted full names: "John Smith"
- Include identifying context (organization, field)
- Search LinkedIn, official pages, publications

### For concept_explanation / technical_docs:
- Include "explained", "how it works", "documentation"
- Search official docs, tutorials, technical deep-dives
- Include practical examples

### For comparative_analysis:
- Search "X vs Y", "X compared to Y"
- Look for pros/cons, differences, use cases
- Include expert opinions

## Search Query Optimization Rules

1. **Length**: 3-7 words per query (optimal for search APIs)
2. **Quotes**: Use quotes for multi-word proper nouns: "Donald Trump", "machine learning"
3. **Recency**: Add year/recency for current events: "2025", "latest", "recent"
4. **Specificity**: Be specific, not generic
5. **Variety**: Mix query types:
   - Exact: "Trump immigration policy executive order"
   - Broad: Trump border policy changes
   - Source-specific: Trump immigration policy official announcement

## Output Format

Return ONLY valid JSON:
{{
  "topic": "Clear, specific topic title",
  "outline": [
    "Section 1 - directly relevant to the query",
    "Section 2",
    "Section 3",
    "Section 4",
    "Section 5"
  ],
  "research_tree": {{
    "primary": [
      {{
        "question": "Most important research question",
        "queries": ["optimized query 1", "optimized query 2", "optimized query 3"],
        "target_sources": ["news", "official", "analysis"],
        "priority": 1.0
      }}
    ],
    "secondary": [
      {{
        "question": "Supporting research question",
        "queries": ["query 1", "query 2"],
        "target_sources": ["general"],
        "priority": 0.7
      }}
    ]
  }},
  "search_strategy": "Brief description of the overall search approach",
  "expected_challenges": ["potential challenge 1", "potential challenge 2"]
}}

## Examples

### Example: Current Events Query
Query Analysis:
- Intent: Understand recent Trump immigration policies
- Query Class: current_events
- Primary Subject: Trump administration immigration policy
- Temporal Scope: recent
- Domain: politics

Output:
{{
  "topic": "Trump Administration Immigration Policies (2024-2025)",
  "outline": [
    "Executive Summary",
    "Key Policy Changes",
    "Executive Orders and Legal Framework",
    "Border Security Measures",
    "Deportation Policies",
    "Legal Challenges and Court Decisions",
    "Humanitarian Impact",
    "Expert Analysis and Perspectives",
    "Comparison with Previous Administration"
  ],
  "research_tree": {{
    "primary": [
      {{
        "question": "What are the key immigration policy changes under Trump in 2024-2025?",
        "queries": [
          "Trump immigration policy changes 2025",
          "Trump executive orders immigration 2024 2025",
          "Trump border policy latest news"
        ],
        "target_sources": ["news", "official"],
        "priority": 1.0
      }},
      {{
        "question": "What executive orders has Trump signed on immigration?",
        "queries": [
          "Trump immigration executive orders list 2025",
          "Trump border security executive order",
          "Trump deportation executive order text"
        ],
        "target_sources": ["official", "legal"],
        "priority": 0.95
      }},
      {{
        "question": "What legal challenges have been filed against these policies?",
        "queries": [
          "Trump immigration policy court challenges 2025",
          "Trump deportation policy lawsuit",
          "immigration policy legal analysis Trump"
        ],
        "target_sources": ["news", "legal"],
        "priority": 0.9
      }}
    ],
    "secondary": [
      {{
        "question": "What do immigration experts say about these policies?",
        "queries": [
          "immigration experts analysis Trump policy",
          "Trump immigration policy expert opinion"
        ],
        "target_sources": ["analysis", "academic"],
        "priority": 0.7
      }},
      {{
        "question": "What is the humanitarian impact of these policies?",
        "queries": [
          "Trump immigration policy humanitarian impact",
          "Trump deportation families affected"
        ],
        "target_sources": ["news", "advocacy"],
        "priority": 0.6
      }}
    ]
  }},
  "search_strategy": "Start with recent news and official announcements for policy overview, then search for legal analysis and expert commentary. Include both supportive and critical perspectives.",
  "expected_challenges": ["Rapidly changing news landscape", "Distinguishing policy from rhetoric", "Finding balanced expert opinions"]
}}
"""


# ============================================================================
# PLANNER NODE
# ============================================================================

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a research plan based on query analysis.

    This node:
    1. Takes the query_analysis from the analyzer
    2. Uses enriched_context if clarification was provided
    3. Calls LLM to generate a tailored research plan
    4. Structures the plan for downstream orchestrator

    No hardcoded templates - the LLM generates plans based on actual intent.

    Returns:
        Dict with research plan, queries, and worker setup
    """
    # Get inputs
    original_query = state.get("original_query", "")
    query_analysis = state.get("query_analysis") or {}
    enriched_context = state.get("enriched_context")

    # Extract analysis fields
    intent = query_analysis.get("intent", original_query)
    query_class = query_analysis.get("query_class", "general")
    primary_subject = query_analysis.get("primary_subject", "")
    topic_focus = query_analysis.get("topic_focus")
    temporal_scope = query_analysis.get("temporal_scope", "timeless")
    domain = query_analysis.get("domain", "general")
    complexity = query_analysis.get("complexity", "medium")
    suggested_questions = query_analysis.get("suggested_questions", [])

    # Build enriched context section if available
    enriched_context_section = ""
    if enriched_context:
        enriched_context_section = f"""
## Enriched Context (from user clarification)
{enriched_context}
"""

    # Format suggested questions
    suggested_questions_str = "None provided" if not suggested_questions else "\n".join(
        f"- {q}" for q in suggested_questions
    )

    # Call LLM to generate plan
    llm = create_chat_model(model="gpt-4o", temperature=0.2)

    prompt = PLANNER_SYSTEM_PROMPT.format(
        intent=intent,
        query_class=query_class,
        primary_subject=primary_subject,
        topic_focus=topic_focus or "General overview",
        temporal_scope=temporal_scope,
        domain=domain,
        complexity=complexity,
        enriched_context_section=enriched_context_section,
        suggested_questions=suggested_questions_str,
    )

    response = llm.invoke([
        SystemMessage(content="You are a research planning expert. Return only valid JSON."),
        HumanMessage(content=prompt + f"\n\nOriginal query: {original_query}"),
    ])

    result = parse_json_object(response.content, default={})

    # Extract plan components
    topic = result.get("topic", primary_subject or original_query[:100])
    outline = result.get("outline", ["Overview", "Key Findings", "Analysis", "Conclusion"])
    research_tree = result.get("research_tree", {})
    search_strategy = result.get("search_strategy", "")

    # Build flat queries list for compatibility
    all_queries: List[PlanQuery] = []
    qid_counter = 1

    # Process primary questions
    for q in research_tree.get("primary", []):
        question = q.get("question", "")
        for query_text in q.get("queries", []):
            all_queries.append({
                "qid": f"Q{qid_counter}",
                "query": query_text,
                "section": _map_question_to_section(question, outline),
                "lane": _determine_lane(q.get("target_sources", [])),
                "priority": q.get("priority", 1.0),
            })
            qid_counter += 1

    # Process secondary questions
    for q in research_tree.get("secondary", []):
        question = q.get("question", "")
        for query_text in q.get("queries", []):
            all_queries.append({
                "qid": f"Q{qid_counter}",
                "query": query_text,
                "section": _map_question_to_section(question, outline),
                "lane": _determine_lane(q.get("target_sources", [])),
                "priority": q.get("priority", 0.7),
            })
            qid_counter += 1

    # Ensure we have at least some queries
    if not all_queries:
        all_queries = _generate_fallback_queries(primary_subject, query_class, outline)

    # Build the plan
    plan: Plan = {
        "topic": topic,
        "outline": outline,
        "queries": all_queries,
        "research_tree": research_tree,
    }

    # Log for debugging
    print(f"\n[Planner] Topic: {topic}")
    print(f"[Planner] Outline: {outline}")
    print(f"[Planner] Generated {len(all_queries)} queries")
    print(f"[Planner] Strategy: {search_strategy[:100]}...")

    return {
        "plan": plan,
        "total_workers": len(all_queries),
        "done_workers": 0,
        "raw_sources": [],
        "raw_evidence": [],
        "sources": None,
        "evidence": None,
        "claims": None,
        "citations": None,
        "issues": [],
        # Initialize research iteration
        "research_iteration": 0,
        "max_iterations": 3,
        "overall_confidence": 0.0,
        "previous_confidence": 0.0,
    }


def _map_question_to_section(question: str, outline: List[str]) -> str:
    """Map a research question to the most relevant outline section."""
    question_lower = question.lower()

    # Simple keyword matching
    for section in outline:
        section_lower = section.lower()
        # Check for word overlap
        section_words = set(section_lower.split())
        question_words = set(question_lower.split())
        if section_words & question_words:
            return section

    # Default to first content section (skip "Executive Summary" if present)
    for section in outline:
        if "summary" not in section.lower() and "overview" not in section.lower():
            return section

    return outline[0] if outline else "General"


def _determine_lane(target_sources: List[str]) -> str:
    """Determine the search lane based on target sources."""
    if not target_sources:
        return "general"

    # Map source types to lanes
    lane_mapping = {
        "news": "news",
        "official": "general",
        "legal": "general",
        "academic": "papers",
        "analysis": "general",
        "docs": "docs",
        "documentation": "docs",
        "code": "code",
        "github": "code",
        "community": "forums",
        "forums": "forums",
        "advocacy": "general",
    }

    for source in target_sources:
        if source.lower() in lane_mapping:
            return lane_mapping[source.lower()]

    return "general"


def _generate_fallback_queries(
    primary_subject: str,
    query_class: str,
    outline: List[str]
) -> List[PlanQuery]:
    """Generate fallback queries if LLM fails to generate any."""
    queries = []

    # Basic queries based on subject
    base_queries = [
        f'"{primary_subject}" overview',
        f'"{primary_subject}" explained',
        f'{primary_subject} latest news',
        f'{primary_subject} analysis',
    ]

    for i, query_text in enumerate(base_queries, 1):
        queries.append({
            "qid": f"Q{i}",
            "query": query_text,
            "section": outline[min(i-1, len(outline)-1)] if outline else "General",
            "lane": "general",
            "priority": 1.0 - (i * 0.1),
        })

    return queries


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "planner_node",
]
