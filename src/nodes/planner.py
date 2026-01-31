# src/nodes/planner.py
"""
Research planner with optimized search query generation.

Query Optimization Strategies:
1. Optimal length: 3-6 words per query
2. Quote full names for exact matching
3. 3-2-1 Rule: 3 precise, 2 medium, 1 broad queries
4. Query templates for different purposes (profile, academic, news, etc.)
5. Smart primary + context combination
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Plan, Lane
from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# QUERY TEMPLATES - Optimized for web search
# ============================================================================

def generate_person_queries(
    name: str,
    context_terms: List[str],
    query_type: str = "person"
) -> List[Dict[str, Any]]:
    """
    Generate optimized search queries for a person.

    Uses the 3-2-1 strategy:
    - 3 HIGH PRECISION queries (quoted name + specific context)
    - 2 MEDIUM queries (quoted name + broader terms)
    - 1 BROAD query (fallback)

    Query patterns based on search optimization research:
    - Quotes around full name for exact match
    - 3-6 words optimal length
    - Context terms for disambiguation
    """
    queries = []
    quoted_name = f'"{name}"'

    # Get primary context (most important identifier)
    primary_context = context_terms[0] if context_terms else ""
    secondary_context = context_terms[1] if len(context_terms) > 1 else ""

    # ====== HIGH PRECISION QUERIES (3) ======
    # These use quoted name + specific context

    # Q1: Profile/Identity query
    if primary_context:
        queries.append({
            "query": f'{quoted_name} {primary_context}',
            "section": "Background",
            "lane": "general",
            "purpose": "identity"
        })
    else:
        queries.append({
            "query": f'{quoted_name} profile biography',
            "section": "Background",
            "lane": "general",
            "purpose": "identity"
        })

    # Q2: Professional/Work query
    if primary_context:
        queries.append({
            "query": f'{quoted_name} {primary_context} work projects',
            "section": "Work & Achievements",
            "lane": "general",
            "purpose": "work"
        })
    else:
        queries.append({
            "query": f'{quoted_name} work achievements career',
            "section": "Work & Achievements",
            "lane": "general",
            "purpose": "work"
        })

    # Q3: LinkedIn/Social profile query (high signal for people)
    queries.append({
        "query": f'{quoted_name} LinkedIn',
        "section": "Background",
        "lane": "general",
        "purpose": "social_profile"
    })

    # ====== MEDIUM PRECISION QUERIES (2-3) ======

    # Q4: Academic/Research query
    queries.append({
        "query": f'{quoted_name} research publications',
        "section": "Publications",
        "lane": "papers",
        "purpose": "academic"
    })

    # Q5: News/Recent activity
    if primary_context:
        queries.append({
            "query": f'{quoted_name} {primary_context} news',
            "section": "Recent Activity",
            "lane": "news",
            "purpose": "news"
        })
    else:
        queries.append({
            "query": f'{quoted_name} news announcement',
            "section": "Recent Activity",
            "lane": "news",
            "purpose": "news"
        })

    # Q6: Discussion/Community mentions
    queries.append({
        "query": f'{quoted_name} {primary_context}' if primary_context else f'{name}',
        "section": "Discussions",
        "lane": "forums",
        "purpose": "community"
    })

    # ====== CONTEXT-SPECIFIC QUERIES ======

    # If we have secondary context, add targeted queries
    if secondary_context:
        queries.append({
            "query": f'{quoted_name} {secondary_context}',
            "section": "Background",
            "lane": "general",
            "purpose": "context_specific"
        })

    # ====== BROAD FALLBACK QUERY (1) ======
    # Unquoted for maximum recall
    queries.append({
        "query": f'{name} {primary_context} profile' if primary_context else f'{name} biography background',
        "section": "Overview",
        "lane": "general",
        "purpose": "broad_fallback"
    })

    # Add QIDs
    for i, q in enumerate(queries, 1):
        q["qid"] = f"Q{i}"

    return queries


def generate_concept_queries(
    concept: str,
    context_terms: List[str]
) -> List[Dict[str, Any]]:
    """Generate optimized queries for concept/technical research."""
    queries = []
    quoted = f'"{concept}"' if ' ' in concept else concept

    # Definition/Overview
    queries.append({
        "query": f'{quoted} definition explained',
        "section": "Overview",
        "lane": "general",
        "purpose": "definition"
    })

    # How it works
    queries.append({
        "query": f'{quoted} how it works',
        "section": "How It Works",
        "lane": "docs",
        "purpose": "explanation"
    })

    # Use cases/Applications
    queries.append({
        "query": f'{quoted} use cases applications',
        "section": "Applications",
        "lane": "general",
        "purpose": "applications"
    })

    # Research/Academic
    queries.append({
        "query": f'{quoted} research paper',
        "section": "Research",
        "lane": "papers",
        "purpose": "academic"
    })

    # Best practices
    queries.append({
        "query": f'{quoted} best practices guide',
        "section": "Best Practices",
        "lane": "docs",
        "purpose": "guide"
    })

    # Limitations/Critiques
    queries.append({
        "query": f'{quoted} limitations problems',
        "section": "Limitations",
        "lane": "forums",
        "purpose": "critique"
    })

    # Add context if available
    if context_terms:
        ctx = context_terms[0]
        queries.append({
            "query": f'{quoted} {ctx}',
            "section": "Context",
            "lane": "general",
            "purpose": "context"
        })

    for i, q in enumerate(queries, 1):
        q["qid"] = f"Q{i}"

    return queries


def generate_technical_queries(
    tech: str,
    context_terms: List[str]
) -> List[Dict[str, Any]]:
    """Generate optimized queries for technical/tool research."""
    queries = []

    # Quote multi-word tech terms
    quoted_tech = f'"{tech}"' if ' ' in tech else tech

    # Documentation
    queries.append({
        "query": f'{quoted_tech} documentation official',
        "section": "Documentation",
        "lane": "docs",
        "purpose": "docs"
    })

    # Tutorial/Getting started
    queries.append({
        "query": f'{quoted_tech} tutorial guide',
        "section": "Getting Started",
        "lane": "docs",
        "purpose": "tutorial"
    })

    # How it works
    queries.append({
        "query": f'{quoted_tech} how it works explained',
        "section": "How It Works",
        "lane": "general",
        "purpose": "explanation"
    })

    # Examples
    queries.append({
        "query": f'{quoted_tech} examples',
        "section": "Examples",
        "lane": "code",
        "purpose": "examples"
    })

    # GitHub/Source
    queries.append({
        "query": f'{quoted_tech} github',
        "section": "Source Code",
        "lane": "code",
        "purpose": "source"
    })

    # Best practices
    queries.append({
        "query": f'{quoted_tech} best practices',
        "section": "Best Practices",
        "lane": "general",
        "purpose": "best_practices"
    })

    # Issues/Troubleshooting
    queries.append({
        "query": f'{quoted_tech} common issues',
        "section": "Troubleshooting",
        "lane": "forums",
        "purpose": "troubleshooting"
    })

    # Add context-specific query if available
    if context_terms:
        queries.append({
            "query": f'{quoted_tech} {context_terms[0]}',
            "section": "Context",
            "lane": "general",
            "purpose": "context"
        })

    for i, q in enumerate(queries, 1):
        q["qid"] = f"Q{i}"

    return queries


def generate_comparison_queries(
    entity1: str,
    entity2: str,
    context_terms: List[str]
) -> List[Dict[str, Any]]:
    """Generate optimized queries for comparison research."""
    queries = []

    # Direct comparison
    queries.append({
        "query": f'{entity1} vs {entity2}',
        "section": "Comparison",
        "lane": "general",
        "purpose": "direct_comparison"
    })

    queries.append({
        "query": f'{entity1} vs {entity2} differences',
        "section": "Key Differences",
        "lane": "general",
        "purpose": "differences"
    })

    queries.append({
        "query": f'{entity1} vs {entity2} which is better',
        "section": "Recommendation",
        "lane": "forums",
        "purpose": "recommendation"
    })

    # Individual entity queries
    queries.append({
        "query": f'{entity1} advantages benefits',
        "section": f"{entity1} Pros",
        "lane": "general",
        "purpose": "entity1_pros"
    })

    queries.append({
        "query": f'{entity2} advantages benefits',
        "section": f"{entity2} Pros",
        "lane": "general",
        "purpose": "entity2_pros"
    })

    queries.append({
        "query": f'{entity1} disadvantages limitations',
        "section": f"{entity1} Cons",
        "lane": "forums",
        "purpose": "entity1_cons"
    })

    queries.append({
        "query": f'{entity2} disadvantages limitations',
        "section": f"{entity2} Cons",
        "lane": "forums",
        "purpose": "entity2_cons"
    })

    # Use cases
    queries.append({
        "query": f'when to use {entity1} vs {entity2}',
        "section": "When to Use",
        "lane": "general",
        "purpose": "use_cases"
    })

    for i, q in enumerate(queries, 1):
        q["qid"] = f"Q{i}"

    return queries


# ============================================================================
# LLM-BASED PLANNER (for complex/custom queries)
# ============================================================================

PLANNER_SYSTEM = """You are a search query optimization expert.

Generate search queries that follow these RULES:

1. QUERY FORMAT:
   - Use quotes around full names: "John Smith" MIT
   - Keep queries 3-6 words (optimal for search APIs)
   - Put most important terms first

2. QUERY TYPES (generate a mix):
   - PRECISE: "Full Name" + specific context (high precision)
   - MEDIUM: "Full Name" + broader terms (balanced)
   - BROAD: Name without quotes (high recall fallback)

3. For PERSON searches, include:
   - Profile query: "Name" affiliation
   - LinkedIn query: "Name" LinkedIn
   - Work query: "Name" projects/achievements
   - Academic query: "Name" research/publications

4. EXAMPLES of good queries:
   - "Pranjal Chalise" Amherst College
   - "Pranjal Chalise" LinkedIn
   - "Pranjal Chalise" Microsoft engineer
   - "Pranjal Chalise" projects research

Return ONLY valid JSON:
{
  "topic": "...",
  "outline": ["Section1", "Section2", ...],
  "queries": [
    {"qid": "Q1", "query": "...", "section": "...", "lane": "general|papers|news|forums|docs|code"}
  ]
}
"""


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a research plan with optimized search queries.

    Uses template-based query generation for common patterns,
    falls back to LLM for complex/custom queries.
    """
    original_query = state.get("original_query") or state["messages"][-1].content
    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []
    selected_entity = state.get("selected_entity") or {}
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")

    # If no primary anchor, try to get from selected entity or anchor_terms
    if not primary_anchor and selected_entity:
        primary_anchor = selected_entity.get("name", "")
    if not primary_anchor and anchor_terms:
        primary_anchor = anchor_terms[0]
        anchor_terms = anchor_terms[1:]

    # ====== TEMPLATE-BASED QUERY GENERATION ======
    # Use optimized templates for known query types

    if query_type == "person" and primary_anchor:
        queries = generate_person_queries(primary_anchor, anchor_terms, query_type)
        outline = ["Overview", "Background", "Work & Achievements", "Publications", "Recent Activity", "Discussions"]

    elif query_type == "comparison" and len(anchor_terms) >= 1:
        # Comparison query: entity1 vs entity2
        entity1 = primary_anchor or anchor_terms[0]
        entity2 = anchor_terms[0] if primary_anchor else (anchor_terms[1] if len(anchor_terms) > 1 else "")
        if entity2 and entity1 != entity2:
            queries = generate_comparison_queries(entity1, entity2, anchor_terms[2:] if len(anchor_terms) > 2 else [])
            outline = ["Overview", f"{entity1} Overview", f"{entity2} Overview", "Key Differences", "Pros & Cons", "When to Use", "Conclusion"]
        else:
            # Fallback to concept if only one entity
            queries = generate_concept_queries(entity1, anchor_terms)
            outline = ["Overview", "How It Works", "Applications", "Research", "Best Practices", "Limitations"]

    elif query_type == "technical" and primary_anchor:
        # Use the full technical term, not just the first word
        # Combine primary_anchor with context if it makes sense
        full_tech_term = primary_anchor
        if anchor_terms and len(primary_anchor.split()) == 1:
            # If primary is single word like "React", combine with context like "useState"
            full_tech_term = f"{primary_anchor} {anchor_terms[0]}"
            anchor_terms = anchor_terms[1:]
        queries = generate_technical_queries(full_tech_term, anchor_terms)
        outline = ["Overview", "How It Works", "Documentation", "Examples", "Best Practices", "Troubleshooting"]

    elif query_type == "concept" and primary_anchor:
        queries = generate_concept_queries(primary_anchor, anchor_terms)
        outline = ["Overview", "How It Works", "Applications", "Research", "Best Practices", "Limitations"]

    else:
        # ====== LLM-BASED GENERATION (fallback) ======
        queries = _generate_queries_with_llm(
            original_query, primary_anchor, anchor_terms, query_type, selected_entity
        )
        outline = ["Overview", "Key Findings", "Background", "Analysis", "Conclusion"]

    # Build the plan
    plan: Plan = {
        "topic": f"{primary_anchor}" if primary_anchor else f"Research: {original_query}",
        "outline": outline,
        "queries": queries,
    }

    # Ensure all queries have the primary anchor (safety check)
    if primary_anchor:
        _ensure_primary_anchor_in_queries(plan["queries"], primary_anchor)

    total_workers = len(plan["queries"])

    return {
        "plan": plan,
        "total_workers": total_workers,
        "done_workers": 0,
        "raw_sources": [],
        "raw_evidence": [],
        "sources": None,
        "evidence": None,
        "claims": None,
        "citations": None,
        "issues": [],
    }


def _generate_queries_with_llm(
    original_query: str,
    primary_anchor: str,
    anchor_terms: List[str],
    query_type: str,
    selected_entity: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate queries using LLM for complex/custom cases."""

    entity_context = ""
    if selected_entity:
        entity_context = f"""
Entity identified:
- Name: {selected_entity.get('name', 'Unknown')}
- Description: {selected_entity.get('description', '')}
- Key identifiers: {selected_entity.get('identifiers', [])}
"""

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)

    prompt = f"""User question: {original_query}

{entity_context}

PRIMARY ENTITY: "{primary_anchor}"
CONTEXT TERMS: {anchor_terms}
QUERY TYPE: {query_type}

Generate 6-10 optimized search queries following the rules in my instructions.
Remember:
- Quote the full name: "{primary_anchor}"
- Keep queries 3-6 words
- Mix of precise, medium, and broad queries
"""

    resp = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    result = parse_json_object(resp.content, default={})
    return result.get("queries", [])


def _ensure_primary_anchor_in_queries(queries: List[Dict], primary_anchor: str):
    """Ensure every query contains the primary anchor."""
    if not primary_anchor:
        return

    primary_lower = primary_anchor.lower()
    quoted_primary = f'"{primary_anchor}"'

    for q in queries:
        query_text = q.get("query", "")
        query_lower = query_text.lower()

        # Check if primary anchor (or quoted version) is in query
        if primary_lower not in query_lower:
            # Prepend quoted name
            q["query"] = f'{quoted_primary} {query_text}'
