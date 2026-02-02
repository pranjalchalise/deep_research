# src/utils/semantic.py
"""
Semantic Query Understanding and Reformulation.

This module transforms user queries into semantically-rich research briefs
and generates optimized search queries based on facets.

Inspired by industry patterns:
- OpenAI Deep Research: Model learns to generate effective queries
- Perplexity: Vertical-specific trust tiers
- Exa: Meaning-based search via embeddings
- Gemini: Multi-step research plans with facets

Key insight: Instead of filtering domains (brittle), we reformulate queries
to naturally find the right content (robust).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# FACET DEFINITIONS
# ============================================================================

# Facets represent different aspects/dimensions of a research query.
# Each facet has query hints that help find relevant content WITHOUT domain filtering.

FACET_HINTS: Dict[str, Dict[str, Any]] = {
    "background": {
        "description": "Basic info, biography, profile, identity",
        "query_hints": ["profile", "biography", "about", "background", "who is"],
        "priority": 1,
    },
    "professional": {
        "description": "Work, career, role, position, company",
        "query_hints": ["work", "career", "role", "position", "job", "company"],
        "priority": 2,
    },
    "academic": {
        "description": "Research, publications, papers, studies",
        "query_hints": ["research", "paper", "publication", "study", "journal", "thesis"],
        "priority": 2,
    },
    "achievements": {
        "description": "Awards, recognition, accomplishments",
        "query_hints": ["award", "achievement", "recognition", "accomplishment", "honor"],
        "priority": 3,
    },
    "news": {
        "description": "Recent activity, announcements, updates",
        "query_hints": ["news", "recent", "announcement", "update", "2024", "2025"],
        "priority": 3,
    },
    "opinions": {
        "description": "Discussions, reviews, community views",
        "query_hints": ["discussion", "review", "opinion", "reddit", "forum"],
        "priority": 4,
    },
    "technical": {
        "description": "How it works, implementation, code",
        "query_hints": ["how", "works", "implementation", "code", "github", "tutorial"],
        "priority": 2,
    },
    "comparison": {
        "description": "Versus, differences, alternatives",
        "query_hints": ["vs", "versus", "comparison", "difference", "alternative"],
        "priority": 2,
    },
    "definition": {
        "description": "What is, meaning, explanation",
        "query_hints": ["what is", "definition", "meaning", "explained", "overview"],
        "priority": 1,
    },
    "applications": {
        "description": "Use cases, examples, real-world usage",
        "query_hints": ["use case", "example", "application", "usage", "real-world"],
        "priority": 3,
    },
    "profiles": {
        "description": "Online presence and profiles",
        "query_hints": ["profile", "online", "social", "portfolio", "website"],
        "priority": 2,
    },
}


# ============================================================================
# SEMANTIC ANALYZER PROMPT
# ============================================================================

SEMANTIC_SYSTEM = """You are a semantic query analyzer that transforms user queries into comprehensive research briefs.

Your job is to:
1. Understand the FULL semantic meaning of the query
2. Identify the core subject and context
3. Determine what FACETS (aspects) of information the user implicitly wants
4. Generate a detailed research brief

Return ONLY valid JSON:
{
  "subject": "The main entity/topic being researched",
  "subject_type": "person|concept|technology|organization|event|comparison",
  "context": ["contextual qualifiers that help identify the subject"],
  "intent": "What the user actually wants to know (1 sentence)",
  "research_brief": "A detailed 2-3 sentence description of what to research, making implicit questions explicit",
  "facets": ["list of relevant facets from: background, professional, academic, achievements, news, opinions, technical, comparison, definition, applications, profiles"],
  "implicit_questions": ["List of specific questions the user implicitly wants answered"]
}

Examples:

Query: "Tell me about Elon Musk"
→ subject: "Elon Musk"
→ subject_type: "person"
→ facets: ["background", "professional", "achievements", "news", "profiles"]
→ implicit_questions: ["Who is Elon Musk?", "What companies does he lead?", "What are his major achievements?", "What is he currently working on?"]

Query: "How does React useState work"
→ subject: "React useState hook"
→ subject_type: "technology"
→ facets: ["definition", "technical", "applications"]
→ implicit_questions: ["What is useState?", "How does it work internally?", "When should I use it?", "What are common patterns?"]

Query: "Compare PostgreSQL vs MySQL"
→ subject: "PostgreSQL vs MySQL"
→ subject_type: "comparison"
→ facets: ["comparison", "technical", "applications", "opinions"]
→ implicit_questions: ["What are the key differences?", "Which is better for what use cases?", "What do developers prefer?"]
"""


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def analyze_query_semantically(query: str, existing_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyze a query to extract its full semantic meaning.

    Args:
        query: The user's research query
        existing_context: Optional context from discovery phase (primary_anchor, etc.)

    Returns:
        Semantic analysis with subject, facets, research brief, etc.
    """
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    context_info = ""
    if existing_context:
        context_info = f"""
Additional context from discovery:
- Primary subject: {existing_context.get('primary_anchor', 'Unknown')}
- Subject type: {existing_context.get('query_type', 'general')}
- Context terms: {existing_context.get('anchor_terms', [])}
"""

    resp = llm.invoke([
        SystemMessage(content=SEMANTIC_SYSTEM),
        HumanMessage(content=f"Query: {query}\n{context_info}"),
    ])

    analysis = parse_json_object(resp.content, default={
        "subject": query,
        "subject_type": "general",
        "context": [],
        "intent": f"Research about {query}",
        "research_brief": f"Find comprehensive information about {query}",
        "facets": ["background", "definition"],
        "implicit_questions": [f"What is {query}?"],
    })

    return analysis


def generate_facet_queries(
    subject: str,
    facets: List[str],
    context: Optional[List[str]] = None,
    max_queries: int = 8,
) -> List[Dict[str, Any]]:
    """
    Generate optimized search queries for each facet.

    Instead of domain filtering, we use semantic query reformulation:
    - Add facet-specific keywords that naturally lead to relevant sources
    - Quote the subject for exact matching
    - Include context for disambiguation

    Args:
        subject: The main subject to research
        facets: List of facets to generate queries for
        context: Optional context terms for disambiguation
        max_queries: Maximum number of queries to generate

    Returns:
        List of query dicts with qid, query, section, facet
    """
    queries = []
    quoted_subject = f'"{subject}"' if ' ' in subject else subject
    context_str = f" {context[0]}" if context else ""

    # Sort facets by priority
    sorted_facets = sorted(
        facets,
        key=lambda f: FACET_HINTS.get(f, {}).get("priority", 99)
    )

    for facet in sorted_facets[:max_queries]:
        facet_info = FACET_HINTS.get(facet, {})
        hints = facet_info.get("query_hints", [])
        description = facet_info.get("description", facet)

        if not hints:
            continue

        # Pick 1-2 hints for the query
        hint_str = " ".join(hints[:2])

        # Build the query with semantic hints (no domain filtering needed)
        query_text = f'{quoted_subject}{context_str} {hint_str}'

        queries.append({
            "qid": f"Q{len(queries)+1}",
            "query": query_text.strip(),
            "section": description.title(),
            "facet": facet,
            "lane": "news" if facet == "news" else "general",  # Only news uses special lane
        })

    # Add a broad fallback query
    if len(queries) < max_queries:
        queries.append({
            "qid": f"Q{len(queries)+1}",
            "query": f'{quoted_subject}{context_str}',
            "section": "Overview",
            "facet": "general",
            "lane": "general",
        })

    return queries


def build_research_plan(
    query: str,
    semantic_analysis: Dict[str, Any],
    max_queries: int = 8,
) -> Dict[str, Any]:
    """
    Build a complete research plan from semantic analysis.

    Args:
        query: Original user query
        semantic_analysis: Output from analyze_query_semantically()
        max_queries: Maximum search queries to generate

    Returns:
        Research plan with topic, outline, queries, and brief
    """
    subject = semantic_analysis.get("subject", query)
    facets = semantic_analysis.get("facets", ["background", "definition"])
    context = semantic_analysis.get("context", [])
    research_brief = semantic_analysis.get("research_brief", "")
    implicit_questions = semantic_analysis.get("implicit_questions", [])

    # Generate queries based on facets
    queries = generate_facet_queries(
        subject=subject,
        facets=facets,
        context=context,
        max_queries=max_queries,
    )

    # Build outline from facets
    outline = [FACET_HINTS.get(f, {}).get("description", f).title() for f in facets]
    outline = ["Overview"] + outline + ["Conclusion"]

    return {
        "topic": subject,
        "outline": outline,
        "queries": queries,
        "research_brief": research_brief,
        "implicit_questions": implicit_questions,
    }
