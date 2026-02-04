"""
Research planner that decomposes queries into hierarchical research trees
(primary/secondary/tertiary questions) with optimized search queries.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from src.advanced.state import (
    AgentState,
    Plan,
    ResearchTree,
    ResearchQuestion,
    PlanQuery,
)
from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


PLANNER_V8_SYSTEM = """You are a research planning expert.

Create a RESEARCH TREE that decomposes the research question into structured questions.

Guidelines:
1. PRIMARY questions (3-4): Must answer to fulfill the research goal
2. SECONDARY questions (2-3): Should answer for comprehensive coverage
3. TERTIARY questions (1-2): Nice to have, deeper exploration

For each question, provide:
- 2-3 optimized search queries (3-6 words, quote full names)
- Target source types (academic, news, official, community)
- Confidence weight (how much this contributes to overall understanding)

Query optimization rules:
- Use quotes around full names: "John Smith" MIT
- Keep queries 3-6 words (optimal for search APIs)
- Include context terms for disambiguation

Return ONLY valid JSON:
{
  "topic": "Main topic",
  "outline": ["Section1", "Section2", ...],
  "research_tree": {
    "primary": [
      {
        "question": "What is their educational background?",
        "queries": ["\"Name\" university education", "\"Name\" degree school"],
        "target_sources": ["academic", "official"],
        "confidence_weight": 0.3
      }
    ],
    "secondary": [...],
    "tertiary": [...]
  },
  "estimated_confidence": 0.85,
  "complexity": "moderate"
}
"""


def planner_node(state: AgentState) -> Dict[str, Any]:
    """Generate a hierarchical research plan with a research tree."""
    original_query = state.get("original_query") or state["messages"][-1].content
    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []
    selected_entity = state.get("selected_entity") or {}
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")

    if not primary_anchor:
        if selected_entity:
            primary_anchor = selected_entity.get("name", "")
        elif anchor_terms:
            primary_anchor = anchor_terms[0]
            anchor_terms = anchor_terms[1:]

    if query_type == "person" and primary_anchor:
        research_tree = _generate_person_research_tree(primary_anchor, anchor_terms)
        outline = ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]
        queries = _flatten_tree_to_queries(research_tree)

    elif query_type == "concept" and primary_anchor:
        research_tree = _generate_concept_research_tree(primary_anchor, anchor_terms)
        outline = ["Overview", "Definition", "How It Works", "Applications", "Research", "Limitations"]
        queries = _flatten_tree_to_queries(research_tree)

    elif query_type == "technical" and primary_anchor:
        research_tree = _generate_technical_research_tree(primary_anchor, anchor_terms)
        outline = ["Overview", "Architecture", "Usage", "Examples", "Best Practices", "Troubleshooting"]
        queries = _flatten_tree_to_queries(research_tree)

    elif query_type == "comparison":
        entities = [primary_anchor] + anchor_terms[:1] if primary_anchor else anchor_terms[:2]
        if len(entities) >= 2:
            research_tree = _generate_comparison_research_tree(entities[0], entities[1])
            outline = ["Overview", f"{entities[0]}", f"{entities[1]}", "Comparison", "Recommendation"]
            queries = _flatten_tree_to_queries(research_tree)
        else:
            research_tree, outline, queries = _generate_llm_research_tree(
                original_query, primary_anchor, anchor_terms, query_type, selected_entity
            )

    else:
        research_tree, outline, queries = _generate_llm_research_tree(
            original_query, primary_anchor, anchor_terms, query_type, selected_entity
        )

    plan: Plan = {
        "topic": primary_anchor if primary_anchor else f"Research: {original_query[:50]}",
        "outline": outline,
        "queries": queries,
        "research_tree": research_tree,
    }

    if primary_anchor:
        _ensure_primary_anchor_in_queries(plan["queries"], primary_anchor)
        _ensure_primary_anchor_in_tree(plan["research_tree"], primary_anchor)

    total_workers = len(plan["queries"])

    return {
        "plan": plan,
        "total_workers": total_workers,
        "done_workers": 0,
        "done_subagents": 0,
        "raw_sources": [],
        "raw_evidence": [],
        "sources": None,
        "evidence": None,
        "claims": None,
        "citations": None,
        "issues": [],
        "research_iteration": 0,
        "knowledge_gaps": [],
        "research_trajectory": [],
        "dead_ends": [],
        "subagent_findings": [],
    }


def _generate_person_research_tree(name: str, context: List[str]) -> ResearchTree:
    """Generate research tree for person queries."""
    quoted_name = f'"{name}"'
    ctx = context[0] if context else ""

    primary: List[ResearchQuestion] = [
        {
            "question": f"Who is {name}? What is their background?",
            "queries": [
                f'{quoted_name} {ctx}' if ctx else f'{quoted_name} biography',
                f'{quoted_name} profile background',
            ],
            "target_sources": ["official", "news"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"What is {name}'s educational background?",
            "queries": [
                f'{quoted_name} education university',
                f'{quoted_name} degree school college',
            ],
            "target_sources": ["academic", "official"],
            "confidence_weight": 0.2,
        },
        {
            "question": f"What is {name}'s professional work and career?",
            "queries": [
                f'{quoted_name} career work experience',
                f'{quoted_name} projects achievements',
            ],
            "target_sources": ["official", "news"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"What are {name}'s notable achievements or contributions?",
            "queries": [
                f'{quoted_name} achievements awards',
                f'{quoted_name} contributions notable',
            ],
            "target_sources": ["news", "academic"],
            "confidence_weight": 0.2,
        },
    ]

    secondary: List[ResearchQuestion] = [
        {
            "question": f"What is {name}'s online presence?",
            "queries": [
                f'{quoted_name} LinkedIn',
                f'{quoted_name} GitHub Twitter',
            ],
            "target_sources": ["official"],
            "confidence_weight": 0.1,
        },
        {
            "question": f"What has {name} published or presented?",
            "queries": [
                f'{quoted_name} publications papers',
                f'{quoted_name} research talks',
            ],
            "target_sources": ["academic"],
            "confidence_weight": 0.1,
        },
    ]

    tertiary: List[ResearchQuestion] = [
        {
            "question": f"What do others say about {name}?",
            "queries": [
                f'{quoted_name} mentioned recommended',
            ],
            "target_sources": ["community", "news"],
            "confidence_weight": 0.05,
        },
    ]

    return {"primary": primary, "secondary": secondary, "tertiary": tertiary}


def _generate_concept_research_tree(concept: str, context: List[str]) -> ResearchTree:
    """Generate research tree for concept queries."""
    quoted = f'"{concept}"' if ' ' in concept else concept

    primary: List[ResearchQuestion] = [
        {
            "question": f"What is {concept}? Definition and overview.",
            "queries": [
                f'{quoted} definition explained',
                f'{quoted} what is overview',
            ],
            "target_sources": ["academic", "docs"],
            "confidence_weight": 0.3,
        },
        {
            "question": f"How does {concept} work?",
            "queries": [
                f'{quoted} how it works',
                f'{quoted} mechanism process',
            ],
            "target_sources": ["academic", "docs"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"What are the applications of {concept}?",
            "queries": [
                f'{quoted} applications use cases',
                f'{quoted} examples real world',
            ],
            "target_sources": ["news", "academic"],
            "confidence_weight": 0.2,
        },
    ]

    secondary: List[ResearchQuestion] = [
        {
            "question": f"What research exists on {concept}?",
            "queries": [
                f'{quoted} research paper',
                f'{quoted} study findings',
            ],
            "target_sources": ["academic"],
            "confidence_weight": 0.15,
        },
        {
            "question": f"What are the limitations of {concept}?",
            "queries": [
                f'{quoted} limitations problems',
                f'{quoted} challenges criticism',
            ],
            "target_sources": ["academic", "community"],
            "confidence_weight": 0.1,
        },
    ]

    tertiary: List[ResearchQuestion] = [
        {
            "question": f"What is the future of {concept}?",
            "queries": [
                f'{quoted} future trends',
            ],
            "target_sources": ["news"],
            "confidence_weight": 0.05,
        },
    ]

    return {"primary": primary, "secondary": secondary, "tertiary": tertiary}


def _generate_technical_research_tree(tech: str, context: List[str]) -> ResearchTree:
    """Generate research tree for technical queries."""
    quoted = f'"{tech}"' if ' ' in tech else tech

    primary: List[ResearchQuestion] = [
        {
            "question": f"What is {tech}? Official documentation.",
            "queries": [
                f'{quoted} documentation official',
                f'{quoted} guide tutorial',
            ],
            "target_sources": ["docs"],
            "confidence_weight": 0.3,
        },
        {
            "question": f"How does {tech} work architecturally?",
            "queries": [
                f'{quoted} architecture how it works',
                f'{quoted} internals explained',
            ],
            "target_sources": ["docs", "academic"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"How to use {tech}? Examples and patterns.",
            "queries": [
                f'{quoted} examples code',
                f'{quoted} usage patterns',
            ],
            "target_sources": ["docs", "code"],
            "confidence_weight": 0.2,
        },
    ]

    secondary: List[ResearchQuestion] = [
        {
            "question": f"What are best practices for {tech}?",
            "queries": [
                f'{quoted} best practices',
                f'{quoted} tips recommendations',
            ],
            "target_sources": ["docs", "community"],
            "confidence_weight": 0.15,
        },
        {
            "question": f"Common issues and troubleshooting for {tech}?",
            "queries": [
                f'{quoted} common issues problems',
                f'{quoted} troubleshooting errors',
            ],
            "target_sources": ["community", "docs"],
            "confidence_weight": 0.1,
        },
    ]

    tertiary: List[ResearchQuestion] = [
        {
            "question": f"How does {tech} compare to alternatives?",
            "queries": [
                f'{quoted} vs alternatives comparison',
            ],
            "target_sources": ["community"],
            "confidence_weight": 0.05,
        },
    ]

    return {"primary": primary, "secondary": secondary, "tertiary": tertiary}


def _generate_comparison_research_tree(entity1: str, entity2: str) -> ResearchTree:
    """Generate research tree for comparison queries."""

    primary: List[ResearchQuestion] = [
        {
            "question": f"How do {entity1} and {entity2} compare directly?",
            "queries": [
                f'{entity1} vs {entity2}',
                f'{entity1} versus {entity2} comparison',
            ],
            "target_sources": ["general", "community"],
            "confidence_weight": 0.3,
        },
        {
            "question": f"What are the key differences between {entity1} and {entity2}?",
            "queries": [
                f'{entity1} vs {entity2} differences',
                f'{entity1} {entity2} pros cons',
            ],
            "target_sources": ["general", "community"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"What are the strengths of {entity1}?",
            "queries": [
                f'{entity1} advantages benefits',
                f'{entity1} strengths pros',
            ],
            "target_sources": ["general"],
            "confidence_weight": 0.15,
        },
        {
            "question": f"What are the strengths of {entity2}?",
            "queries": [
                f'{entity2} advantages benefits',
                f'{entity2} strengths pros',
            ],
            "target_sources": ["general"],
            "confidence_weight": 0.15,
        },
    ]

    secondary: List[ResearchQuestion] = [
        {
            "question": f"When should you use {entity1} vs {entity2}?",
            "queries": [
                f'when to use {entity1} vs {entity2}',
                f'{entity1} {entity2} use cases',
            ],
            "target_sources": ["community"],
            "confidence_weight": 0.1,
        },
    ]

    tertiary: List[ResearchQuestion] = [
        {
            "question": f"What do experts recommend between {entity1} and {entity2}?",
            "queries": [
                f'{entity1} vs {entity2} recommendation',
            ],
            "target_sources": ["community"],
            "confidence_weight": 0.05,
        },
    ]

    return {"primary": primary, "secondary": secondary, "tertiary": tertiary}


def _generate_llm_research_tree(
    original_query: str,
    primary_anchor: str,
    anchor_terms: List[str],
    query_type: str,
    selected_entity: Dict
) -> tuple:
    """Generate research tree using LLM for complex queries."""
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)

    entity_context = ""
    if selected_entity:
        entity_context = f"""
Entity identified:
- Name: {selected_entity.get('name', 'Unknown')}
- Description: {selected_entity.get('description', '')}
- Identifiers: {selected_entity.get('identifiers', [])}
"""

    resp = llm.invoke([
        SystemMessage(content=PLANNER_V8_SYSTEM),
        HumanMessage(content=f"""Research question: {original_query}

PRIMARY ENTITY: "{primary_anchor}"
CONTEXT TERMS: {anchor_terms}
QUERY TYPE: {query_type}
{entity_context}

Create a comprehensive research tree.""")
    ])

    result = parse_json_object(resp.content, default={})

    research_tree = result.get("research_tree", {"primary": [], "secondary": [], "tertiary": []})
    outline = result.get("outline", ["Overview", "Details", "Analysis", "Conclusion"])

    queries = _flatten_tree_to_queries(research_tree)

    return research_tree, outline, queries


def _flatten_tree_to_queries(tree: ResearchTree) -> List[PlanQuery]:
    """Convert research tree to flat query list."""
    queries = []
    qid = 1

    for priority, questions in [("primary", tree.get("primary", [])),
                                 ("secondary", tree.get("secondary", [])),
                                 ("tertiary", tree.get("tertiary", []))]:
        for q in questions:
            for query_text in q.get("queries", []):
                queries.append({
                    "qid": f"Q{qid}",
                    "query": query_text,
                    "section": q.get("question", "General")[:50],
                    "lane": "general",
                    "priority": q.get("confidence_weight", 0.5),
                })
                qid += 1

    return queries


def _ensure_primary_anchor_in_queries(queries: List[PlanQuery], primary_anchor: str):
    """Ensure every query contains the primary anchor."""
    if not primary_anchor:
        return

    primary_lower = primary_anchor.lower()
    quoted_primary = f'"{primary_anchor}"'

    for q in queries:
        query_text = q.get("query", "")
        query_lower = query_text.lower()

        if primary_lower not in query_lower:
            q["query"] = f'{quoted_primary} {query_text}'


def _ensure_primary_anchor_in_tree(tree: ResearchTree, primary_anchor: str):
    """Ensure all queries in research tree contain the primary anchor."""
    if not primary_anchor:
        return

    primary_lower = primary_anchor.lower()
    quoted_primary = f'"{primary_anchor}"'

    for priority in ["primary", "secondary", "tertiary"]:
        questions = tree.get(priority, [])
        for q in questions:
            updated_queries = []
            for query_text in q.get("queries", []):
                if primary_lower not in query_text.lower():
                    query_text = f'{quoted_primary} {query_text}'
                updated_queries.append(query_text)
            q["queries"] = updated_queries
