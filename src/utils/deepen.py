# src/utils/deepen.py
"""
Unified Research Deepening.

ONE system that works for ALL query types. No hardcoded paths for
"person queries" vs "concept queries" vs "comparison queries".

Inspired by how industry leaders do it:
- OpenAI Deep Research: Plan → Act → Observe → Reflect → Repeat
- Google Gemini: Research plan → Execute → Reflect on gaps → Follow-up
- Perplexity Pro: Search → Analyze → Decide if deeper needed → Follow-up

The LLM decides what to explore next based on WHAT WAS FOUND,
not based on predefined query type rules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# UNIFIED KNOWLEDGE EXTRACTION
# ============================================================================

EXTRACT_KNOWLEDGE_SYSTEM = """You analyze research findings to extract structured knowledge and identify opportunities for deeper exploration.

Given what was found about a topic, extract:

1. KEY ENTITIES: People, organizations, places, products, concepts discovered
2. KEY FACTS: Important verified information learned
3. CONNECTIONS: Relationships between entities discovered
4. GAPS: What important aspects haven't been covered yet
5. DEEPER OPPORTUNITIES: Specific things mentioned that deserve more exploration

Return ONLY valid JSON:
{
  "entities_discovered": [
    {"name": "entity name", "type": "person|org|place|concept|product|event", "context": "how it relates to the topic"}
  ],
  "key_facts": [
    "Important fact 1",
    "Important fact 2"
  ],
  "connections": [
    {"from": "entity1", "to": "entity2", "relationship": "how they connect"}
  ],
  "gaps_identified": [
    {"aspect": "what's missing", "importance": "high|medium|low"}
  ],
  "deepen_opportunities": [
    {"topic": "specific thing to explore", "why": "why this would add value", "suggested_query": "search query to find this"}
  ]
}

Be specific and actionable. Focus on what would genuinely deepen understanding.
The same approach should work whether researching a person, concept, event, or anything else.
"""


# ============================================================================
# UNIFIED DEEPENING SUGGESTIONS
# ============================================================================

SUGGEST_DEEPER_SYSTEM = """You are a research strategist. Given what was found so far, suggest follow-up searches to deepen the research.

Your suggestions should:
1. Follow up on specific things mentioned (names, projects, papers, events)
2. Fill gaps in coverage
3. Find authoritative/primary sources for claims
4. Explore connections and relationships discovered
5. Get different perspectives if needed

Return ONLY valid JSON:
{
  "follow_up_searches": [
    {
      "query": "specific search query",
      "rationale": "why this search would add value",
      "targets": "what kind of information this should find"
    }
  ]
}

Guidelines:
- Be specific - vague queries get vague results
- Build on what was discovered - use names, terms, and facts found
- Don't repeat searches already done
- Prioritize high-value searches (authoritative sources, specific details)
- 3-5 searches maximum
"""


def extract_knowledge(
    topic: str,
    claims: List[Dict],
    existing_knowledge: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Extract structured knowledge from research findings.

    UNIFIED - works the same for any query type.
    The LLM figures out what's important based on content.

    Args:
        topic: The main research topic
        claims: List of claims/findings from research
        existing_knowledge: Knowledge from previous rounds (to avoid duplicates)

    Returns:
        Structured knowledge dictionary
    """
    if not claims:
        return {
            "entities_discovered": [],
            "key_facts": [],
            "connections": [],
            "gaps_identified": [],
            "deepen_opportunities": [],
        }

    # Format claims
    claims_text = "\n".join([
        f"- {c.get('text', '')}" for c in claims[:20]
    ])

    # What we already know (to avoid re-discovering)
    existing_facts = []
    if existing_knowledge:
        existing_facts = existing_knowledge.get("key_facts", [])

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    prompt = f"""Research Topic: {topic}

Findings so far:
{claims_text}

{f"Previously known facts (avoid duplicates): {existing_facts[:5]}" if existing_facts else ""}

Extract structured knowledge and identify opportunities for deeper research."""

    resp = llm.invoke([
        SystemMessage(content=EXTRACT_KNOWLEDGE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    knowledge = parse_json_object(resp.content, default={
        "entities_discovered": [],
        "key_facts": [],
        "connections": [],
        "gaps_identified": [],
        "deepen_opportunities": [],
    })

    return knowledge


def suggest_deeper_searches(
    topic: str,
    knowledge: Dict,
    existing_queries: List[str],
    max_queries: int = 4,
) -> List[Dict[str, Any]]:
    """
    Suggest follow-up searches based on extracted knowledge.

    UNIFIED - the LLM decides what's valuable based on content,
    not based on query type.

    Args:
        topic: The main research topic
        knowledge: Output from extract_knowledge()
        existing_queries: Queries already executed (to avoid duplicates)
        max_queries: Maximum new queries to suggest

    Returns:
        List of query dicts ready for workers
    """
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)

    # Format knowledge for the prompt
    entities = knowledge.get("entities_discovered", [])
    entities_str = "\n".join([
        f"- {e.get('name', '')} ({e.get('type', '')}): {e.get('context', '')}"
        for e in entities[:10]
    ])

    facts = knowledge.get("key_facts", [])
    facts_str = "\n".join([f"- {f}" for f in facts[:10]])

    gaps = knowledge.get("gaps_identified", [])
    gaps_str = "\n".join([
        f"- {g.get('aspect', '')} (importance: {g.get('importance', 'medium')})"
        for g in gaps[:5]
    ])

    opportunities = knowledge.get("deepen_opportunities", [])
    opps_str = "\n".join([
        f"- {o.get('topic', '')}: {o.get('why', '')}"
        for o in opportunities[:5]
    ])

    existing_str = "\n".join([f"- {q}" for q in existing_queries[:10]])

    prompt = f"""Research Topic: {topic}

=== What We've Discovered ===

Entities found:
{entities_str if entities_str else "None yet"}

Key facts learned:
{facts_str if facts_str else "None yet"}

Gaps in coverage:
{gaps_str if gaps_str else "None identified"}

Opportunities to explore deeper:
{opps_str if opps_str else "None identified"}

=== Searches Already Done ===
{existing_str if existing_str else "None yet"}

Based on this, suggest {max_queries} follow-up searches that would most improve the research.
Build on what was discovered - use specific names, terms, and facts found."""

    resp = llm.invoke([
        SystemMessage(content=SUGGEST_DEEPER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    result = parse_json_object(resp.content, default={"follow_up_searches": []})
    suggestions = result.get("follow_up_searches", [])

    # Format for workers
    queries = []
    existing_lower = set(q.lower() for q in existing_queries)

    for i, s in enumerate(suggestions[:max_queries]):
        query_text = s.get("query", "")
        if not query_text:
            continue

        # Skip if too similar to existing
        if query_text.lower() in existing_lower:
            continue

        queries.append({
            "qid": f"DEEP{i+1}",
            "query": query_text,
            "section": s.get("targets", "Deep Research"),
            "rationale": s.get("rationale", ""),
            "lane": "general",
            "source": "unified_deepening",
        })

    return queries


def merge_knowledge(existing: Optional[Dict], new: Dict) -> Dict:
    """
    Merge knowledge across research rounds.

    Accumulates entities and facts, keeps fresh suggestions.
    """
    if not existing:
        return new

    merged = {
        "entities_discovered": [],
        "key_facts": [],
        "connections": [],
        "gaps_identified": [],
        "deepen_opportunities": [],
    }

    # Merge entities (deduplicate by name)
    seen_entities = set()
    for e in existing.get("entities_discovered", []) + new.get("entities_discovered", []):
        name = e.get("name", "").lower()
        if name and name not in seen_entities:
            seen_entities.add(name)
            merged["entities_discovered"].append(e)

    # Merge facts (deduplicate)
    seen_facts = set()
    for f in existing.get("key_facts", []) + new.get("key_facts", []):
        f_lower = f.lower() if isinstance(f, str) else str(f).lower()
        if f_lower not in seen_facts:
            seen_facts.add(f_lower)
            merged["key_facts"].append(f)

    # Merge connections
    merged["connections"] = existing.get("connections", []) + new.get("connections", [])

    # Use new gaps and opportunities (old ones are stale)
    merged["gaps_identified"] = new.get("gaps_identified", [])
    merged["deepen_opportunities"] = new.get("deepen_opportunities", [])

    return merged


def should_deepen(
    knowledge: Dict,
    current_round: int,
    max_rounds: int,
    min_confidence: float = 0.7,
) -> bool:
    """
    Decide if we should do another round of research.

    Based on gaps and opportunities found, not hardcoded rules.
    """
    if current_round >= max_rounds:
        return False

    # Check if there are high-importance gaps
    gaps = knowledge.get("gaps_identified", [])
    high_gaps = [g for g in gaps if g.get("importance") == "high"]

    # Check if there are good deepening opportunities
    opportunities = knowledge.get("deepen_opportunities", [])

    # Deepen if we have high-importance gaps or good opportunities
    return len(high_gaps) > 0 or len(opportunities) >= 2
