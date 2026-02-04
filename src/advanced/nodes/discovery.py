"""
Disambiguates the user's query before research begins. Figures out whether the query
refers to a specific person, concept, or technology, and asks for clarification when
the target is ambiguous (e.g., a common name with no context).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from langchain_core.messages import SystemMessage, HumanMessage

from src.advanced.state import (
    AgentState,
    DiscoveryResult,
    EntityCandidate,
    QueryType,
)
from src.tools.tavily import cached_search
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model


# Regex patterns for fast query classification before hitting the LLM.
# Each tuple: (pattern, query_type, group_names for captured groups).
QUERY_PATTERNS: List[Tuple[str, str, List[str]]] = [
    (r"(?i)^(?:research|find|who is|tell me about)?\s*(.+?)\s+at\s+(.+)$", "person", ["person", "organization"]),
    (r"(?i)^(?:research|find|who is)?\s*(.+?)\s+from\s+(.+)$", "person", ["person", "origin"]),
    (r"(?i)^tell\s+me\s+(?:all\s+)?about\s+(.+?)[\?\s]*$", "concept", ["subject"]),
    (r"(?i)^(?:research|find\s+out\s+about|learn\s+about|explain)\s+(.+?)[\?\s]*$", "concept", ["subject"]),
    (r"(?i)^what\s+(?:is|are)\s+(.+?)[\?\s]*$", "concept", ["subject"]),
    (r"(?i)^how\s+(?:does|do)\s+(.+?)\s+work[\?\s]*$", "technical", ["subject"]),
    (r"(?i)^how\s+to\s+(.+?)[\?\s]*$", "technical", ["action"]),
    (r"(?i)^(.+?)\s+(?:vs\.?|versus|compared to|or)\s+(.+?)[\?\s]*$", "comparison", ["entity1", "entity2"]),
    (r"(?i)^why\s+(?:does|do|is|are)\s+(.+?)[\?\s]*$", "concept", ["subject"]),
    (r"(?i)^((?:professor|prof\.?|dr\.?|ceo|cto|founder|director)\s+.+)$", "person", ["titled_person"]),
    (r"(?i)^(.+?)\s+(?:paper|research|study|thesis|dissertation)$", "technical", ["topic"]),
    (r"(?i)^(.+?)\s+(?:company|startup|organization|org|inc|corp|llc)$", "organization", ["org_name"]),
]


def pattern_preprocess(query: str) -> Optional[Dict[str, Any]]:
    """Fast regex-based classification. Returns match info or None."""
    query = query.strip()

    for pattern, query_type, group_names in QUERY_PATTERNS:
        match = re.match(pattern, query)
        if match:
            groups = match.groups()
            extracted = {}
            for i, name in enumerate(group_names):
                if i < len(groups):
                    extracted[name] = groups[i].strip()

            anchor_terms = [v for v in extracted.values() if v and len(v) > 1]

            return {
                "pattern_matched": True,
                "query_type": query_type,
                "extracted_entities": extracted,
                "anchor_terms": anchor_terms,
                "pattern_confidence": 0.75,
            }

    return None


ANALYZER_SYSTEM_V2 = """You are an expert query analyzer using structured reasoning (ERA-CoT).

Analyze the research query step by step:

**Step 1 - Parse Query Structure:**
Identify the query pattern and extract explicit entities.
- What type of query is this? (asking about a person, concept, technology, event, etc.)
- What entities are mentioned? (names, organizations, technologies, etc.)
- What relationships are implied? ("at" implies affiliation, "by" implies authorship, etc.)

**Step 2 - Entity Classification:**
For each entity found:
- What type is it? (person, organization, concept, technology, product, event)
- How ambiguous is it? (low = unique/specific, medium = few possibilities, high = many possibilities)
- Why might it be ambiguous? (common name, multiple meanings, incomplete info)

**Step 3 - Relationship Mapping:**
What relationships connect the entities?
- affiliation: person AT organization
- creation: person CREATED thing
- part_of: concept PART_OF field
- temporal: event IN time_period
- location: entity IN place

**Step 4 - Anchor Term Strategy:**
Design search anchors to find the RIGHT entity:
- core: Most specific, unique identifying terms (always include these)
- disambiguation: Terms that distinguish from similar entities
- expansion: Related terms for broader coverage (optional)

**Step 5 - Confidence Assessment:**
- How certain is the interpretation? (0.0 to 1.0)
- What could cause confusion?
- Would clarification help?

Return ONLY valid JSON:
{
  "query_type": "person|concept|technical|event|organization|comparison|general",
  "entities": [
    {
      "text": "extracted text",
      "type": "person|organization|concept|technology|event|place",
      "ambiguity": "low|medium|high",
      "ambiguity_reason": "why this might be ambiguous"
    }
  ],
  "relationships": [
    {"from": "entity1", "to": "entity2", "type": "affiliation|creation|part_of|temporal|location"}
  ],
  "anchor_strategy": {
    "core": ["most specific identifying terms"],
    "disambiguation": ["terms that help distinguish"],
    "expansion": ["related broader terms"]
  },
  "confidence": 0.0-1.0,
  "clarification_needed": true/false,
  "suggested_clarification": "question to ask if clarification needed",
  "reasoning": "brief explanation of your analysis"
}

Examples:

Query: "Research Marius at Amherst College"
→ Person query, "Marius" is HIGH ambiguity (common name), "Amherst College" provides disambiguation
→ anchor_strategy.core: ["Marius", "Amherst College"]
→ confidence: 0.6 (need more context - which Marius?)

Query: "How does React useState work"
→ Technical query, "React" and "useState" are both LOW ambiguity (specific tech terms)
→ anchor_strategy.core: ["React", "useState"]
→ confidence: 0.95 (very specific)

Query: "John Smith professor"
→ Person query, "John Smith" is HIGH ambiguity (extremely common name)
→ confidence: 0.2 (need institution, field, or other identifier)
"""


def _extract_query_from_state(state: AgentState) -> str:
    """Walk backward through messages to find the most recent human message."""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif hasattr(msg, "content") and "HumanMessage" in str(type(msg)):
            return msg.content
    return state["messages"][-1].content if state["messages"] else ""


def _run_llm_analysis(query: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Single-pass ERA-CoT query analysis via the LLM."""
    llm = create_chat_model(model="gpt-4o-mini", temperature=temperature)

    resp = llm.invoke([
        SystemMessage(content=ANALYZER_SYSTEM_V2),
        HumanMessage(content=f"Research query: {query}"),
    ])

    return parse_json_object(resp.content, default={})


def _self_consistency_analysis(query: str, n_samples: int = 3) -> Dict[str, Any]:
    """
    Run N analysis passes with varied temperature, then merge via majority vote
    on query_type, union of anchor terms, and averaged confidence. Helps with
    ambiguous queries where a single pass might misclassify.
    """
    results = []

    for i in range(n_samples):
        # First pass is near-deterministic; later passes use higher temp for diversity
        temp = 0.1 if i == 0 else 0.5
        result = _run_llm_analysis(query, temperature=temp)
        if result:
            results.append(result)

    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    types = [r.get("query_type", "general") for r in results]
    type_counts = Counter(types)
    final_type = type_counts.most_common(1)[0][0]

    merged_anchors = {
        "core": [],
        "disambiguation": [],
        "expansion": [],
    }
    for r in results:
        strategy = r.get("anchor_strategy", {})
        for key in merged_anchors:
            terms = strategy.get(key, [])
            for t in terms:
                if t and t not in merged_anchors[key]:
                    merged_anchors[key].append(t)

    confidences = [r.get("confidence", 0.5) for r in results]
    avg_confidence = sum(confidences) / len(confidences)

    consistency = type_counts.most_common(1)[0][1] / len(results)

    seen_entities = set()
    merged_entities = []
    for r in results:
        for e in r.get("entities", []):
            text = e.get("text", "").lower()
            if text and text not in seen_entities:
                seen_entities.add(text)
                merged_entities.append(e)

    # Trigger clarification if results disagree or confidence is low
    clarification_needed = (
        any(r.get("clarification_needed", False) for r in results) or
        consistency < 0.7 or
        avg_confidence < 0.6
    )

    best_result = max(results, key=lambda r: r.get("confidence", 0))

    return {
        "query_type": final_type,
        "entities": merged_entities,
        "relationships": best_result.get("relationships", []),
        "anchor_strategy": merged_anchors,
        "confidence": avg_confidence,
        "clarification_needed": clarification_needed,
        "suggested_clarification": best_result.get("suggested_clarification", ""),
        "reasoning": best_result.get("reasoning", ""),
        "_consistency_score": consistency,
        "_n_samples": len(results),
    }


def analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify the query by fusing regex pattern matches with LLM analysis.
    Optionally uses self-consistency (multiple LLM passes) for ambiguous queries.
    """
    original_query = _extract_query_from_state(state)

    if not original_query:
        return {
            "original_query": "",
            "discovery": {
                "query_type": "general",
                "entity_candidates": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "anchor_terms": [],
            },
            "anchor_terms": [],
        }

    pattern_result = pattern_preprocess(original_query)

    # High-confidence pattern match only needs a single LLM pass for validation.
    # No pattern match + self-consistency flag triggers multi-pass for accuracy.
    use_self_consistency = state.get("use_self_consistency", False)

    if pattern_result and pattern_result["pattern_confidence"] >= 0.7:
        llm_analysis = _run_llm_analysis(original_query, temperature=0.1)
    elif use_self_consistency:
        llm_analysis = _self_consistency_analysis(original_query, n_samples=3)
    else:
        llm_analysis = _run_llm_analysis(original_query, temperature=0.1)

    # Fuse pattern and LLM results - LLM is primary, pattern validates
    query_type: QueryType = llm_analysis.get("query_type", "general")
    if query_type not in ("person", "concept", "technical", "event", "organization", "comparison", "general"):
        query_type = "general"

    # Pattern-detected comparisons override LLM since the regex is very reliable here
    if pattern_result and pattern_result.get("query_type") == "comparison":
        query_type = "comparison"

    # Agreement between pattern and LLM increases our confidence
    confidence_boost = 0.0
    if pattern_result:
        pattern_type = pattern_result["query_type"]
        if pattern_type == query_type or (pattern_type == "comparison" and query_type in ["concept", "technical"]):
            confidence_boost = 0.1

    anchor_strategy = llm_analysis.get("anchor_strategy", {})
    core_anchors = anchor_strategy.get("core", [])
    disambiguation_anchors = anchor_strategy.get("disambiguation", [])

    anchor_terms = list(core_anchors)
    if pattern_result:
        for term in pattern_result.get("anchor_terms", []):
            if term and term not in anchor_terms:
                anchor_terms.append(term)

    for term in disambiguation_anchors[:2]:
        if term and term not in anchor_terms:
            anchor_terms.append(term)

    if not anchor_terms:
        entities = llm_analysis.get("entities", [])
        anchor_terms = [e.get("text") for e in entities if e.get("text") and e.get("ambiguity") != "high"]
        if not anchor_terms:
            # Last resort: grab long-ish words from the raw query
            words = original_query.split()
            anchor_terms = [w for w in words if len(w) > 3][:3]

    base_confidence = llm_analysis.get("confidence", 0.5)
    final_confidence = min(1.0, base_confidence + confidence_boost)

    needs_clarification = llm_analysis.get("clarification_needed", False)

    # Ambiguous person names (e.g., "John Smith") should always prompt clarification
    entities = llm_analysis.get("entities", [])
    high_ambiguity_count = sum(1 for e in entities if e.get("ambiguity") == "high")
    if high_ambiguity_count > 0 and query_type == "person":
        needs_clarification = True
        final_confidence = min(final_confidence, 0.6)

    # Concept/technical queries like "osteoporosis" or "React useState" are
    # well-defined terms that almost never need clarification
    if query_type in ("concept", "technical"):
        final_confidence = max(final_confidence, 0.85)
        needs_clarification = False

    return {
        "original_query": original_query,
        "discovery": {
            "query_type": query_type,
            "entity_candidates": [],
            "confidence": 0.0,
            "needs_clarification": needs_clarification,
            "anchor_terms": anchor_terms,
            "_analysis": {
                "pattern_result": pattern_result,
                "llm_analysis": llm_analysis,
                "entities": entities,
                "relationships": llm_analysis.get("relationships", []),
            },
        },
        "anchor_terms": anchor_terms,
        "_suggested_clarification": llm_analysis.get("suggested_clarification", ""),
    }


DISCOVERY_SYSTEM = """You analyze search results to identify distinct entities and assess confidence.

Given search results for a query, determine:
1. Are these results about ONE clear entity, or MULTIPLE different entities?
2. What are the distinct entity candidates found?
3. How confident are you that we've identified the correct entity?

For PERSON queries, cluster results by:
- Institution/affiliation (same university, company, etc.)
- Role/title (professor, engineer, researcher, etc.)
- Field/domain (AI, biology, economics, etc.)
- Time period (current vs historical)
- Unique identifiers (personal website, LinkedIn, publications)

Return ONLY valid JSON:
{
  "confidence": 0.0-1.0,
  "entity_candidates": [
    {
      "name": "Full name or title",
      "description": "Brief description with key identifiers (role, affiliation, field)",
      "identifiers": ["unique identifier 1", "unique identifier 2"],
      "confidence": 0.0-1.0,
      "evidence_urls": ["urls that mention this entity"]
    }
  ],
  "reasoning": "Why you assessed confidence this way",
  "needs_clarification": true/false,
  "clarification_question": "Specific question to disambiguate (if needed)"
}

Confidence guidelines:
- 0.9+: Clearly ONE entity, consistent info across ALL sources
- 0.7-0.9: Likely one entity, minor inconsistencies or gaps
- 0.5-0.7: Uncertain, 2-3 possible entities
- 0.3-0.5: Multiple distinct entities found
- <0.3: Very ambiguous, many possibilities or no clear matches

IMPORTANT: When multiple people share a name, create SEPARATE candidates for each.
Look for distinguishing features: different institutions, different fields, different time periods.
"""


def discovery_node(state: AgentState) -> Dict[str, Any]:
    """Search for the entity and cluster results into distinct candidates."""
    original_query = state.get("original_query") or state["messages"][-1].content
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")
    anchor_terms = state.get("anchor_terms") or []
    suggested_clarification = state.get("_suggested_clarification", "")

    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir")

    # Person queries benefit from searching with combined anchor terms (name + affiliation)
    if anchor_terms:
        if query_type == "person":
            search_query = " ".join(anchor_terms[:3])
        else:
            search_query = original_query
    else:
        search_query = original_query

    results = cached_search(
        query=search_query,
        max_results=10,
        lane="general",
        use_cache=use_cache,
        cache_dir=f"{cache_dir}/search" if cache_dir else None,
    )

    if not results:
        clarification = suggested_clarification or f"I couldn't find much information about '{original_query}'. Could you provide more details or context?"
        return {
            "discovery": {
                **discovery,
                "confidence": 0.3,
                "entity_candidates": [],
                "needs_clarification": True,
            },
            "clarification_request": clarification,
        }

    results_text = "\n\n".join([
        f"[{i+1}] {r['title']}\nURL: {r['url']}\nSnippet: {r.get('snippet', '')[:500]}"
        for i, r in enumerate(results[:10])
    ])

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    resp = llm.invoke([
        SystemMessage(content=DISCOVERY_SYSTEM),
        HumanMessage(content=f"""Query type: {query_type}
Original query: {original_query}
Anchor terms: {anchor_terms}

Search results:
{results_text}

Analyze these results and identify entity candidates. Be thorough in distinguishing between different entities that might share similar names."""),
    ])

    analysis = parse_json_object(resp.content, default={})

    confidence = float(analysis.get("confidence", 0.5))
    needs_clarification = analysis.get("needs_clarification", confidence < 0.7)

    # Concepts like "osteoporosis" or "React hooks" are well-defined and rarely ambiguous
    if query_type in ("concept", "technical"):
        confidence = max(confidence, 0.8)
        needs_clarification = False

    candidates_raw = analysis.get("entity_candidates", [])
    entity_candidates: List[EntityCandidate] = []
    for c in candidates_raw:
        if isinstance(c, dict):
            entity_candidates.append({
                "name": c.get("name", "Unknown"),
                "description": c.get("description", ""),
                "identifiers": c.get("identifiers", []),
                "confidence": float(c.get("confidence", 0.5)),
            })

    clarification_request = None
    if needs_clarification:
        clarification_request = analysis.get("clarification_question", "")

        if not clarification_request:
            if len(entity_candidates) > 1:
                options = "\n".join([
                    f"  {i+1}. {c['name']} - {c['description'][:100]}"
                    for i, c in enumerate(entity_candidates[:5])
                ])
                clarification_request = f"I found multiple possible matches:\n{options}\n\nWhich one are you looking for? (Enter number or provide more details)"
            elif len(entity_candidates) == 1:
                c = entity_candidates[0]
                clarification_request = f"I found: {c['name']} - {c['description']}\n\nIs this correct? (yes/no, or provide more details)"
            else:
                clarification_request = suggested_clarification or f"I'm not sure what you're looking for. Could you provide more details about '{original_query}'?"

    selected_entity = None
    primary_anchor = None

    # For concepts, the anchor term IS the topic; for persons, we auto-select
    # only when confidence is high enough to avoid misidentification
    if query_type in ("concept", "technical"):
        if anchor_terms:
            primary_anchor = anchor_terms[0]
        else:
            clean_query = re.sub(r"(?i)^(tell\s+me\s+(?:all\s+)?about|what\s+(?:is|are)|explain|research)\s+", "", original_query).strip()
            primary_anchor = clean_query
        needs_clarification = False
        clarification_request = None
    elif confidence >= 0.8 and len(entity_candidates) == 1:
        selected_entity = entity_candidates[0]
        primary_anchor = selected_entity.get("name", "")
        needs_clarification = False
        clarification_request = None
    elif confidence >= 0.85 and len(entity_candidates) > 1:
        top = entity_candidates[0]
        if top.get("confidence", 0) >= 0.85:
            selected_entity = top
            primary_anchor = selected_entity.get("name", "")
            needs_clarification = False
            clarification_request = None

    # Context anchors (affiliations, roles) help disambiguate but aren't the main subject
    context_anchors = list(anchor_terms)
    if selected_entity:
        for identifier in selected_entity.get("identifiers", []):
            if identifier and identifier not in context_anchors:
                if primary_anchor and identifier.lower() != primary_anchor.lower():
                    context_anchors.append(identifier)

    return {
        "discovery": {
            "query_type": query_type,
            "entity_candidates": entity_candidates,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "anchor_terms": context_anchors,
        },
        "selected_entity": selected_entity,
        "primary_anchor": primary_anchor,
        "anchor_terms": context_anchors,
        "clarification_request": clarification_request,
    }


def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    Process the human's clarification response (number selection, yes/no,
    or free-text context). The graph interrupts before this node to collect input.
    """
    discovery = state.get("discovery") or {}
    entity_candidates = discovery.get("entity_candidates", [])
    human_response = state.get("human_clarification", "")
    anchor_terms = state.get("anchor_terms") or []

    if not human_response:
        if entity_candidates:
            selected = entity_candidates[0]
            return {
                "selected_entity": selected,
                "anchor_terms": anchor_terms + selected.get("identifiers", []),
                "discovery": {**discovery, "needs_clarification": False},
            }
        return {"discovery": {**discovery, "needs_clarification": False}}

    try:
        idx = int(human_response.strip()) - 1
        if 0 <= idx < len(entity_candidates):
            selected = entity_candidates[idx]
            return {
                "selected_entity": selected,
                "anchor_terms": anchor_terms + selected.get("identifiers", []),
                "discovery": {**discovery, "needs_clarification": False},
            }
    except ValueError:
        pass

    response_lower = human_response.strip().lower()
    if response_lower in ("yes", "y", "correct", "that's right", "yep", "yeah"):
        if entity_candidates:
            selected = entity_candidates[0]
            return {
                "selected_entity": selected,
                "anchor_terms": anchor_terms + selected.get("identifiers", []),
                "discovery": {**discovery, "needs_clarification": False},
            }

    if response_lower in ("no", "n", "nope", "wrong", "none of these"):
        pass

    # Free-text response: use LLM to separate the primary entity from context qualifiers
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    query_type = discovery.get("query_type", "general")

    resp = llm.invoke([
        SystemMessage(content="""Extract the PRIMARY entity and CONTEXT qualifiers from the user's clarification.

IMPORTANT: Distinguish between:
- PRIMARY ENTITY: The main subject being researched (person's full name, concept name, etc.)
  This should appear in EVERY search query.
- CONTEXT QUALIFIERS: Additional terms that help identify/disambiguate (organization, role, field, etc.)
  These provide context but shouldn't replace the primary entity in searches.

Return ONLY a JSON object:
{
  "primary_entity": "The main subject's name or title (e.g., 'Pranjal Chalise', 'React hooks')",
  "entity_type": "person|organization|concept|technology|other",
  "context_qualifiers": ["qualifier1", "qualifier2"],
  "identifiers": ["unique identifier 1", "unique identifier 2"]
}

Examples:
- "pranjal chalise amherst college" →
  {"primary_entity": "Pranjal Chalise", "entity_type": "person", "context_qualifiers": ["Amherst College"], "identifiers": ["Amherst College", "student"]}

- "the AI safety researcher at Apollo Research" →
  {"primary_entity": "Unknown", "entity_type": "person", "context_qualifiers": ["Apollo Research", "AI safety"], "identifiers": ["AI safety researcher", "Apollo Research"]}

- "john smith mit professor machine learning" →
  {"primary_entity": "John Smith", "entity_type": "person", "context_qualifiers": ["MIT", "professor", "machine learning"], "identifiers": ["MIT", "professor", "machine learning"]}
"""),
        HumanMessage(content=f"Query type: {query_type}\nUser clarification: {human_response}"),
    ])

    clarification_analysis = parse_json_object(resp.content, default={})

    primary_entity = clarification_analysis.get("primary_entity", "")
    if not primary_entity or primary_entity == "Unknown":
        parts = human_response.split()
        if query_type == "person" and len(parts) >= 2:
            primary_entity = " ".join(parts[:2]).title()
        elif parts:
            primary_entity = parts[0]

    context_qualifiers = clarification_analysis.get("context_qualifiers", [])
    identifiers = clarification_analysis.get("identifiers", [])

    all_context = list(context_qualifiers)
    for ident in identifiers:
        if ident and ident not in all_context and ident.lower() != primary_entity.lower():
            all_context.append(ident)

    if not all_context:
        words = human_response.split()
        primary_words = set(primary_entity.lower().split())
        all_context = [w for w in words if w.lower() not in primary_words and len(w) > 2][:4]

    # High confidence since the user explicitly provided this info
    selected_entity: EntityCandidate = {
        "name": primary_entity or "Unknown",
        "description": human_response[:200],
        "identifiers": all_context,
        "confidence": 0.9,
    }

    return {
        "selected_entity": selected_entity,
        "primary_anchor": primary_entity,
        "anchor_terms": all_context,
        "discovery": {**discovery, "needs_clarification": False},
        "human_clarification": human_response,
    }


def route_after_discovery(state: AgentState) -> str:
    """Route to 'clarify' if ambiguous, otherwise straight to 'planner'."""
    discovery = state.get("discovery") or {}
    confidence = discovery.get("confidence", 0.0)
    needs_clarification = discovery.get("needs_clarification", False)
    query_type = discovery.get("query_type", "general")

    if query_type in ("concept", "technical", "comparison"):
        return "planner"

    if needs_clarification or confidence < 0.7:
        return "clarify"
    return "planner"
