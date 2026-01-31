# src/nodes/discovery.py
"""
Discovery phase nodes for entity disambiguation.

This module implements human-in-the-loop disambiguation with:
1. ERA-CoT (Entity Relationship Analysis Chain-of-Thought) prompting
2. Pattern-based preprocessing for common query structures
3. Self-consistency with multiple reasoning paths
4. Multi-signal fusion for robust analysis

Pipeline:
1. Analyze the query to determine type (person, concept, etc.)
2. Do initial discovery search to find entity candidates
3. Assess confidence - is this clearly one entity or ambiguous?
4. If ambiguous, prepare clarification request for human input
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import (
    AgentState,
    DiscoveryResult,
    EntityCandidate,
    QueryType,
)
from src.tools.tavily import cached_search
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model


# ============================================================================
# PATTERN-BASED PREPROCESSING
# ============================================================================

# Common query patterns with their likely types and entity extraction
QUERY_PATTERNS: List[Tuple[str, str, List[str]]] = [
    # "X at Y" → person at organization
    (r"(?i)^(?:research|find|who is|tell me about)?\s*(.+?)\s+at\s+(.+)$", "person", ["person", "organization"]),

    # "X from Y" → person from organization/place
    (r"(?i)^(?:research|find|who is)?\s*(.+?)\s+from\s+(.+)$", "person", ["person", "origin"]),

    # "Tell me about X" / "Tell me all about X" → concept (general knowledge request)
    (r"(?i)^tell\s+me\s+(?:all\s+)?about\s+(.+?)[\?\s]*$", "concept", ["subject"]),

    # "Research X" / "Find out about X" → concept
    (r"(?i)^(?:research|find\s+out\s+about|learn\s+about|explain)\s+(.+?)[\?\s]*$", "concept", ["subject"]),

    # "What is X" → concept/technical
    (r"(?i)^what\s+(?:is|are)\s+(.+?)[\?\s]*$", "concept", ["subject"]),

    # "How does X work" → technical
    (r"(?i)^how\s+(?:does|do)\s+(.+?)\s+work[\?\s]*$", "technical", ["subject"]),

    # "How to X" → technical/concept
    (r"(?i)^how\s+to\s+(.+?)[\?\s]*$", "technical", ["action"]),

    # "X vs Y" or "X versus Y" → comparison
    (r"(?i)^(.+?)\s+(?:vs\.?|versus|compared to|or)\s+(.+?)[\?\s]*$", "comparison", ["entity1", "entity2"]),

    # "Why does X" → concept
    (r"(?i)^why\s+(?:does|do|is|are)\s+(.+?)[\?\s]*$", "concept", ["subject"]),

    # "Professor/Dr./CEO X" → person with title
    (r"(?i)^((?:professor|prof\.?|dr\.?|ceo|cto|founder|director)\s+.+)$", "person", ["titled_person"]),

    # "X paper/research/study" → academic
    (r"(?i)^(.+?)\s+(?:paper|research|study|thesis|dissertation)$", "technical", ["topic"]),

    # "X company/startup/org" → organization
    (r"(?i)^(.+?)\s+(?:company|startup|organization|org|inc|corp|llc)$", "organization", ["org_name"]),
]


def pattern_preprocess(query: str) -> Optional[Dict[str, Any]]:
    """
    Try to match query against known patterns for quick classification.

    Returns:
        Dict with pattern match info, or None if no pattern matched
    """
    query = query.strip()

    for pattern, query_type, group_names in QUERY_PATTERNS:
        match = re.match(pattern, query)
        if match:
            groups = match.groups()
            extracted = {}
            for i, name in enumerate(group_names):
                if i < len(groups):
                    extracted[name] = groups[i].strip()

            # Build anchor terms from extracted entities
            anchor_terms = [v for v in extracted.values() if v and len(v) > 1]

            return {
                "pattern_matched": True,
                "query_type": query_type,
                "extracted_entities": extracted,
                "anchor_terms": anchor_terms,
                "pattern_confidence": 0.75,  # Pattern matches are fairly confident
            }

    return None


# ============================================================================
# ERA-CoT ANALYZER PROMPT
# ============================================================================

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


# ============================================================================
# ANALYZER NODE - Classify the query
# ============================================================================

def _extract_query_from_state(state: AgentState) -> str:
    """Extract the original query from state messages."""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif hasattr(msg, "content") and "HumanMessage" in str(type(msg)):
            return msg.content
    return state["messages"][-1].content if state["messages"] else ""


def _run_llm_analysis(query: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Run a single LLM analysis pass."""
    llm = create_chat_model(model="gpt-4o-mini", temperature=temperature)

    resp = llm.invoke([
        SystemMessage(content=ANALYZER_SYSTEM_V2),
        HumanMessage(content=f"Research query: {query}"),
    ])

    return parse_json_object(resp.content, default={})


def _self_consistency_analysis(query: str, n_samples: int = 3) -> Dict[str, Any]:
    """
    Run multiple analysis passes and merge results using self-consistency.

    This improves accuracy for ambiguous queries by:
    1. Running N analysis passes with temperature > 0
    2. Taking majority vote on query_type
    3. Merging anchor terms (union)
    4. Averaging confidence scores
    """
    results = []

    for i in range(n_samples):
        # Use temperature 0.1 for first, higher for others to get diversity
        temp = 0.1 if i == 0 else 0.5
        result = _run_llm_analysis(query, temperature=temp)
        if result:
            results.append(result)

    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # Majority vote on query_type
    types = [r.get("query_type", "general") for r in results]
    type_counts = Counter(types)
    final_type = type_counts.most_common(1)[0][0]

    # Merge anchor strategies (union of all)
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

    # Average confidence
    confidences = [r.get("confidence", 0.5) for r in results]
    avg_confidence = sum(confidences) / len(confidences)

    # Calculate consistency score
    consistency = type_counts.most_common(1)[0][1] / len(results)

    # Merge entities (deduplicate by text)
    seen_entities = set()
    merged_entities = []
    for r in results:
        for e in r.get("entities", []):
            text = e.get("text", "").lower()
            if text and text not in seen_entities:
                seen_entities.add(text)
                merged_entities.append(e)

    # Clarification needed if any result said so OR if consistency is low
    clarification_needed = (
        any(r.get("clarification_needed", False) for r in results) or
        consistency < 0.7 or
        avg_confidence < 0.6
    )

    # Use best reasoning (from highest confidence result)
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
    Analyze the query using multi-signal fusion:
    1. Pattern-based quick analysis
    2. ERA-CoT LLM analysis
    3. Optional self-consistency for ambiguous queries
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

    # ---- Step 1: Pattern-based preprocessing ----
    pattern_result = pattern_preprocess(original_query)

    # ---- Step 2: LLM Analysis ----
    # If pattern matched with high confidence, do single LLM pass
    # Otherwise, use self-consistency for better accuracy
    use_self_consistency = state.get("use_self_consistency", False)

    if pattern_result and pattern_result["pattern_confidence"] >= 0.7:
        # Pattern matched - single LLM pass for validation
        llm_analysis = _run_llm_analysis(original_query, temperature=0.1)
    elif use_self_consistency:
        # No pattern match and self-consistency enabled
        llm_analysis = _self_consistency_analysis(original_query, n_samples=3)
    else:
        # Standard single-pass analysis
        llm_analysis = _run_llm_analysis(original_query, temperature=0.1)

    # ---- Step 3: Multi-signal fusion ----
    # Combine pattern + LLM results

    # Determine query type (prefer LLM, validate with pattern if available)
    query_type: QueryType = llm_analysis.get("query_type", "general")
    if query_type not in ("person", "concept", "technical", "event", "organization", "comparison", "general"):
        query_type = "general"

    # Override with pattern if it detected comparison
    if pattern_result and pattern_result.get("query_type") == "comparison":
        query_type = "comparison"

    # If pattern matched same type, boost confidence
    confidence_boost = 0.0
    if pattern_result:
        pattern_type = pattern_result["query_type"]
        # Map comparison to general for matching (pattern uses different type)
        if pattern_type == query_type or (pattern_type == "comparison" and query_type in ["concept", "technical"]):
            confidence_boost = 0.1

    # Build anchor terms from both sources
    anchor_strategy = llm_analysis.get("anchor_strategy", {})
    core_anchors = anchor_strategy.get("core", [])
    disambiguation_anchors = anchor_strategy.get("disambiguation", [])

    # Merge with pattern-extracted anchors
    anchor_terms = list(core_anchors)
    if pattern_result:
        for term in pattern_result.get("anchor_terms", []):
            if term and term not in anchor_terms:
                anchor_terms.append(term)

    # Add disambiguation anchors (but limit total)
    for term in disambiguation_anchors[:2]:
        if term and term not in anchor_terms:
            anchor_terms.append(term)

    # Fallback if still empty
    if not anchor_terms:
        entities = llm_analysis.get("entities", [])
        anchor_terms = [e.get("text") for e in entities if e.get("text") and e.get("ambiguity") != "high"]
        if not anchor_terms:
            # Last resort: use the query itself
            words = original_query.split()
            anchor_terms = [w for w in words if len(w) > 3][:3]

    # Calculate final confidence
    base_confidence = llm_analysis.get("confidence", 0.5)
    final_confidence = min(1.0, base_confidence + confidence_boost)

    # Determine if clarification is needed
    needs_clarification = llm_analysis.get("clarification_needed", False)

    # Check for high-ambiguity entities
    entities = llm_analysis.get("entities", [])
    high_ambiguity_count = sum(1 for e in entities if e.get("ambiguity") == "high")
    if high_ambiguity_count > 0 and query_type == "person":
        # Person queries with ambiguous names should ask for clarification
        needs_clarification = True
        final_confidence = min(final_confidence, 0.6)

    # Boost confidence for concept/technical queries - these are usually unambiguous
    # "osteoporosis", "machine learning", "React useState" are well-defined terms
    if query_type in ("concept", "technical"):
        final_confidence = max(final_confidence, 0.85)
        needs_clarification = False

    return {
        "original_query": original_query,
        "discovery": {
            "query_type": query_type,
            "entity_candidates": [],
            "confidence": 0.0,  # Will be updated by discovery_node
            "needs_clarification": needs_clarification,
            "anchor_terms": anchor_terms,
            # Store full analysis for debugging/logging
            "_analysis": {
                "pattern_result": pattern_result,
                "llm_analysis": llm_analysis,
                "entities": entities,
                "relationships": llm_analysis.get("relationships", []),
            },
        },
        "anchor_terms": anchor_terms,
        # Store suggested clarification for later use
        "_suggested_clarification": llm_analysis.get("suggested_clarification", ""),
    }


# ============================================================================
# DISCOVERY NODE - Find entity candidates
# ============================================================================

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
    """
    Perform discovery search and identify entity candidates.
    """
    original_query = state.get("original_query") or state["messages"][-1].content
    discovery = state.get("discovery") or {}
    query_type = discovery.get("query_type", "general")
    anchor_terms = state.get("anchor_terms") or []
    suggested_clarification = state.get("_suggested_clarification", "")

    # Config
    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir")

    # Build search query - use anchor terms if available
    if anchor_terms:
        # For person queries, search with full context
        if query_type == "person":
            search_query = " ".join(anchor_terms[:3])
        else:
            search_query = original_query
    else:
        search_query = original_query

    # Do a broad discovery search
    results = cached_search(
        query=search_query,
        max_results=10,  # Get more results for better disambiguation
        lane="general",
        use_cache=use_cache,
        cache_dir=f"{cache_dir}/search" if cache_dir else None,
    )

    if not results:
        # No results - proceed with low confidence
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

    # Format results for LLM analysis
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

    # For concept/technical queries, boost confidence - these are less ambiguous than person queries
    # Concepts like "osteoporosis", "machine learning", "React hooks" are well-defined
    if query_type in ("concept", "technical"):
        # Boost confidence for concept queries (usually unambiguous)
        confidence = max(confidence, 0.8)
        needs_clarification = False  # Concepts rarely need clarification

    # Parse entity candidates
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

    # Build clarification request if needed
    clarification_request = None
    if needs_clarification:
        # Use LLM's question if provided, otherwise build one
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

    # Auto-select if high confidence and single clear entity
    selected_entity = None
    primary_anchor = None

    # For concept/technical queries, use the anchor terms as primary
    if query_type in ("concept", "technical"):
        # Use the first anchor term as the primary concept
        if anchor_terms:
            primary_anchor = anchor_terms[0]
        else:
            # Extract from original query
            # Remove common prefixes like "tell me about", "what is", etc.
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
        # Very high confidence on top candidate
        top = entity_candidates[0]
        if top.get("confidence", 0) >= 0.85:
            selected_entity = top
            primary_anchor = selected_entity.get("name", "")
            needs_clarification = False
            clarification_request = None

    # Context anchors are the identifiers (affiliations, roles, etc.)
    # These help with disambiguation but aren't the main subject
    context_anchors = list(anchor_terms)
    if selected_entity:
        for identifier in selected_entity.get("identifiers", []):
            if identifier and identifier not in context_anchors:
                # Don't add the primary entity name as a context anchor
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
        "primary_anchor": primary_anchor,  # The main entity name
        "anchor_terms": context_anchors,   # Context qualifiers
        "clarification_request": clarification_request,
    }


# ============================================================================
# CLARIFY NODE - Process human clarification
# ============================================================================

def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    Process human clarification and update entity selection.

    This node runs AFTER the human provides clarification.
    The graph should interrupt BEFORE this node to get human input.
    """
    discovery = state.get("discovery") or {}
    entity_candidates = discovery.get("entity_candidates", [])
    human_response = state.get("human_clarification", "")
    anchor_terms = state.get("anchor_terms") or []

    if not human_response:
        # No clarification provided - proceed with best guess
        if entity_candidates:
            selected = entity_candidates[0]
            return {
                "selected_entity": selected,
                "anchor_terms": anchor_terms + selected.get("identifiers", []),
                "discovery": {**discovery, "needs_clarification": False},
            }
        return {"discovery": {**discovery, "needs_clarification": False}}

    # Try to parse as number (selecting from candidates)
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

    # Check for yes/no confirmation
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
        # User rejected all candidates - need more context
        # Extract terms from their response if they provided any
        pass

    # User provided additional context - extract PRIMARY entity vs CONTEXT qualifiers
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

    # Extract primary entity (the main subject)
    primary_entity = clarification_analysis.get("primary_entity", "")
    if not primary_entity or primary_entity == "Unknown":
        # Try to extract from the first part of the response (usually the name)
        parts = human_response.split()
        if query_type == "person" and len(parts) >= 2:
            # Assume first two words are the name
            primary_entity = " ".join(parts[:2]).title()
        elif parts:
            primary_entity = parts[0]

    # Extract context qualifiers (help disambiguate but aren't the main subject)
    context_qualifiers = clarification_analysis.get("context_qualifiers", [])
    identifiers = clarification_analysis.get("identifiers", [])

    # Merge context qualifiers and identifiers, remove duplicates
    all_context = list(context_qualifiers)
    for ident in identifiers:
        if ident and ident not in all_context and ident.lower() != primary_entity.lower():
            all_context.append(ident)

    # If we couldn't extract terms, parse the response manually
    if not all_context:
        words = human_response.split()
        # Skip the primary entity words
        primary_words = set(primary_entity.lower().split())
        all_context = [w for w in words if w.lower() not in primary_words and len(w) > 2][:4]

    # Build the selected entity
    selected_entity: EntityCandidate = {
        "name": primary_entity or "Unknown",
        "description": human_response[:200],
        "identifiers": all_context,
        "confidence": 0.9,  # High confidence since user confirmed
    }

    return {
        "selected_entity": selected_entity,
        "primary_anchor": primary_entity,  # NEW: Primary anchor for all queries
        "anchor_terms": all_context,       # Context qualifiers
        "discovery": {**discovery, "needs_clarification": False},
        "human_clarification": human_response,
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_discovery(state: AgentState) -> str:
    """
    Route based on discovery confidence.

    Returns:
        "clarify" - Need human input (will trigger interrupt)
        "planner" - High confidence, proceed to planning
    """
    discovery = state.get("discovery") or {}
    confidence = discovery.get("confidence", 0.0)
    needs_clarification = discovery.get("needs_clarification", False)
    query_type = discovery.get("query_type", "general")

    # Concept/technical queries rarely need clarification - skip to planner
    if query_type in ("concept", "technical", "comparison"):
        return "planner"

    if needs_clarification or confidence < 0.7:
        return "clarify"
    return "planner"
