# src/nodes/clarify.py
"""
Clarification and Context Update Nodes.

Handles human-in-the-loop clarification at two points:
1. DISCOVERY CLARIFY: After discovery finds multiple entities (pre-search)
2. RANKER CLARIFY: After ranker detects mixed entities (post-search, backup)

Also includes:
- UPDATE_CONTEXT: Injects confirmed entity into targeted queries
- FILTER: Filters sources to selected entity (post-search)
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.core.state import AgentState, PlanQuery
from src.utils.logging_setup import get_logger
from src.utils.validation import (
    get_str, get_list, get_dict,
    ensure_str, ensure_list
)

logger = get_logger(__name__)


def _looks_like_name_or_entity(text: str) -> bool:
    """
    Check if text looks like a person/entity name (possibly with context).

    Examples that should return True:
    - "Pierce Berke" (capitalized name)
    - "pierce berke" (lowercase name - user may not capitalize)
    - "pierce berke amherst college" (name + context)
    - "Dr. John Smith Stanford" (name with title + context)

    Examples that should return False:
    - "what is pierce" (question)
    - "tell me more" (command)
    - "yes" (confirmation)
    """
    text = text.strip()
    if not text:
        return False

    words = text.split()

    # Single word could be a name, but need at least 2 for confidence
    # (single words are handled by cluster matching)
    if len(words) < 2:
        return False

    # Too many words is likely a question or long description
    if len(words) > 6:
        return False

    # Check for question/command patterns
    lower = text.lower()
    question_patterns = [
        "what", "who", "how", "tell", "show", "find", "search",
        "can you", "could you", "please", "i want", "i need"
    ]
    if any(lower.startswith(p) for p in question_patterns):
        return False

    # If text has proper capitalization, it's likely a name
    has_capital = any(w[0].isupper() for w in words if len(w) > 1)
    if has_capital:
        return True

    # For lowercase input, check if it matches common name patterns:
    # - 2-4 words that are not common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "about", "more", "some", "any", "all", "this", "that",
        "yes", "no", "ok", "okay", "sure", "maybe", "first"
    }

    non_stop_words = [w for w in words if w.lower() not in stop_words]

    # If we have 2+ non-stop words, likely a name/entity
    return len(non_stop_words) >= 2


def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    Process human clarification after discovery or ranker detects multiple entities.

    The graph INTERRUPTS before this node to get human input.
    Human response is stored in state["human_clarification"].

    Possible responses:
    - Number (1, 2, 3): Select that cluster
    - "yes" / "y" / "first": Confirm first option
    - Name text (e.g., "Claudiu Babin"): Use as NEW entity to search
    - Free text: Additional context to add to search
    """
    human_response = get_str(state, "human_clarification", "")
    clusters = get_list(state, "entity_clusters", [])
    subject = get_str(state, "subject", "")

    if not human_response:
        logger.info("No response, using first cluster")
        if clusters:
            selected = clusters[0]
            return {
                "selected_cluster": selected,
                "primary_anchor": ensure_str(selected.get("entity_name"), default=subject),
                "needs_clarification": False,
            }
        return {"needs_clarification": False}

    logger.info(f"Received clarification: {human_response}")
    response_stripped = human_response.strip()
    response_lower = response_stripped.lower()

    # === Try to parse as number ===
    try:
        idx = int(response_stripped) - 1
        if 0 <= idx < len(clusters):
            selected = clusters[idx]
            entity_name = ensure_str(selected.get("entity_name"), default=subject)
            logger.info(f"Selected #{idx + 1}: {entity_name}")
            return {
                "selected_cluster": selected,
                "primary_anchor": entity_name,
                "needs_clarification": False,
            }
    except ValueError:
        pass

    # === Check for yes/no/first ===
    if response_lower in ("yes", "y", "1", "first", "the first one", "first one"):
        if clusters:
            selected = clusters[0]
            entity_name = ensure_str(selected.get("entity_name"), default=subject)
            logger.info(f"Confirmed first: {entity_name}")
            return {
                "selected_cluster": selected,
                "primary_anchor": entity_name,
                "needs_clarification": False,
            }

    # === Check if response matches any cluster name ===
    for cluster in clusters:
        entity_name = ensure_str(cluster.get("entity_name"), default="").lower()
        entity_desc = ensure_str(cluster.get("entity_description"), default="").lower()

        # Check for substring match in name or description
        if response_lower in entity_name or entity_name in response_lower:
            logger.info(f"Matched cluster: {cluster.get('entity_name', '?')}")
            return {
                "selected_cluster": cluster,
                "primary_anchor": cluster.get("entity_name", subject),
                "needs_clarification": False,
            }

        # Also check key words in description
        response_words = set(response_lower.split())
        desc_words = set(entity_desc.split())
        if len(response_words & desc_words) >= 2:  # At least 2 words match
            logger.info(f"Matched cluster by description: {cluster.get('entity_name', '?')}")
            return {
                "selected_cluster": cluster,
                "primary_anchor": cluster.get("entity_name", subject),
                "needs_clarification": False,
            }

    # === User provided a NEW name (not in clusters) ===
    # This is the KEY fix: if they type a name like "Claudiu Babin" or "pierce berke amherst college",
    # we should search for THAT person, not default to first cluster
    # IMPORTANT: This triggers a REPLAN because the entity is completely different
    if _looks_like_name_or_entity(response_stripped):
        logger.info(f"User specified NEW entity: {response_stripped}")

        # Create a new "virtual" cluster for this specific person
        new_cluster = {
            "entity_name": response_stripped,
            "entity_description": f"Specific person/entity: {response_stripped}",
            "entity_type": "person",  # Assume person when user specifies a name
            "source_urls": [],
            "evidence": "User-specified entity",
            "confidence": 1.0,
        }

        # Update anchor terms with the full name for better search
        anchor_terms = get_list(state, "anchor_terms", [])
        # Don't add if it's already there
        if response_stripped not in anchor_terms:
            anchor_terms = anchor_terms + [response_stripped]

        # Get original query to preserve user's intent
        original_query = get_str(state, "original_query", "")

        return {
            "selected_cluster": new_cluster,
            "primary_anchor": response_stripped,
            "anchor_terms": anchor_terms,
            "needs_clarification": False,
            # Update subject to the new specific name
            "subject": response_stripped,
            "subject_type": "person",  # User specified a name, treat as person
            # CRITICAL: Flag that we need to replan with the new entity
            # The original query + new entity context = new research direction
            "needs_replan": True,
            "replan_context": {
                "original_query": original_query,
                "clarified_entity": response_stripped,
                "anchor_terms": anchor_terms,
            },
        }

    # === Generic free text - use as additional context ===
    logger.info(f"Using as additional context: {response_stripped[:50]}...")
    anchor_terms = get_list(state, "anchor_terms", [])
    anchor_terms = anchor_terms + [response_stripped]

    # If we have clusters, pick the first one but add the context
    if clusters:
        selected = clusters[0]
        return {
            "selected_cluster": selected,
            "primary_anchor": ensure_str(selected.get("entity_name"), default=subject),
            "anchor_terms": anchor_terms,
            "needs_clarification": False,
        }

    return {
        "anchor_terms": anchor_terms,
        "needs_clarification": False,
    }


def update_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Update targeted queries with confirmed entity context.

    After human selects an entity, this node:
    1. Gets the confirmed entity name and description
    2. Updates all targeted_queries to include this context
    3. Updates the plan for workers
    """
    selected_cluster = get_dict(state, "selected_cluster", {})
    targeted_queries = get_list(state, "targeted_queries", [])
    plan = get_dict(state, "plan", {})
    subject = get_str(state, "subject", "")
    anchor_terms = get_list(state, "anchor_terms", [])

    entity_name = ensure_str(selected_cluster.get("entity_name"), default="")
    entity_desc = ensure_str(selected_cluster.get("entity_description"), default="")

    if not entity_name:
        logger.warning("No entity selected, keeping original queries")
        return {
            "total_workers": len(targeted_queries),
        }

    logger.info(f"Updating queries for: {entity_name}")

    # Extract key context terms from entity description
    context_terms: List[str] = []
    if entity_desc:
        # Take key identifying words (first few meaningful words)
        words = entity_desc.replace(",", " ").replace("-", " ").split()
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "for", "of", "in", "on", "at"}
        context_terms = [w for w in words if len(w) > 2 and w.lower() not in stop_words][:3]

    # Also include anchor terms if they're different from entity name
    for term in anchor_terms:
        term_clean = ensure_str(term).strip()
        if term_clean and term_clean.lower() != entity_name.lower():
            context_terms.append(term_clean)

    context_str = " ".join(context_terms[:4])  # Limit to avoid too long queries

    # Update targeted queries with confirmed entity
    updated_queries: List[PlanQuery] = []
    for q in targeted_queries:
        if not isinstance(q, dict):
            continue

        query_text = ensure_str(q.get("query"), default="")
        if not query_text:
            continue

        # Replace generic subject with specific entity name
        if subject and subject.lower() in query_text.lower():
            # Replace subject with full entity name
            import re
            query_text = re.sub(
                re.escape(subject),
                entity_name,
                query_text,
                flags=re.IGNORECASE
            )
        elif entity_name.lower() not in query_text.lower():
            # Add entity name to query
            query_text = f'"{entity_name}" {context_str} {query_text}'.strip()

        updated_queries.append({
            **q,
            "query": query_text,
        })

    # Ensure at least one query
    if not updated_queries:
        updated_queries.append({
            "qid": "T1",
            "query": f'"{entity_name}" {context_str}'.strip(),
            "section": "Overview",
            "facet": "general",
            "lane": "general",
        })

    # Update plan
    updated_plan = {
        **plan,
        "topic": entity_name,
        "queries": updated_queries,
    }

    logger.info(f"Updated {len(updated_queries)} queries")
    for q in updated_queries[:2]:
        logger.info(f"  - {q['query'][:50]}...")

    return {
        "plan": updated_plan,
        "targeted_queries": updated_queries,
        "primary_anchor": entity_name,
        "total_workers": len(updated_queries),
    }


def filter_node(state: AgentState) -> Dict[str, Any]:
    """
    Filter sources to only those about the selected entity.

    Used after ranker detects mixed entities (post-search backup).
    Keeps only sources from the selected cluster.
    """
    selected_cluster = get_dict(state, "selected_cluster", {})
    sources = get_list(state, "sources", [])
    evidence = get_list(state, "evidence", [])
    plan = get_dict(state, "plan", {})

    if not selected_cluster:
        logger.info("No cluster selected, keeping all sources")
        return {}

    # Get URLs in selected cluster
    cluster_urls = set(ensure_list(selected_cluster.get("source_urls")))
    entity_name = ensure_str(selected_cluster.get("entity_name"), default="")

    if not cluster_urls:
        logger.info("No URLs in cluster, keeping all sources")
        if entity_name:
            return {
                "plan": {**plan, "topic": entity_name},
                "primary_anchor": entity_name,
                "needs_clarification": False,
            }
        return {"needs_clarification": False}

    # Filter sources
    original_count = len(sources)
    filtered_sources = []
    for s in sources:
        if isinstance(s, dict) and s.get("url") in cluster_urls:
            filtered_sources.append(s)

    # If no sources match (URL mismatch), keep all but update topic
    if not filtered_sources:
        logger.warning("URL mismatch, keeping all sources")
        filtered_sources = sources

    # Filter evidence
    valid_sids = {s.get("sid") for s in filtered_sources if isinstance(s, dict)}
    filtered_evidence = [
        e for e in evidence
        if isinstance(e, dict) and e.get("sid") in valid_sids
    ]

    logger.info(f"Kept {len(filtered_sources)}/{original_count} sources for: {entity_name}")

    # Update plan topic
    updated_plan = plan
    if entity_name:
        updated_plan = {**plan, "topic": entity_name}

    return {
        "sources": filtered_sources,
        "evidence": filtered_evidence,
        "plan": updated_plan,
        "primary_anchor": entity_name or get_str(state, "primary_anchor", ""),
        "needs_clarification": False,
    }
