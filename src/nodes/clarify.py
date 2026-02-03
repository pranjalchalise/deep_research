# src/nodes/clarify.py
"""
V9 Clarification Node - Human-in-the-Loop for Ambiguous Queries

This module handles human clarification when the query analyzer detects
ambiguity that requires user input to resolve.

Key Principle: Once clarified, ALL downstream nodes have access to
the full context (original query + clarification + enriched understanding).

Flow:
1. Query Analyzer detects ambiguity → sets needs_clarification=True
2. Graph INTERRUPTS before clarify_node
3. User provides clarification (stored in user_clarification)
4. This node processes the clarification with LLM
5. Creates enriched_context for downstream nodes
6. Proceeds to Planner with complete information
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, QueryAnalysisResult
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model


# ============================================================================
# CLARIFICATION PROCESSING PROMPT
# ============================================================================

CLARIFICATION_PROCESSOR_PROMPT = """You are processing a user's clarification response to improve a research query.

## Original Query
{original_query}

## Original Analysis
- Intent: {original_intent}
- Query Class: {original_class}
- Primary Subject: {original_subject}
- Topic Focus: {topic_focus}
- Ambiguity Reasons: {ambiguity_reasons}

## Clarification Asked
{clarification_question}

## User's Response
{user_response}

## Your Task

Analyze the user's clarification and produce an updated, enriched understanding.

Output JSON:
{{
  "enriched_context": "A clear, comprehensive statement combining the original query with the clarification. This should be a complete description of what the user wants to research.",
  "updated_intent": "Refined statement of what the user wants to learn",
  "updated_subject": "The clarified primary subject",
  "updated_topic_focus": "Specific angle or aspect (null if general)",
  "updated_class": "The query class (may change based on clarification)",
  "additional_context": ["key context item 1", "key context item 2"],
  "suggested_questions": [
    "Research question 1 based on clarified understanding",
    "Research question 2",
    "Research question 3",
    "Research question 4",
    "Research question 5"
  ],
  "suggested_outline": ["Section 1", "Section 2", "Section 3", "Section 4"],
  "confidence_boost": 0.1-0.4,
  "reasoning": "Brief explanation of how the clarification helped"
}}

## Examples

### Example 1: Person Disambiguation
Original: "John Smith professor"
Clarification asked: "Which John Smith? Please specify university or field."
User response: "MIT, computer science"

Output:
{{
  "enriched_context": "Research John Smith, a computer science professor at MIT. Find information about his background, research work, publications, and academic contributions in computer science.",
  "updated_intent": "Learn about John Smith, the computer science professor at MIT",
  "updated_subject": "John Smith (MIT Computer Science)",
  "updated_topic_focus": "academic career and research",
  "updated_class": "person_profile",
  "additional_context": ["MIT", "computer science", "professor"],
  "suggested_questions": [
    "What is John Smith's educational background and career path?",
    "What are his main research areas in computer science?",
    "What notable publications has he authored?",
    "What courses does he teach at MIT?",
    "What awards or recognition has he received?"
  ],
  "suggested_outline": ["Background", "Research Areas", "Publications", "Teaching", "Awards"],
  "confidence_boost": 0.35,
  "reasoning": "User provided specific institution (MIT) and field (computer science), which uniquely identifies the professor."
}}

### Example 2: Topic Focus Clarification
Original: "Tell me about Apple"
Clarification asked: "Are you asking about Apple the company, or apples the fruit?"
User response: "the company, specifically their AI strategy"

Output:
{{
  "enriched_context": "Research Apple Inc.'s artificial intelligence strategy, including their AI products, research initiatives, acquisitions, and competitive positioning in the AI market.",
  "updated_intent": "Understand Apple Inc.'s AI strategy and initiatives",
  "updated_subject": "Apple Inc.",
  "updated_topic_focus": "AI strategy and initiatives",
  "updated_class": "current_events",
  "additional_context": ["Apple Inc.", "artificial intelligence", "AI strategy", "technology"],
  "suggested_questions": [
    "What AI features has Apple integrated into their products?",
    "What is Apple's AI research division working on?",
    "What AI-related acquisitions has Apple made?",
    "How does Apple's AI strategy compare to competitors?",
    "What are Apple's plans for generative AI?"
  ],
  "suggested_outline": ["AI Product Features", "Research Initiatives", "Acquisitions", "Competitive Analysis", "Future Plans"],
  "confidence_boost": 0.3,
  "reasoning": "User clarified they want Apple the company (not fruit) and specifically their AI strategy, providing clear focus."
}}
"""


# ============================================================================
# CLARIFICATION NODE
# ============================================================================

def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    Process user clarification and enrich context for downstream nodes.

    This node runs AFTER the LangGraph interrupt. The user's response
    is expected in state["user_clarification"] or state["human_clarification"].

    The node:
    1. Takes the user's clarification response
    2. Uses LLM to create enriched context
    3. Updates query_analysis with clarified information
    4. Generates research questions if not already present

    All downstream nodes will have access to:
    - original_query: The raw query
    - user_clarification: What the user said
    - enriched_context: LLM-processed comprehensive context
    - query_analysis: Updated with clarified information

    Returns:
        Dict with enriched_context, updated query_analysis, and cleared flags
    """
    original_query = state.get("original_query", "")
    query_analysis = state.get("query_analysis") or {}
    clarification_request = state.get("clarification_request", "")

    # Check both field names for user's response
    user_clarification = state.get("user_clarification", "")
    if not user_clarification:
        user_clarification = state.get("human_clarification", "")

    # If no clarification provided, try to proceed with best effort
    if not user_clarification:
        print("[Clarify] No clarification provided, proceeding with original query")
        return {
            "enriched_context": original_query,
            "needs_clarification": False,
            "query_analysis": {
                **query_analysis,
                "needs_clarification": False,
                "ambiguity_level": "low",
            }
        }

    # Process the clarification with LLM
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    prompt = CLARIFICATION_PROCESSOR_PROMPT.format(
        original_query=original_query,
        original_intent=query_analysis.get("intent", "Unknown"),
        original_class=query_analysis.get("query_class", "general"),
        original_subject=query_analysis.get("primary_subject", "Unknown"),
        topic_focus=query_analysis.get("topic_focus", "None"),
        ambiguity_reasons=", ".join(query_analysis.get("ambiguity_reasons", ["Unspecified"])),
        clarification_question=clarification_request or "Please provide more details",
        user_response=user_clarification,
    )

    response = llm.invoke([
        SystemMessage(content="You process clarifications to improve research queries. Return only valid JSON."),
        HumanMessage(content=prompt),
    ])

    result = parse_json_object(response.content, default={})

    # Extract processed clarification
    enriched_context = result.get("enriched_context", f"{original_query} - Context: {user_clarification}")
    updated_intent = result.get("updated_intent", query_analysis.get("intent", ""))
    updated_subject = result.get("updated_subject", query_analysis.get("primary_subject", ""))
    updated_topic_focus = result.get("updated_topic_focus")
    updated_class = result.get("updated_class", query_analysis.get("query_class", "general"))
    additional_context = result.get("additional_context", [])
    suggested_questions = result.get("suggested_questions", [])
    suggested_outline = result.get("suggested_outline", [])
    confidence_boost = float(result.get("confidence_boost", 0.25))

    # Update query analysis with clarified information
    updated_analysis: Dict[str, Any] = {
        **query_analysis,
        "intent": updated_intent,
        "primary_subject": updated_subject,
        "topic_focus": updated_topic_focus,
        "query_class": updated_class,
        "needs_clarification": False,
        "ambiguity_level": "none",
        "analysis_confidence": min(1.0, query_analysis.get("analysis_confidence", 0.5) + confidence_boost),
    }

    # Add suggested questions if generated
    if suggested_questions:
        updated_analysis["suggested_questions"] = suggested_questions

    if suggested_outline:
        updated_analysis["suggested_outline"] = suggested_outline

    # Log for debugging
    print(f"\n[Clarify] User clarification: {user_clarification}")
    print(f"[Clarify] Enriched context: {enriched_context[:150]}...")
    print(f"[Clarify] Updated subject: {updated_subject}")
    print(f"[Clarify] Updated class: {updated_class}")
    print(f"[Clarify] Confidence: {query_analysis.get('analysis_confidence', 0.5):.2f} → {updated_analysis['analysis_confidence']:.2f}")

    return {
        # Core clarification outputs
        "enriched_context": enriched_context,
        "user_clarification": user_clarification,
        "needs_clarification": False,
        "clarification_request": None,

        # Updated analysis
        "query_analysis": updated_analysis,

        # Anchor terms for downstream compatibility
        "primary_anchor": updated_subject,
        "anchor_terms": additional_context,

        # Legacy field support
        "human_clarification": user_clarification,
    }


# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def route_after_clarify(state: AgentState) -> str:
    """
    Route after clarification - always proceed to planner.

    The clarification has been processed, so we continue with enriched context.
    """
    return "planner"


# ============================================================================
# HELPER: Format clarification for display
# ============================================================================

def format_clarification_request(state: AgentState) -> str:
    """
    Format the clarification request for user-friendly display.

    Called by UI layer to present clarification to the user.
    """
    query_analysis = state.get("query_analysis", {})
    clarification_question = state.get("clarification_request", "")
    clarification_options = query_analysis.get("clarification_options", [])

    if not clarification_question:
        return "Could you provide more details about your query?"

    formatted = f"**Clarification needed:**\n\n{clarification_question}"

    if clarification_options:
        formatted += "\n\n**Suggestions:**"
        for i, option in enumerate(clarification_options, 1):
            formatted += f"\n{i}. {option}"

    return formatted


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "clarify_node",
    "route_after_clarify",
    "format_clarification_request",
]
