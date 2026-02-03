# src/nodes/analyzer.py
"""
Query Analyzer Node - LLM-Driven Semantic Understanding

This module implements intelligent query analysis inspired by:
- OpenAI Deep Research: End-to-end understanding, no hardcoded patterns
- Anthropic: Extended thinking with clear reasoning chains
- Google Gemini: Structured output with research plan suggestions
- Perplexity: Semantic parsing with intent classification

The analyzer replaces ALL regex pattern matching with LLM-based understanding,
enabling the system to handle ANY query type without hardcoding edge cases.

Key Features:
1. Semantic intent understanding (not keyword matching)
2. Query classification with nuanced categories
3. Ambiguity detection with clarification suggestions
4. Research question generation
5. Temporal and complexity assessment
6. Self-consistency validation for uncertain queries
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from src.core.state import AgentState
from src.utils.json_utils import parse_json_object
from src.utils.llm import create_chat_model


# ============================================================================
# QUERY CLASSIFICATION TAXONOMY
# ============================================================================

class QueryClass(str, Enum):
    """
    Comprehensive query classification taxonomy.

    Unlike simple "person/concept/technical" classification, this taxonomy
    captures the TRUE intent of what the user wants to learn.
    """
    # Person-related
    PERSON_PROFILE = "person_profile"       # Who is X? Biography, background
    PERSON_WORK = "person_work"             # What has X done? Achievements, projects
    PERSON_OPINIONS = "person_opinions"     # What does X think about Y?

    # Topic/Concept-related
    CONCEPT_EXPLANATION = "concept_explanation"  # What is X? How does it work?
    CONCEPT_DEEP_DIVE = "concept_deep_dive"      # Comprehensive understanding of X

    # Current Events & News
    CURRENT_EVENTS = "current_events"       # Recent news, developments, policies
    TRENDING_TOPIC = "trending_topic"       # What's happening with X right now?

    # Analysis & Research
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # X vs Y, pros/cons
    CAUSAL_ANALYSIS = "causal_analysis"           # Why did X happen? Effects?
    POLICY_ANALYSIS = "policy_analysis"           # Government/corporate policies

    # Technical/How-To
    TECHNICAL_HOWTO = "technical_howto"     # How to do X? Implementation
    TECHNICAL_DEBUG = "technical_debug"     # Why isn't X working? Troubleshooting
    TECHNICAL_DOCS = "technical_docs"       # Documentation, API reference

    # Academic/Research
    ACADEMIC_RESEARCH = "academic_research"  # Scientific papers, studies
    LITERATURE_REVIEW = "literature_review"  # State of research on X

    # Other
    FACTUAL_LOOKUP = "factual_lookup"       # Simple fact: When was X? Where is Y?
    RECOMMENDATION = "recommendation"        # What's the best X for Y?
    GENERAL = "general"                      # Catch-all for unclassified


class AmbiguityLevel(str, Enum):
    """Ambiguity assessment levels."""
    NONE = "none"           # Query is clear and specific
    LOW = "low"             # Minor ambiguity, can proceed with assumptions
    MEDIUM = "medium"       # Some ambiguity, clarification would help
    HIGH = "high"           # Significant ambiguity, clarification needed


class TemporalScope(str, Enum):
    """Temporal relevance of the query."""
    RECENT = "recent"       # Last few months, current events
    HISTORICAL = "historical"  # Past events, historical context
    TIMELESS = "timeless"   # Evergreen content, doesn't depend on time
    SPECIFIC_PERIOD = "specific_period"  # Specific time range mentioned


class Complexity(str, Enum):
    """Query complexity for routing decisions."""
    SIMPLE = "simple"       # Can be answered quickly, few sources needed
    MEDIUM = "medium"       # Moderate depth, multiple perspectives
    COMPLEX = "complex"     # Deep research, many sources, synthesis needed


# ============================================================================
# QUERY ANALYSIS SCHEMA
# ============================================================================

@dataclass
class QueryAnalysis:
    """
    Complete analysis of a user query.

    This structured output captures everything needed for downstream nodes
    to generate appropriate research plans and search strategies.
    """
    # Core Understanding
    intent: str                          # What the user actually wants to learn
    query_class: QueryClass              # Classification category

    # Subject Analysis
    primary_subject: str                 # Main entity/topic being researched
    subject_type: str                    # person, organization, concept, event, policy, technology
    topic_focus: Optional[str]           # Specific angle or aspect (if any)

    # Context
    temporal_scope: TemporalScope        # Time relevance
    geographic_scope: Optional[str]      # Geographic relevance (if any)
    domain: Optional[str]                # Field/domain (politics, tech, science, etc.)

    # Complexity Assessment
    complexity: Complexity               # Simple/medium/complex
    estimated_sources_needed: int        # How many sources likely needed
    estimated_search_queries: int        # How many searches likely needed

    # Ambiguity Detection
    ambiguity_level: AmbiguityLevel      # How ambiguous is the query?
    ambiguity_reasons: List[str]         # Why it's ambiguous (if applicable)
    needs_clarification: bool            # Should we ask user for clarification?
    clarification_question: Optional[str] # What to ask (if needed)
    clarification_options: List[str]     # Possible options to present

    # Research Guidance
    suggested_questions: List[str]       # Research questions to investigate
    suggested_outline: List[str]         # Suggested report sections
    suggested_source_types: List[str]    # news, academic, official, community, etc.
    search_strategy: str                 # Brief description of search approach

    # Confidence
    analysis_confidence: float           # How confident is this analysis (0-1)
    reasoning: str                       # Explanation of the analysis

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "intent": self.intent,
            "query_class": self.query_class.value if isinstance(self.query_class, QueryClass) else self.query_class,
            "primary_subject": self.primary_subject,
            "subject_type": self.subject_type,
            "topic_focus": self.topic_focus,
            "temporal_scope": self.temporal_scope.value if isinstance(self.temporal_scope, TemporalScope) else self.temporal_scope,
            "geographic_scope": self.geographic_scope,
            "domain": self.domain,
            "complexity": self.complexity.value if isinstance(self.complexity, Complexity) else self.complexity,
            "estimated_sources_needed": self.estimated_sources_needed,
            "estimated_search_queries": self.estimated_search_queries,
            "ambiguity_level": self.ambiguity_level.value if isinstance(self.ambiguity_level, AmbiguityLevel) else self.ambiguity_level,
            "ambiguity_reasons": self.ambiguity_reasons,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "clarification_options": self.clarification_options,
            "suggested_questions": self.suggested_questions,
            "suggested_outline": self.suggested_outline,
            "suggested_source_types": self.suggested_source_types,
            "search_strategy": self.search_strategy,
            "analysis_confidence": self.analysis_confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysis":
        """Create from dictionary."""
        return cls(
            intent=data.get("intent", ""),
            query_class=QueryClass(data.get("query_class", "general")),
            primary_subject=data.get("primary_subject", ""),
            subject_type=data.get("subject_type", "unknown"),
            topic_focus=data.get("topic_focus"),
            temporal_scope=TemporalScope(data.get("temporal_scope", "timeless")),
            geographic_scope=data.get("geographic_scope"),
            domain=data.get("domain"),
            complexity=Complexity(data.get("complexity", "medium")),
            estimated_sources_needed=data.get("estimated_sources_needed", 10),
            estimated_search_queries=data.get("estimated_search_queries", 6),
            ambiguity_level=AmbiguityLevel(data.get("ambiguity_level", "none")),
            ambiguity_reasons=data.get("ambiguity_reasons", []),
            needs_clarification=data.get("needs_clarification", False),
            clarification_question=data.get("clarification_question"),
            clarification_options=data.get("clarification_options", []),
            suggested_questions=data.get("suggested_questions", []),
            suggested_outline=data.get("suggested_outline", []),
            suggested_source_types=data.get("suggested_source_types", ["general"]),
            search_strategy=data.get("search_strategy", ""),
            analysis_confidence=data.get("analysis_confidence", 0.5),
            reasoning=data.get("reasoning", ""),
        )


# ============================================================================
# ANALYZER SYSTEM PROMPT
# ============================================================================

ANALYZER_SYSTEM_PROMPT = """You are an expert research query analyst. Your task is to deeply understand what a user wants to learn and provide structured guidance for a research agent.

## Your Analysis Process

1. **Understand True Intent**: Look beyond the surface query to understand what the user actually wants to learn. "Research X" doesn't mean "define X" - it means conduct thorough research about X.

2. **Classify Accurately**: Choose the most specific classification that captures the intent:
   - person_profile: Learning about who someone IS (biography, background)
   - person_work: Learning about what someone has DONE (achievements, projects)
   - person_opinions: Learning what someone THINKS about a topic
   - concept_explanation: Understanding what something IS and how it works
   - concept_deep_dive: Comprehensive exploration of a topic
   - current_events: Recent news, policy changes, developments
   - trending_topic: What's happening RIGHT NOW with something
   - comparative_analysis: Comparing multiple things (X vs Y)
   - causal_analysis: Understanding WHY something happened
   - policy_analysis: Government/corporate policies and their implications
   - technical_howto: How to implement/do something
   - technical_debug: Troubleshooting problems
   - technical_docs: Looking for documentation/references
   - academic_research: Scientific papers and studies
   - literature_review: State of research on a topic
   - factual_lookup: Simple factual question
   - recommendation: Best X for Y type questions
   - general: Only if nothing else fits

3. **Detect Ambiguity**: Identify if the query is ambiguous and WHY:
   - Multiple people/things with the same name
   - Unclear which aspect the user cares about
   - Missing context that would change the research direction
   - Time period unclear for time-sensitive topics

4. **Generate Research Questions**: Create specific, answerable research questions that will help achieve the user's intent.

5. **Suggest Structure**: Propose an outline that matches the query type and complexity.

## Critical Rules

- NEVER classify a query about recent policies/events as "concept_explanation"
- If a query mentions a person + topic, it's likely "person_opinions" or "current_events", NOT "person_profile"
- "Research X" or "Deep research about X" means COMPREHENSIVE investigation, not just definition
- Pay attention to temporal indicators: "new", "recent", "2024", "2025", "latest" → current_events
- Pay attention to action words: "policies", "actions", "decisions" → policy_analysis or current_events

## Output Format

Return ONLY valid JSON with this structure:
{
  "intent": "Clear statement of what the user wants to learn",
  "query_class": "one of the classification values",
  "primary_subject": "Main entity/topic being researched",
  "subject_type": "person|organization|concept|event|policy|technology|place|other",
  "topic_focus": "Specific angle or aspect (null if general)",
  "temporal_scope": "recent|historical|timeless|specific_period",
  "geographic_scope": "Geographic relevance or null",
  "domain": "Field/domain: politics, technology, science, business, health, etc.",
  "complexity": "simple|medium|complex",
  "estimated_sources_needed": 5-30,
  "estimated_search_queries": 4-15,
  "ambiguity_level": "none|low|medium|high",
  "ambiguity_reasons": ["reason1", "reason2"] or [],
  "needs_clarification": true/false,
  "clarification_question": "Question to ask user (null if not needed)",
  "clarification_options": ["option1", "option2"] or [],
  "suggested_questions": [
    "Specific research question 1",
    "Specific research question 2",
    "..."
  ],
  "suggested_outline": [
    "Section 1",
    "Section 2",
    "..."
  ],
  "suggested_source_types": ["news", "official", "academic", "analysis", "community"],
  "search_strategy": "Brief description of how to approach searching for this",
  "analysis_confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your analysis"
}

## Examples

### Example 1: Policy/Current Events Query
Query: "Deep research about Trump's new immigrant related policies"

{
  "intent": "Understand the recent immigration policy changes under the Trump administration, including specific policies, their implementation, and their impact",
  "query_class": "current_events",
  "primary_subject": "Trump administration immigration policies",
  "subject_type": "policy",
  "topic_focus": "recent immigration-related policies and changes",
  "temporal_scope": "recent",
  "geographic_scope": "United States",
  "domain": "politics",
  "complexity": "complex",
  "estimated_sources_needed": 20,
  "estimated_search_queries": 10,
  "ambiguity_level": "none",
  "ambiguity_reasons": [],
  "needs_clarification": false,
  "clarification_question": null,
  "clarification_options": [],
  "suggested_questions": [
    "What immigration-related executive orders has Trump signed since taking office?",
    "What are the key policy changes compared to the previous administration?",
    "How are these policies being implemented at the border?",
    "What legal challenges have been filed against these policies?",
    "What do immigration experts and legal scholars say about these policies?",
    "What is the humanitarian impact of these policies?",
    "How have these policies affected different immigrant groups?"
  ],
  "suggested_outline": [
    "Executive Summary",
    "Key Policy Changes",
    "Executive Orders and Legal Framework",
    "Implementation and Enforcement",
    "Legal Challenges",
    "Humanitarian Impact",
    "Expert Analysis and Perspectives",
    "Comparison with Previous Administration"
  ],
  "suggested_source_types": ["news", "official", "legal", "analysis", "advocacy"],
  "search_strategy": "Start with recent news on Trump immigration policies 2024-2025, then search for specific executive orders, legal analysis, and expert commentary. Include both supportive and critical perspectives.",
  "analysis_confidence": 0.95,
  "reasoning": "The query explicitly asks for 'deep research' about 'new' policies, indicating current events research rather than a conceptual explanation. The subject is clearly immigration policies under Trump, which is a policy analysis in the political domain."
}

### Example 2: Ambiguous Person Query
Query: "John Smith professor"

{
  "intent": "Learn about a professor named John Smith",
  "query_class": "person_profile",
  "primary_subject": "John Smith",
  "subject_type": "person",
  "topic_focus": "academic career and research",
  "temporal_scope": "timeless",
  "geographic_scope": null,
  "domain": "academia",
  "complexity": "medium",
  "estimated_sources_needed": 10,
  "estimated_search_queries": 6,
  "ambiguity_level": "high",
  "ambiguity_reasons": [
    "John Smith is an extremely common name",
    "No institution specified",
    "No field/discipline specified",
    "No other identifying information provided"
  ],
  "needs_clarification": true,
  "clarification_question": "There are many professors named John Smith. Could you provide more context? For example: Which university? What field do they teach? Any other identifying information?",
  "clarification_options": [
    "Specify the university (e.g., 'John Smith at MIT')",
    "Specify the field (e.g., 'John Smith economics professor')",
    "Provide other context (e.g., 'John Smith who wrote about X')"
  ],
  "suggested_questions": [],
  "suggested_outline": [],
  "suggested_source_types": ["academic", "university"],
  "search_strategy": "Cannot proceed without disambiguation - too many possible matches",
  "analysis_confidence": 0.3,
  "reasoning": "The query is highly ambiguous because 'John Smith' is one of the most common names in English-speaking countries, and 'professor' provides minimal disambiguation. Without additional context like institution, field, or other identifiers, research would likely return information about the wrong person."
}

### Example 3: Technical Query
Query: "How does React useState work internally"

{
  "intent": "Understand the internal implementation and mechanics of React's useState hook",
  "query_class": "technical_docs",
  "primary_subject": "React useState hook",
  "subject_type": "technology",
  "topic_focus": "internal implementation",
  "temporal_scope": "timeless",
  "geographic_scope": null,
  "domain": "technology",
  "complexity": "medium",
  "estimated_sources_needed": 8,
  "estimated_search_queries": 5,
  "ambiguity_level": "none",
  "ambiguity_reasons": [],
  "needs_clarification": false,
  "clarification_question": null,
  "clarification_options": [],
  "suggested_questions": [
    "How does useState maintain state between re-renders?",
    "What is the fiber architecture and how does it relate to hooks?",
    "How does React track which useState call corresponds to which state?",
    "What happens internally when setState is called?",
    "How does the reconciliation process work with state updates?"
  ],
  "suggested_outline": [
    "Overview of useState API",
    "The Fiber Architecture",
    "How Hooks Are Stored",
    "State Update Mechanism",
    "Re-rendering Process",
    "Common Pitfalls Explained"
  ],
  "suggested_source_types": ["docs", "technical_blog", "github", "community"],
  "search_strategy": "Search for React useState internals, React fiber architecture, and look at React source code explanations. Include official docs and deep-dive technical articles.",
  "analysis_confidence": 0.95,
  "reasoning": "Clear technical query about a specific, well-defined technology. No ambiguity - React's useState is a specific hook with documented behavior. The 'internally' keyword indicates the user wants implementation details, not just usage."
}
"""


# ============================================================================
# ANALYZER NODE IMPLEMENTATION
# ============================================================================

def _extract_query_from_state(state: AgentState) -> str:
    """Extract the user's query from state messages."""
    messages = state.get("messages", [])

    # Look for the most recent human message
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif hasattr(msg, "content"):
            # Check if it's a HumanMessage by class name
            if "HumanMessage" in str(type(msg)):
                return msg.content

    # Fallback to last message
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            return last_msg.content

    return ""


def _run_analysis(query: str, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Run LLM analysis on the query.

    Uses low temperature for consistent, deterministic analysis.
    """
    llm = create_chat_model(model="gpt-4o", temperature=temperature)

    response = llm.invoke([
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this research query:\n\n{query}"),
    ])

    return parse_json_object(response.content, default={})


def _run_self_consistency_analysis(query: str, n_samples: int = 3) -> Dict[str, Any]:
    """
    Run multiple analysis passes for uncertain queries.

    Inspired by Anthropic's approach - when confidence is low,
    run multiple passes and take consensus.
    """
    results = []

    for i in range(n_samples):
        # Vary temperature to get diverse perspectives
        temp = 0.1 if i == 0 else 0.4
        result = _run_analysis(query, temperature=temp)
        if result:
            results.append(result)

    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # Find consensus on key fields
    from collections import Counter

    # Consensus on query_class
    classes = [r.get("query_class", "general") for r in results]
    class_counts = Counter(classes)
    consensus_class = class_counts.most_common(1)[0][0]

    # Consensus on needs_clarification
    clarification_votes = [r.get("needs_clarification", False) for r in results]
    needs_clarification = sum(clarification_votes) > len(results) / 2

    # Average confidence
    confidences = [r.get("analysis_confidence", 0.5) for r in results]
    avg_confidence = sum(confidences) / len(confidences)

    # Use the result with highest confidence as base
    best_result = max(results, key=lambda r: r.get("analysis_confidence", 0))

    # Override with consensus values
    best_result["query_class"] = consensus_class
    best_result["needs_clarification"] = needs_clarification
    best_result["analysis_confidence"] = avg_confidence
    best_result["_self_consistency_samples"] = len(results)

    return best_result


def _validate_and_fix_analysis(analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Validate and fix common issues in LLM analysis.

    This is a safety net, not the primary classification mechanism.
    """
    # Ensure required fields exist
    if not analysis.get("intent"):
        analysis["intent"] = f"Research about: {query[:100]}"

    if not analysis.get("query_class"):
        analysis["query_class"] = "general"

    if not analysis.get("primary_subject"):
        # Try to extract from query
        analysis["primary_subject"] = query[:100]

    # Validate enum values
    valid_classes = [e.value for e in QueryClass]
    if analysis.get("query_class") not in valid_classes:
        analysis["query_class"] = "general"

    valid_temporal = [e.value for e in TemporalScope]
    if analysis.get("temporal_scope") not in valid_temporal:
        analysis["temporal_scope"] = "timeless"

    valid_complexity = [e.value for e in Complexity]
    if analysis.get("complexity") not in valid_complexity:
        analysis["complexity"] = "medium"

    valid_ambiguity = [e.value for e in AmbiguityLevel]
    if analysis.get("ambiguity_level") not in valid_ambiguity:
        analysis["ambiguity_level"] = "none"

    # Ensure lists are lists
    for field in ["ambiguity_reasons", "clarification_options", "suggested_questions",
                  "suggested_outline", "suggested_source_types"]:
        if not isinstance(analysis.get(field), list):
            analysis[field] = []

    # Ensure confidence is valid
    try:
        analysis["analysis_confidence"] = float(analysis.get("analysis_confidence", 0.5))
        analysis["analysis_confidence"] = max(0.0, min(1.0, analysis["analysis_confidence"]))
    except (ValueError, TypeError):
        analysis["analysis_confidence"] = 0.5

    # Ensure numeric fields are valid
    try:
        analysis["estimated_sources_needed"] = int(analysis.get("estimated_sources_needed", 10))
    except (ValueError, TypeError):
        analysis["estimated_sources_needed"] = 10

    try:
        analysis["estimated_search_queries"] = int(analysis.get("estimated_search_queries", 6))
    except (ValueError, TypeError):
        analysis["estimated_search_queries"] = 6

    return analysis


def _should_use_self_consistency(query: str) -> bool:
    """
    Determine if we should use self-consistency (multiple passes).

    Use for queries that might be ambiguous or complex.
    """
    # Short queries are often ambiguous
    if len(query.split()) <= 3:
        return True

    # Queries with common names might need multiple passes
    common_name_indicators = ["john", "michael", "david", "james", "robert",
                             "smith", "johnson", "williams", "brown", "jones"]
    query_lower = query.lower()
    if any(name in query_lower for name in common_name_indicators):
        return True

    return False


def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze user query using LLM-based semantic understanding.

    This node:
    1. Extracts the query from state
    2. Runs LLM analysis (with optional self-consistency)
    3. Validates and structures the result
    4. Determines if clarification is needed

    Returns state updates for:
    - original_query: The raw query string
    - query_analysis: Structured QueryAnalysis object
    - clarification_request: Question for user (if needed)
    - needs_clarification: Boolean flag for routing

    Inspired by:
    - OpenAI: No hardcoded patterns, pure LLM understanding
    - Anthropic: Extended thinking with reasoning
    - Google: Structured output with research guidance
    - Perplexity: Semantic parsing with intent classification
    """
    # Extract query
    original_query = _extract_query_from_state(state)

    if not original_query:
        # No query found - return minimal state
        return {
            "original_query": "",
            "query_analysis": QueryAnalysis(
                intent="No query provided",
                query_class=QueryClass.GENERAL,
                primary_subject="",
                subject_type="unknown",
                topic_focus=None,
                temporal_scope=TemporalScope.TIMELESS,
                geographic_scope=None,
                domain=None,
                complexity=Complexity.SIMPLE,
                estimated_sources_needed=0,
                estimated_search_queries=0,
                ambiguity_level=AmbiguityLevel.HIGH,
                ambiguity_reasons=["No query provided"],
                needs_clarification=True,
                clarification_question="What would you like me to research?",
                clarification_options=[],
                suggested_questions=[],
                suggested_outline=[],
                suggested_source_types=[],
                search_strategy="",
                analysis_confidence=0.0,
                reasoning="No query was provided",
            ).to_dict(),
            "needs_clarification": True,
            "clarification_request": "What would you like me to research?",
        }

    # Run analysis
    use_self_consistency = _should_use_self_consistency(original_query)

    if use_self_consistency:
        raw_analysis = _run_self_consistency_analysis(original_query, n_samples=3)
    else:
        raw_analysis = _run_analysis(original_query)

    # Validate and fix
    analysis = _validate_and_fix_analysis(raw_analysis, original_query)

    # Build clarification request if needed
    clarification_request = None
    needs_clarification = analysis.get("needs_clarification", False)

    if needs_clarification:
        clarification_request = analysis.get("clarification_question")
        if not clarification_request:
            # Generate a default clarification question
            ambiguity_reasons = analysis.get("ambiguity_reasons", [])
            if ambiguity_reasons:
                clarification_request = f"I need some clarification: {ambiguity_reasons[0]}. Could you provide more details?"
            else:
                clarification_request = f"Could you provide more context about '{analysis.get('primary_subject', original_query)}'?"

    # Log analysis for debugging
    print(f"\n[Query Analyzer] Query: {original_query[:100]}...")
    print(f"[Query Analyzer] Class: {analysis.get('query_class')}")
    print(f"[Query Analyzer] Intent: {analysis.get('intent', '')[:100]}...")
    print(f"[Query Analyzer] Confidence: {analysis.get('analysis_confidence', 0):.2f}")
    print(f"[Query Analyzer] Needs Clarification: {needs_clarification}")

    return {
        "original_query": original_query,
        "query_analysis": analysis,
        "needs_clarification": needs_clarification,
        "clarification_request": clarification_request,
        # For backwards compatibility with existing nodes
        "primary_anchor": analysis.get("primary_subject", ""),
        "anchor_terms": [],  # Will be populated by planner
    }


# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def route_after_analysis(state: AgentState) -> str:
    """
    Route based on whether clarification is needed.

    Returns:
        "clarify" - Query is ambiguous, need human input
        "planner" - Query is clear, proceed to planning
    """
    needs_clarification = state.get("needs_clarification", False)

    # Also check analysis confidence
    query_analysis = state.get("query_analysis", {})
    confidence = query_analysis.get("analysis_confidence", 1.0)
    ambiguity_level = query_analysis.get("ambiguity_level", "none")

    # High ambiguity OR explicit needs_clarification OR very low confidence
    if needs_clarification or ambiguity_level == "high" or confidence < 0.3:
        return "clarify"

    return "planner"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "QueryClass",
    "AmbiguityLevel",
    "TemporalScope",
    "Complexity",
    "QueryAnalysis",
    "query_analyzer_node",
    "route_after_analysis",
]
