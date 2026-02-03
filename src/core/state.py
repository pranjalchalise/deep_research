# src/core/state.py
"""
Research Studio v9 State Schema.

Features:
- LLM-driven query analysis (replaces pattern matching)
- Human-in-the-loop clarification with enriched context
- Knowledge gap tracking with iterative refinement
- Research trajectory history
- Dead end tracking for backtracking
- Source credibility scoring (E-E-A-T)
- Verified citations with span matching
- Per-claim confidence scores
- Multi-agent orchestration state
"""
from __future__ import annotations

import operator
from typing import Dict, List, Optional, TypedDict, Literal, Any
from typing_extensions import Annotated

from langgraph.graph import MessagesState


# -------------------------
# Lanes / routing category
# -------------------------
Lane = Literal[
    "general",       # broad web (no domain filter)
    "docs",          # official docs
    "papers",        # academic / arxiv / journals
    "code",          # github / repos
    "news",          # news / announcements
    "forums",        # reddit / discourse / hn
]


# -------------------------
# Query Types (Legacy - kept for backwards compatibility)
# -------------------------
QueryType = Literal[
    "person",
    "concept",
    "technical",
    "event",
    "comparison",
    "organization",
    "general"
]


# -------------------------
# V9 Query Classification (New)
# -------------------------
QueryClass = Literal[
    # Person-related
    "person_profile",       # Who is X? Biography, background
    "person_work",          # What has X done? Achievements, projects
    "person_opinions",      # What does X think about Y?

    # Topic/Concept-related
    "concept_explanation",  # What is X? How does it work?
    "concept_deep_dive",    # Comprehensive understanding of X

    # Current Events & News
    "current_events",       # Recent news, developments, policies
    "trending_topic",       # What's happening with X right now?

    # Analysis & Research
    "comparative_analysis", # X vs Y, pros/cons
    "causal_analysis",      # Why did X happen? Effects?
    "policy_analysis",      # Government/corporate policies

    # Technical/How-To
    "technical_howto",      # How to do X? Implementation
    "technical_debug",      # Why isn't X working?
    "technical_docs",       # Documentation, API reference

    # Academic/Research
    "academic_research",    # Scientific papers, studies
    "literature_review",    # State of research on X

    # Other
    "factual_lookup",       # Simple fact: When was X?
    "recommendation",       # What's the best X for Y?
    "general",              # Catch-all
]


AmbiguityLevel = Literal["none", "low", "medium", "high"]
TemporalScope = Literal["recent", "historical", "timeless", "specific_period"]
ComplexityLevel = Literal["simple", "medium", "complex"]


class QueryAnalysisResult(TypedDict, total=False):
    """
    Structured output from the LLM query analyzer.

    This replaces pattern-based classification with semantic understanding.
    """
    # Core Understanding
    intent: str                          # What the user actually wants to learn
    query_class: QueryClass              # Classification category

    # Subject Analysis
    primary_subject: str                 # Main entity/topic being researched
    subject_type: str                    # person, organization, concept, event, policy, technology
    topic_focus: Optional[str]           # Specific angle or aspect

    # Context
    temporal_scope: TemporalScope        # Time relevance
    geographic_scope: Optional[str]      # Geographic relevance
    domain: Optional[str]                # Field/domain (politics, tech, science, etc.)

    # Complexity Assessment
    complexity: ComplexityLevel          # Simple/medium/complex
    estimated_sources_needed: int        # How many sources likely needed
    estimated_search_queries: int        # How many searches likely needed

    # Ambiguity Detection
    ambiguity_level: AmbiguityLevel      # How ambiguous is the query?
    ambiguity_reasons: List[str]         # Why it's ambiguous
    needs_clarification: bool            # Should we ask user?
    clarification_question: Optional[str] # What to ask
    clarification_options: List[str]     # Options to present

    # Research Guidance
    suggested_questions: List[str]       # Research questions to investigate
    suggested_outline: List[str]         # Suggested report sections
    suggested_source_types: List[str]    # news, academic, official, etc.
    search_strategy: str                 # Brief description of approach

    # Confidence
    analysis_confidence: float           # How confident (0-1)
    reasoning: str                       # Explanation


# ---------- Planning ----------
class PlanQuery(TypedDict, total=False):
    qid: str
    query: str
    section: str
    lane: Lane
    priority: float  # NEW: query priority (0-1)


class ResearchQuestion(TypedDict, total=False):
    """A question to be answered during research."""
    question: str
    queries: List[str]
    target_sources: List[str]  # academic, news, official, etc.
    confidence_weight: float   # How much this contributes to overall confidence


class ResearchTree(TypedDict, total=False):
    """Hierarchical research plan."""
    primary: List[ResearchQuestion]    # Must answer
    secondary: List[ResearchQuestion]  # Should answer
    tertiary: List[ResearchQuestion]   # Nice to have


class Plan(TypedDict, total=False):
    topic: str
    outline: List[str]
    queries: List[PlanQuery]
    research_tree: ResearchTree  # NEW: hierarchical research plan


# ---------- Raw worker outputs (accumulated) ----------
class RawSource(TypedDict, total=False):
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict, total=False):
    url: str
    title: str
    section: str
    text: str


# ---------- Normalized artifacts ----------
class Source(TypedDict, total=False):
    sid: str  # S1
    url: str
    title: str
    snippet: str
    score: float
    credibility: float  # NEW: credibility score


class Evidence(TypedDict, total=False):
    eid: str  # E1
    sid: str  # S1
    url: str
    title: str
    section: str
    text: str


class Claim(TypedDict, total=False):
    cid: str  # C1
    section: str
    text: str
    confidence: float  # NEW: claim confidence


class ClaimCitations(TypedDict, total=False):
    cid: str
    eids: List[str]


class Issue(TypedDict, total=False):
    level: str  # "warn" | "block"
    message: str
    related_cids: List[str]


# ---------- Entity / Disambiguation ----------
class EntityCandidate(TypedDict, total=False):
    """A possible entity match found during discovery."""
    name: str
    description: str
    identifiers: List[str]
    confidence: float


class DiscoveryResult(TypedDict, total=False):
    """Results from the discovery phase."""
    query_type: QueryType
    entity_candidates: List[EntityCandidate]
    confidence: float
    needs_clarification: bool
    anchor_terms: List[str]


# ---------- NEW: Iterative Research State ----------
class KnowledgeGap(TypedDict, total=False):
    """A gap in research coverage that needs to be filled."""
    section: str
    description: str
    suggested_queries: List[str]
    priority: float  # 0-1
    current_confidence: float


class TrajectoryStep(TypedDict, total=False):
    """A single step in the research trajectory."""
    iteration: int
    action: str  # "search", "read", "refine", "backtrack"
    query: str
    result_summary: str
    confidence_delta: float
    timestamp: str


class DeadEnd(TypedDict, total=False):
    """A research path that failed."""
    query: str
    reason: str  # "no_results", "irrelevant", "paywall", "low_credibility"
    iteration: int
    alternative_tried: bool


# ---------- NEW: Source Credibility ----------
class SourceCredibility(TypedDict, total=False):
    """Credibility assessment for a source."""
    sid: str
    url: str
    domain_trust: float      # 0-1 based on domain
    freshness: float         # 0-1 based on publish date
    authority: float         # 0-1 based on citations, author
    content_quality: float   # 0-1 based on depth, structure
    overall: float           # weighted average


# ---------- NEW: Citation Verification ----------
class VerifiedCitation(TypedDict, total=False):
    """A citation that has been verified against source text."""
    cid: str
    eid: str
    claim_text: str
    evidence_span: str       # Exact text from source
    match_score: float       # 0-1 semantic similarity
    verified: bool
    cross_validated: bool    # Supported by multiple sources
    supporting_sids: List[str]


# ---------- NEW: Multi-Agent State ----------
class SubagentAssignment(TypedDict, total=False):
    """Assignment for a subagent."""
    subagent_id: str
    question: str
    queries: List[str]
    target_sources: List[str]


class SubagentFindings(TypedDict, total=False):
    """Results from a subagent."""
    subagent_id: str
    question: str
    findings: str  # Compressed findings
    evidence_ids: List[str]
    confidence: float
    iterations_used: int
    dead_ends: List[DeadEnd]


class OrchestratorState(TypedDict, total=False):
    """State of the orchestrator agent."""
    phase: str  # "primary_research", "secondary_research", "synthesis"
    questions_assigned: int
    questions_completed: int
    overall_confidence: float


# ---------- Research Metadata ----------
class ResearchMetadata(TypedDict, total=False):
    """Metadata about the research process."""
    overall_confidence: float
    verified_claims: int
    total_claims: int
    knowledge_gaps: int
    sources_used: int
    research_iterations: int
    total_searches: int
    time_elapsed_seconds: float


# ---------- Agent State ----------
class AgentState(MessagesState):
    """
    Complete state for the v8 research agent.

    MessagesState already defines:
    messages: Annotated[list[BaseMessage], add_messages]
    """

    # === Config knobs ===
    depth: Optional[int]
    max_results: Optional[int]
    round: Optional[int]

    # === V9: Query Analysis (LLM-Driven) ===
    original_query: Optional[str]
    query_analysis: Optional[QueryAnalysisResult]  # NEW: Structured LLM analysis

    # Clarification (HITL)
    needs_clarification: Optional[bool]     # NEW: Boolean flag for routing
    clarification_request: Optional[str]    # Question to ask user
    user_clarification: Optional[str]       # User's response (renamed from human_clarification)
    enriched_context: Optional[str]         # NEW: Combined context after clarification

    # === Legacy Discovery & Disambiguation (kept for compatibility) ===
    discovery: Optional[DiscoveryResult]
    selected_entity: Optional[EntityCandidate]

    # Anchor term hierarchy (populated from query_analysis)
    primary_anchor: Optional[str]           # Main subject (in EVERY query)
    anchor_terms: Optional[List[str]]       # Context qualifiers

    # Legacy clarification field (use user_clarification instead)
    human_clarification: Optional[str]

    # === Planning ===
    plan: Optional[Plan]

    # === Iterative Research State ===
    research_iteration: Optional[int]                    # Current iteration (0, 1, 2...)
    max_iterations: Optional[int]                        # NEW: Hard limit (default: 3)
    knowledge_gaps: Optional[List[KnowledgeGap]]         # What's still missing
    research_trajectory: Optional[List[TrajectoryStep]]  # History of actions
    dead_ends: Optional[List[DeadEnd]]                   # Paths that failed
    proceed_to_synthesis: Optional[bool]                 # Ready for final synthesis
    refinement_queries: Optional[List[Dict[str, Any]]]   # Queries for next iteration

    # === NEW: Multi-Agent State ===
    subagent_assignments: Optional[List[SubagentAssignment]]
    subagent_findings: Annotated[List[SubagentFindings], operator.add]  # Accumulated
    orchestrator_state: Optional[OrchestratorState]
    done_subagents: Annotated[int, operator.add]  # Counter

    # === Worker accumulation ===
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]

    raw_sources: Annotated[List[RawSource], operator.add]
    raw_evidence: Annotated[List[RawEvidence], operator.add]

    # === Normalized artifacts ===
    sources: Optional[List[Source]]
    evidence: Optional[List[Evidence]]

    # === NEW: Source Credibility ===
    source_credibility: Optional[Dict[str, SourceCredibility]]

    # === Trust engine ===
    claims: Optional[List[Claim]]
    citations: Optional[List[ClaimCitations]]
    issues: Optional[List[Issue]]

    # === NEW: Citation Verification ===
    verified_citations: Optional[List[VerifiedCitation]]
    unverified_claims: Optional[List[str]]  # CIDs without verification
    hallucination_score: Optional[float]    # % of unsupported claims

    # === Confidence Scoring ===
    claim_confidence: Optional[Dict[str, float]]  # CID -> confidence
    section_confidence: Optional[Dict[str, float]]  # Section -> confidence
    overall_confidence: Optional[float]
    previous_confidence: Optional[float]  # NEW: For delta calculation in gap detection

    # === NEW: Cross-Validation ===
    cross_validated_claims: Optional[List[VerifiedCitation]]
    single_source_claims: Optional[List[VerifiedCitation]]

    # === Controller / Assess ===
    needs_more: Optional[bool]
    stop: Optional[bool]

    # === Output ===
    report: Optional[str]
    research_metadata: Optional[ResearchMetadata]
