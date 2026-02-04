"""
State schema for the research pipeline. Every node reads from and writes to
these TypedDicts, so changing a field here ripples through the whole system.
AgentState at the bottom is the top-level container that LangGraph operates on.
"""
from __future__ import annotations

import operator
from typing import Dict, List, Optional, TypedDict, Literal, Any
from typing_extensions import Annotated

from langgraph.graph import MessagesState


# Which search vertical to route a query to
Lane = Literal[
    "general",       # broad web, no domain filter
    "docs",          # official documentation
    "papers",        # academic (arxiv, journals)
    "code",          # github, repos
    "news",          # news, announcements
    "forums",        # reddit, discourse, HN
]


# Kept around for backwards compat with older code paths
QueryType = Literal[
    "person",
    "concept",
    "technical",
    "event",
    "comparison",
    "organization",
    "general"
]


# Fine-grained classification that drives search strategy, outline shape,
# and source selection downstream
QueryClass = Literal[
    "person_profile",       "person_work",          "person_opinions",
    "concept_explanation",  "concept_deep_dive",
    "current_events",       "trending_topic",
    "comparative_analysis", "causal_analysis",      "policy_analysis",
    "technical_howto",      "technical_debug",      "technical_docs",
    "academic_research",    "literature_review",
    "factual_lookup",       "recommendation",       "general",
]


AmbiguityLevel = Literal["none", "low", "medium", "high"]
TemporalScope = Literal["recent", "historical", "timeless", "specific_period"]
ComplexityLevel = Literal["simple", "medium", "complex"]


class QueryAnalysisResult(TypedDict, total=False):
    """LLM-produced understanding of the user's query -- replaces the old
    regex/keyword classifier with a single structured call."""

    intent: str
    query_class: QueryClass

    primary_subject: str
    subject_type: str                    # person, organization, concept, event, policy, technology
    topic_focus: Optional[str]

    temporal_scope: TemporalScope
    geographic_scope: Optional[str]
    domain: Optional[str]

    complexity: ComplexityLevel
    estimated_sources_needed: int
    estimated_search_queries: int

    ambiguity_level: AmbiguityLevel
    ambiguity_reasons: List[str]
    needs_clarification: bool
    clarification_question: Optional[str]
    clarification_options: List[str]

    suggested_questions: List[str]
    suggested_outline: List[str]
    suggested_source_types: List[str]    # e.g. "news", "academic", "official"
    search_strategy: str

    analysis_confidence: float           # 0-1
    reasoning: str


class PlanQuery(TypedDict, total=False):
    qid: str
    query: str
    section: str
    lane: Lane
    priority: float              # 0-1, used to order execution


class ResearchQuestion(TypedDict, total=False):
    question: str
    queries: List[str]
    target_sources: List[str]    # e.g. "academic", "news", "official"
    confidence_weight: float     # how much this contributes to overall confidence


class ResearchTree(TypedDict, total=False):
    """Priority-ranked question buckets so we can bail early when time
    or budget runs out."""
    primary: List[ResearchQuestion]    # must answer
    secondary: List[ResearchQuestion]  # should answer
    tertiary: List[ResearchQuestion]   # nice to have


class Plan(TypedDict, total=False):
    topic: str
    outline: List[str]
    queries: List[PlanQuery]
    research_tree: ResearchTree


# Raw outputs accumulated from search workers before dedup/ranking
class RawSource(TypedDict, total=False):
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict, total=False):
    url: str
    title: str
    section: str
    text: str


# After dedup and ranking, raw items get stable IDs (S1, E1, C1...)
class Source(TypedDict, total=False):
    sid: str
    url: str
    title: str
    snippet: str
    score: float
    credibility: float


class Evidence(TypedDict, total=False):
    eid: str
    sid: str             # back-reference to parent Source
    url: str
    title: str
    section: str
    text: str


class Claim(TypedDict, total=False):
    cid: str
    section: str
    text: str
    confidence: float


class ClaimCitations(TypedDict, total=False):
    cid: str
    eids: List[str]


class Issue(TypedDict, total=False):
    level: str  # "warn" | "block"
    message: str
    related_cids: List[str]


class EntityCandidate(TypedDict, total=False):
    name: str
    description: str
    identifiers: List[str]
    confidence: float


class DiscoveryResult(TypedDict, total=False):
    """What the discovery node figured out about the user's query --
    entities, confidence, and whether we need to ask for clarification."""
    query_type: QueryType
    entity_candidates: List[EntityCandidate]
    confidence: float
    needs_clarification: bool
    anchor_terms: List[str]


class KnowledgeGap(TypedDict, total=False):
    """Detected during gap analysis between iterations. Each gap spawns
    new search queries in the next loop."""
    section: str
    description: str
    suggested_queries: List[str]
    priority: float               # 0-1
    current_confidence: float


class TrajectoryStep(TypedDict, total=False):
    """Breadcrumb for one action in the research loop -- useful for
    debugging and avoiding repeated failed strategies."""
    iteration: int
    action: str                   # "search", "read", "refine", "backtrack"
    query: str
    result_summary: str
    confidence_delta: float
    timestamp: str


class DeadEnd(TypedDict, total=False):
    """Logged when a search path yields nothing useful so we can
    backtrack and try a different angle."""
    query: str
    reason: str                   # "no_results", "irrelevant", "paywall", "low_credibility"
    iteration: int
    alternative_tried: bool


class SourceCredibility(TypedDict, total=False):
    """E-E-A-T inspired scoring. Each dimension is 0-1; overall is a
    weighted average we use to filter and rank sources."""
    sid: str
    url: str
    domain_trust: float
    freshness: float
    authority: float
    content_quality: float
    overall: float


class VerifiedCitation(TypedDict, total=False):
    """Links a claim to the exact span of source text that supports it.
    cross_validated means multiple independent sources agree."""
    cid: str
    eid: str
    claim_text: str
    evidence_span: str
    match_score: float            # 0-1 semantic similarity
    verified: bool
    cross_validated: bool
    supporting_sids: List[str]


class SubagentAssignment(TypedDict, total=False):
    subagent_id: str
    question: str
    queries: List[str]
    target_sources: List[str]


class SubagentFindings(TypedDict, total=False):
    """What a single worker returns to the orchestrator."""
    subagent_id: str
    question: str
    findings: str                 # compressed summary
    evidence_ids: List[str]
    confidence: float
    iterations_used: int
    dead_ends: List[DeadEnd]


class OrchestratorState(TypedDict, total=False):
    phase: str                    # "primary_research" | "secondary_research" | "synthesis"
    questions_assigned: int
    questions_completed: int
    overall_confidence: float


class ResearchMetadata(TypedDict, total=False):
    """End-of-run summary surfaced to the user alongside the report."""
    overall_confidence: float
    verified_claims: int
    total_claims: int
    knowledge_gaps: int
    sources_used: int
    research_iterations: int
    total_searches: int
    time_elapsed_seconds: float


class AgentState(MessagesState):
    """Top-level graph state. MessagesState gives us the `messages` list
    with LangGraph's add_messages reducer. Fields using operator.add are
    append-only accumulators that workers can safely extend in parallel."""

    # Runtime knobs
    depth: Optional[int]
    max_results: Optional[int]
    round: Optional[int]

    # Query understanding
    original_query: Optional[str]
    query_analysis: Optional[QueryAnalysisResult]

    # Human-in-the-loop clarification
    needs_clarification: Optional[bool]
    clarification_request: Optional[str]
    user_clarification: Optional[str]
    enriched_context: Optional[str]        # merged query + clarification for downstream nodes

    # Discovery
    discovery: Optional[DiscoveryResult]
    selected_entity: Optional[EntityCandidate]
    primary_anchor: Optional[str]          # main subject, injected into every search query
    anchor_terms: Optional[List[str]]
    human_clarification: Optional[str]     # deprecated -- use user_clarification

    # Planning
    plan: Optional[Plan]

    # Iterative research loop
    research_iteration: Optional[int]
    max_iterations: Optional[int]
    knowledge_gaps: Optional[List[KnowledgeGap]]
    research_trajectory: Optional[List[TrajectoryStep]]
    dead_ends: Optional[List[DeadEnd]]
    proceed_to_synthesis: Optional[bool]
    refinement_queries: Optional[List[Dict[str, Any]]]

    # Multi-agent orchestration
    subagent_assignments: Optional[List[SubagentAssignment]]
    subagent_findings: Annotated[List[SubagentFindings], operator.add]
    orchestrator_state: Optional[OrchestratorState]
    done_subagents: Annotated[int, operator.add]

    # Search worker accumulation (append-only via operator.add)
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]
    raw_sources: Annotated[List[RawSource], operator.add]
    raw_evidence: Annotated[List[RawEvidence], operator.add]

    # Normalized artifacts (set once after ranking/dedup)
    sources: Optional[List[Source]]
    evidence: Optional[List[Evidence]]
    source_credibility: Optional[Dict[str, SourceCredibility]]

    # Trust engine
    claims: Optional[List[Claim]]
    citations: Optional[List[ClaimCitations]]
    issues: Optional[List[Issue]]

    # Citation verification
    verified_citations: Optional[List[VerifiedCitation]]
    unverified_claims: Optional[List[str]]
    hallucination_score: Optional[float]   # fraction of claims with no supporting evidence

    # Confidence tracking
    claim_confidence: Optional[Dict[str, float]]
    section_confidence: Optional[Dict[str, float]]
    overall_confidence: Optional[float]
    previous_confidence: Optional[float]   # stashed so gap detector can compute delta

    # Cross-validation
    cross_validated_claims: Optional[List[VerifiedCitation]]
    single_source_claims: Optional[List[VerifiedCitation]]

    # Control flow
    needs_more: Optional[bool]
    stop: Optional[bool]

    # Final output
    report: Optional[str]
    research_metadata: Optional[ResearchMetadata]
