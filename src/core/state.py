# src/core/state.py
from __future__ import annotations

import operator
from typing import Dict, List, Optional, TypedDict, Literal
from typing_extensions import Annotated

from langgraph.graph import MessagesState


# -------------------------
# Lanes / routing category
# -------------------------
# Planner can tag queries with a "lane" to control retrieval behavior.
# Keep it as a Literal so Pylance + type-checkers are happy.
Lane = Literal[
    "general",       # broad web (no domain filter)
    "docs",          # official docs
    "papers",        # academic / arxiv / journals
    "code",          # github / repos
    "news",          # news / announcements
    "forums",        # reddit / discourse / hn
]


# ---------- Planning ----------
class PlanQuery(TypedDict, total=False):
    qid: str
    query: str
    section: str
    lane: Lane  # optional, but planner may set it


class Plan(TypedDict, total=False):
    topic: str
    outline: List[str]
    queries: List[PlanQuery]


# ---------- Raw worker outputs (accumulated) ----------
class RawSource(TypedDict, total=False):
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict, total=False):
    url: str
    title: str
    section: str
    text: str  # short extracted evidence text


# ---------- Normalized artifacts ----------
class Source(TypedDict, total=False):
    sid: str  # S1
    url: str
    title: str
    snippet: str
    score: float  # optional: ranker can attach


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


class ClaimCitations(TypedDict, total=False):
    cid: str
    eids: List[str]


class Issue(TypedDict, total=False):
    level: str  # "warn" | "block"
    message: str
    related_cids: List[str]


# ---------- Entity / Disambiguation ----------
QueryType = Literal["person", "concept", "technical", "event", "comparison", "organization", "general"]


class EntityCandidate(TypedDict, total=False):
    """A possible entity match found during discovery."""
    name: str
    description: str
    identifiers: List[str]  # unique identifiers (institution, role, url, etc.)
    confidence: float


class DiscoveryResult(TypedDict, total=False):
    """Results from the discovery phase."""
    query_type: QueryType
    entity_candidates: List[EntityCandidate]
    confidence: float  # 0-1, how confident we are about the entity
    needs_clarification: bool
    anchor_terms: List[str]  # terms to include in all subsequent searches


# ---------- Agent State ----------
class AgentState(MessagesState):
    # MessagesState already defines:
    # messages: Annotated[list[BaseMessage], add_messages]

    # config knobs
    depth: Optional[int]          # how many plan queries to generate
    max_results: Optional[int]    # per query for search
    round: Optional[int]          # refinement round counter

    # === NEW: Discovery & Disambiguation ===
    original_query: Optional[str]           # the user's original question
    discovery: Optional[DiscoveryResult]    # results from discovery phase
    selected_entity: Optional[EntityCandidate]  # user-selected or auto-selected entity

    # Anchor term hierarchy:
    # - primary_anchor: The main subject (MUST appear in EVERY search query)
    # - anchor_terms: Context/qualifiers (appear in SOME queries for disambiguation)
    primary_anchor: Optional[str]           # e.g., "Pranjal Chalise" - always in queries
    anchor_terms: Optional[List[str]]       # e.g., ["Amherst College"] - context qualifiers

    clarification_request: Optional[str]    # question to ask user
    human_clarification: Optional[str]      # user's response

    # planning
    plan: Optional[Plan]

    # worker accumulation (reducers)
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]  # worker returns +1

    raw_sources: Annotated[List[RawSource], operator.add]
    raw_evidence: Annotated[List[RawEvidence], operator.add]

    # normalized artifacts
    sources: Optional[List[Source]]
    evidence: Optional[List[Evidence]]

    # trust engine
    claims: Optional[List[Claim]]
    citations: Optional[List[ClaimCitations]]
    issues: Optional[List[Issue]]

    # controller / assess
    needs_more: Optional[bool]
    stop: Optional[bool]

    # output
    report: Optional[str]
