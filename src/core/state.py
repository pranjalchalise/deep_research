from __future__ import annotations

import operator
from typing import Dict, List, Optional, TypedDict
from typing_extensions import Annotated

from langgraph.graph import MessagesState


# ---------- Planning ----------
class PlanQuery(TypedDict):
    qid: str
    query: str
    section: str


class Plan(TypedDict):
    topic: str
    outline: List[str]
    queries: List[PlanQuery]


# ---------- Raw worker outputs (accumulated) ----------
class RawSource(TypedDict):
    url: str
    title: str
    snippet: str


class RawEvidence(TypedDict):
    url: str
    title: str
    section: str
    text: str  # short extracted evidence text


# ---------- Normalized artifacts ----------
class Source(TypedDict):
    sid: str  # S1
    url: str
    title: str
    snippet: str


class Evidence(TypedDict):
    eid: str  # E1
    sid: str  # S1
    url: str
    title: str
    section: str
    text: str


class Claim(TypedDict):
    cid: str  # C1
    section: str
    text: str


class ClaimCitations(TypedDict):
    cid: str
    eids: List[str]


class Issue(TypedDict):
    level: str  # "warn" | "block"
    message: str
    related_cids: List[str]


# ---------- Agent State ----------
class AgentState(MessagesState):
    # required by assignment:
    # messages: handled by MessagesState reducer

    # config knobs
    depth: Optional[int]          # how many plan queries to generate
    max_results: Optional[int]    # per query for search

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

    # output
    report: Optional[str]
