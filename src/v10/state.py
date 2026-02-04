"""
ResearchState — the single TypedDict that every node reads from and writes to.

Fields using ``Annotated[..., operator.add]`` are append-only accumulators
safe for parallel Send() workers.  Everything else is set/overwritten by
whichever node runs.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List, TypedDict
from typing_extensions import Annotated


class ResearchState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────
    query: str
    mode: str                    # "single" or "multi"
    max_iterations: int          # default 5
    min_coverage: float          # default 0.7

    # ── Understanding ──────────────────────────────────────────────────
    understanding: Dict[str, Any]
    is_ambiguous: bool
    clarification_question: str
    clarification_options: List[Dict[str, str]]
    user_clarification: str      # injected via update_state after interrupt

    # ── Planning ───────────────────────────────────────────────────────
    research_questions: List[str]
    aspects_to_cover: List[str]
    pending_searches: List[str]

    # ── Multi-agent ────────────────────────────────────────────────────
    sub_questions: List[Dict]
    worker_results: Annotated[List[Dict], operator.add]
    total_workers: int
    done_workers: Annotated[int, operator.add]
    synthesis: Dict[str, Any]

    # ── Iteration ──────────────────────────────────────────────────────
    iteration: int
    coverage: float
    ready_to_write: bool
    gaps: List[str]

    # ── Evidence (append-only for parallel workers) ────────────────────
    evidence: Annotated[List[Dict], operator.add]
    sources: Dict[str, Dict]     # url → source dict, merged explicitly in nodes
    searches_done: List[str]

    # ── Output ─────────────────────────────────────────────────────────
    verified_claims: List[Dict]
    confidence: float
    report: str
    metadata: Dict[str, Any]
