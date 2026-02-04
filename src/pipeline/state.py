"""
Research state -- the single TypedDict that every node reads from and writes to.

Extends MessagesState so we get a `messages` field with LangGraph's built-in
add_messages reducer. The user's query comes in as a HumanMessage, internal
steps can optionally append messages, and the final report goes out as an
AIMessage. This keeps the graph compatible with LangGraph Studio and chat UIs.

Fields wrapped in Annotated[..., operator.add] are append-only accumulators.
We use operator.add so parallel workers can safely append without overwriting
each other -- LangGraph merges them automatically.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List
from typing_extensions import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Configuration -- lives in RunnableConfig["configurable"], NOT in state.
# Registered via context_schema on the StateGraph so LangGraph Studio
# renders these as editable config knobs.
# ---------------------------------------------------------------------------

class Configuration(TypedDict, total=False):
    """Pipeline-level settings passed via config={"configurable": {...}}."""
    model: str                  # LLM for planning/writing (default "gpt-4o")
    fast_model: str             # LLM for extraction (default "gpt-4o-mini")
    max_search_results: int     # Tavily results per query (default 5)
    report_structure: str       # "detailed" | "concise" | "bullet_points"
    system_prompt: str          # custom instructions prepended to the writer


CONFIGURATION_DEFAULTS: Dict[str, Any] = {
    "model": "gpt-4o",
    "fast_model": "gpt-4o-mini",
    "max_search_results": 5,
    "report_structure": "detailed",
    "system_prompt": "",
}


def get_configuration(config: RunnableConfig) -> dict:
    """Read pipeline configuration from a RunnableConfig, with defaults."""
    configurable = config.get("configurable", {})
    return {
        key: configurable.get(key, default)
        for key, default in CONFIGURATION_DEFAULTS.items()
    }


class ResearchState(MessagesState, total=False):
    """Graph state. Inherits `messages` (with add_messages reducer) from
    MessagesState. All other fields are research-specific."""

    # -- User input (set once when the graph is invoked) --
    query: str
    mode: str                           # "single" or "multi"
    max_iterations: int                 # cap on search-gap loops (default 5)
    min_coverage: float                 # bail early if coverage passes this (default 0.7)

    # -- Understanding & clarification --
    understanding: Dict[str, Any]       # LLM's breakdown of what the user wants
    is_ambiguous: bool                  # True means we need to ask the user first
    clarification_question: str
    clarification_options: List[Dict[str, str]]  # [{label, description}, ...]
    user_clarification: str             # filled in via update_state() after HITL pause

    # -- Planning --
    research_questions: List[str]       # questions the final report should answer
    aspects_to_cover: List[str]         # angles / dimensions to look at
    pending_searches: List[str]         # search queries waiting to run

    # -- Multi-agent orchestration --
    sub_questions: List[Dict]           # orchestrator splits the query into these
    worker_results: Annotated[List[Dict], operator.add]   # each worker appends here
    total_workers: int
    done_workers: Annotated[int, operator.add]            # bumped by 1 per worker
    synthesis: Dict[str, Any]           # merged view after all workers finish

    # -- Iteration tracking --
    iteration: int
    coverage: float                     # 0-1, how much of the topic we've covered
    ready_to_write: bool                # True once the gap detector says "go"
    gaps: List[str]                     # what's still missing

    # -- Evidence (append-only so parallel workers don't clobber each other) --
    evidence: Annotated[List[Dict], operator.add]
    sources: Dict[str, Dict]            # url -> {title, source_id, ...}
    searches_done: List[str]            # queries we've already run (skip dupes)

    # -- Final output --
    verified_claims: List[Dict]
    confidence: float                   # fraction of claims backed by evidence
    report: str                         # the markdown report we hand back
    metadata: Dict[str, Any]            # run stats: iterations, timing, etc.
