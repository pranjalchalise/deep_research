"""
LangGraph StateGraph for v10 research pipeline.

Supports both single-agent (iterative search → gap detect → loop) and
multi-agent (orchestrate → parallel workers → synthesize → loop) in one
graph, selected by the ``mode`` field in ResearchState.

Usage:
    # Automated (no HITL)
    from src.v10.graph import build_graph
    g = build_graph(interrupt_on_clarify=False)
    result = g.invoke({"query": "...", "mode": "single", "iteration": 0,
                        "evidence": [], "worker_results": [], "done_workers": 0,
                        "searches_done": [], "sources": {}})

    # With HITL
    from langgraph.checkpoint.memory import MemorySaver
    cp = MemorySaver()
    g = build_graph(checkpointer=cp)
    config = {"configurable": {"thread_id": "abc"}}
    result = g.invoke({"query": "Tell me about Python", ...}, config)
    # paused at clarify → inject clarification → resume
    g.update_state(config, {"user_clarification": "the programming language"})
    result = g.invoke(None, config)
"""
from __future__ import annotations

from typing import List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.v10.state import ResearchState
from src.v10.nodes import (
    understand_node,
    clarify_node,
    plan_node,
    orchestrate_node,
    search_worker_node,
    collect_node,
    synthesize_node,
    search_and_extract_node,
    detect_gaps_node,
    verify_node,
    write_report_node,
)


# ── Routing functions ──────────────────────────────────────────────────

def route_after_understand(state: ResearchState) -> str:
    """Route to clarify (HITL) if the query is ambiguous, else straight to plan."""
    return "clarify" if state.get("is_ambiguous") else "plan"


def route_after_plan(state: ResearchState) -> str:
    """Pick the single-agent or multi-agent research path."""
    return "orchestrate" if state.get("mode") == "multi" else "search_and_extract"


def fanout_workers(state: ResearchState) -> List[Send]:
    """Fan out one Send() per sub-question for parallel research."""
    return [
        Send("search_worker", {
            "sub_question": sq,
            "query": state["query"],
            "user_clarification": state.get("user_clarification", ""),
        })
        for sq in state.get("sub_questions", [])
    ]


def route_after_gaps(state: ResearchState) -> str:
    """Decide whether the single-agent loop should iterate or proceed."""
    if state.get("ready_to_write"):
        return "verify"
    if state.get("coverage", 0) >= state.get("min_coverage", 0.7):
        return "verify"
    if not state.get("pending_searches"):
        return "verify"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "verify"
    return "search_and_extract"


def route_after_synthesis(state: ResearchState) -> str:
    """Decide whether the multi-agent loop should iterate or proceed."""
    syn = state.get("synthesis", {})
    if not syn.get("needs_more_research"):
        return "verify"
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        return "verify"
    if not syn.get("additional_searches"):
        return "verify"
    return "orchestrate"


# ── Graph builder ──────────────────────────────────────────────────────

def build_graph(
    checkpointer=None,
    interrupt_on_clarify: bool = True,
):
    """Build and compile the v10 research StateGraph.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. MemorySaver) for
            persistence and HITL.  ``None`` for fire-and-forget runs.
        interrupt_on_clarify: If True, the graph pauses *before* the
            ``clarify`` node so the caller can inject
            ``user_clarification`` via ``update_state()``.
    """
    g = StateGraph(ResearchState)

    # ── Add nodes ──────────────────────────────────────────────────
    g.add_node("understand", understand_node)
    g.add_node("clarify", clarify_node)
    g.add_node("plan", plan_node)

    # Multi-agent path
    g.add_node("orchestrate", orchestrate_node)
    g.add_node("search_worker", search_worker_node)
    g.add_node("collect", collect_node)
    g.add_node("synthesize", synthesize_node)

    # Single-agent path
    g.add_node("search_and_extract", search_and_extract_node)
    g.add_node("detect_gaps", detect_gaps_node)

    # Shared tail
    g.add_node("verify", verify_node)
    g.add_node("write_report", write_report_node)

    # ── Wire edges ─────────────────────────────────────────────────

    # START → understand
    g.add_edge(START, "understand")

    # understand → clarify | plan
    g.add_conditional_edges(
        "understand",
        route_after_understand,
        {"clarify": "clarify", "plan": "plan"},
    )

    # clarify → plan
    g.add_edge("clarify", "plan")

    # plan → search_and_extract | orchestrate
    g.add_conditional_edges(
        "plan",
        route_after_plan,
        {"search_and_extract": "search_and_extract", "orchestrate": "orchestrate"},
    )

    # ── Single-agent loop ──────────────────────────────────────────
    g.add_edge("search_and_extract", "detect_gaps")
    g.add_conditional_edges(
        "detect_gaps",
        route_after_gaps,
        {"search_and_extract": "search_and_extract", "verify": "verify"},
    )

    # ── Multi-agent loop ───────────────────────────────────────────
    g.add_conditional_edges("orchestrate", fanout_workers)
    g.add_edge("search_worker", "collect")
    g.add_edge("collect", "synthesize")
    g.add_conditional_edges(
        "synthesize",
        route_after_synthesis,
        {"orchestrate": "orchestrate", "verify": "verify"},
    )

    # ── Shared tail ────────────────────────────────────────────────
    g.add_edge("verify", "write_report")
    g.add_edge("write_report", END)

    # ── Compile ────────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


# ── Module-level compiled graph (for LangGraph Studio / langgraph.json) ──
graph = build_graph(checkpointer=None, interrupt_on_clarify=True)
