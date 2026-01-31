# src/core/graph.py
"""
LangGraph state machine for the research pipeline.

Features:
- Human-in-the-loop disambiguation via interrupt
- Discovery phase for entity identification
- Parallel search workers with full page fetching
- Trust engine (claims → cite → verify)
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.core.state import AgentState
from src.nodes.discovery import (
    analyzer_node,
    discovery_node,
    clarify_node,
    route_after_discovery,
)
from src.nodes.planner import planner_node
from src.nodes.search_worker import worker_node
from src.nodes.reducer import reducer_node
from src.nodes.ranker import ranker_node
from src.nodes.claims import claims_node
from src.nodes.cite import cite_node
from src.nodes.verify import verify_node
from src.nodes.writer import writer_node


def fanout_workers(state: AgentState):
    """Fan out to parallel search workers based on plan queries."""
    plan = state.get("plan") or {}
    queries = plan.get("queries") or []
    return [Send("worker", {"query_item": qi}) for qi in queries]


def route_after_reduce(state: AgentState) -> str:
    """Route after reduce - proceed to ranker when all workers done."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "ranker" if done >= total else END


def build_graph(checkpointer: Optional[MemorySaver] = None, interrupt_on_clarify: bool = True):
    """
    Build the research pipeline graph.

    Args:
        checkpointer: Optional checkpointer for state persistence (required for interrupt)
        interrupt_on_clarify: Whether to interrupt for human clarification (default: True)

    Returns:
        Compiled LangGraph
    """
    g = StateGraph(AgentState)

    # === Discovery Phase (NEW) ===
    g.add_node("analyzer", analyzer_node)
    g.add_node("discovery", discovery_node)
    g.add_node("clarify", clarify_node)

    # === Research Phase ===
    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)
    g.add_node("ranker", ranker_node)

    # === Trust Engine ===
    g.add_node("claims", claims_node)
    g.add_node("cite", cite_node)
    g.add_node("verify", verify_node)
    g.add_node("write", writer_node)

    # === Edges ===

    # Start with analysis
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "discovery")

    # After discovery, route based on confidence
    g.add_conditional_edges(
        "discovery",
        route_after_discovery,
        {
            "clarify": "clarify",
            "planner": "planner",
        }
    )

    # After clarification, proceed to planning
    g.add_edge("clarify", "planner")

    # Planning fans out to workers
    g.add_conditional_edges("planner", fanout_workers)

    # Workers feed into reducer
    g.add_edge("worker", "reduce")

    # After reduce, proceed to ranker
    g.add_conditional_edges("reduce", route_after_reduce)

    # Ranker filters sources then proceeds to trust engine
    g.add_edge("ranker", "claims")

    # Trust engine pipeline
    g.add_edge("claims", "cite")
    g.add_edge("cite", "verify")
    g.add_edge("verify", "write")
    g.add_edge("write", END)

    # Compile with optional interrupt
    compile_kwargs = {}

    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    if interrupt_on_clarify:
        # Interrupt BEFORE clarify node to get human input
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


def build_graph_with_memory(interrupt_on_clarify: bool = True):
    """
    Build graph with in-memory checkpointer (for human-in-the-loop).

    Returns:
        Tuple of (compiled_graph, checkpointer)
    """
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer, interrupt_on_clarify=interrupt_on_clarify)
    return graph, checkpointer


# === Convenience function for simple usage (no interrupt) ===

def build_simple_graph():
    """
    Build graph without human-in-the-loop (auto-proceeds on low confidence).

    Use this for automated pipelines that shouldn't wait for human input.
    """
    return build_graph(checkpointer=None, interrupt_on_clarify=False)
