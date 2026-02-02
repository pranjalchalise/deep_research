# src/core/graph.py
"""
LangGraph state machine for the research pipeline.

Features:
- Human-in-the-loop disambiguation via interrupt
- Discovery phase for entity identification
- Parallel search workers with full page fetching
- Trust engine (claims → cite → verify)

v8 Features:
- Orchestrator-worker multi-agent pattern
- Iterative research with gap detection
- Backtracking on dead ends
- E-E-A-T source credibility scoring
- Span verification and cross-validation
- Per-claim confidence scoring
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.core.state import AgentState

# === Discovery Phase ===
from src.nodes.discovery import (
    analyzer_node,
    discovery_node,
    clarify_node,
    route_after_discovery,
)

# === v7 nodes (kept for backwards compatibility) ===
from src.nodes.planner import planner_node
from src.nodes.search_worker import worker_node
from src.nodes.reducer import reducer_node
from src.nodes.ranker import ranker_node
from src.nodes.claims import claims_node
from src.nodes.cite import cite_node
from src.nodes.verify import verify_node
from src.nodes.writer import writer_node

# === v8 nodes ===
from src.nodes.planner_v8 import planner_node_v8
from src.nodes.writer_v8 import writer_node_v8
from src.nodes.iterative import (
    confidence_check_node,
    auto_refine_node,
    gap_detector_node,
    backtrack_handler_node,
    route_after_confidence,
    route_after_gaps,
)
from src.nodes.orchestrator import (
    orchestrator_node,
    subagent_node,
    synthesizer_node,
    fanout_subagents,
    route_after_synthesis,
)
from src.nodes.trust_engine import (
    credibility_scorer_node,
    span_verify_node,
    cross_validate_node,
    claim_confidence_scorer_node,
)


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


# ============================================================================
# V8 GRAPH - Multi-Agent Orchestration with Iterative Research
# ============================================================================

def fanout_workers_v8(state: AgentState):
    """Fan out to parallel search workers (v8 - only used in fallback mode)."""
    plan = state.get("plan") or {}
    queries = plan.get("queries") or []
    return [Send("worker", {"query_item": qi}) for qi in queries]


def route_after_reduce_v8(state: AgentState) -> str:
    """Route after reduce - proceed to credibility when all workers done."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "credibility" if done >= total else END


def route_after_backtrack(state: AgentState) -> str:
    """Route after backtrack - go to orchestrator for next iteration."""
    return "orchestrator"


def build_v8_graph(
    checkpointer: Optional[MemorySaver] = None,
    interrupt_on_clarify: bool = True,
    use_multi_agent: bool = True,
):
    """
    Build the v8 research pipeline graph with multi-agent orchestration.

    Architecture:
    ```
    START → Analyzer → Discovery → Confidence Check
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
                Clarify            Auto Refine          Planner
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        ↓
                                   Orchestrator
                                        ↓
                    ┌─────────────┬─────┴─────┬─────────────┐
                    ↓             ↓           ↓             ↓
                Subagent 1   Subagent 2  Subagent 3   Subagent N
                    │             │           │             │
                    └─────────────┴─────┬─────┴─────────────┘
                                        ↓
                                   Synthesizer
                                        ↓
                                   Gap Detector
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
                Backtrack          Orchestrator          Reduce
               (dead ends)        (more research)      (sufficient)
                    │                   │                   │
                    └───────────────────┤                   │
                                        │                   ↓
                                        │              Credibility
                                        │                   ↓
                                        │               Ranker
                                        │                   ↓
                                        │               Claims
                                        │                   ↓
                                        │                Cite
                                        │                   ↓
                                        │            Span Verify
                                        │                   ↓
                                        │          Cross Validate
                                        │                   ↓
                                        │         Confidence Score
                                        │                   ↓
                                        └───────────────→ Writer
                                                            ↓
                                                           END
    ```

    Args:
        checkpointer: Optional checkpointer for state persistence
        interrupt_on_clarify: Whether to interrupt for human clarification
        use_multi_agent: Whether to use orchestrator-worker pattern (default: True)

    Returns:
        Compiled LangGraph
    """
    g = StateGraph(AgentState)

    # === Discovery Phase ===
    g.add_node("analyzer", analyzer_node)
    g.add_node("discovery", discovery_node)
    g.add_node("confidence_check", confidence_check_node)
    g.add_node("clarify", clarify_node)
    g.add_node("auto_refine", auto_refine_node)

    # === Planning Phase ===
    g.add_node("planner", planner_node_v8)

    # === Multi-Agent Research Phase ===
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("subagent", subagent_node)
    g.add_node("synthesizer", synthesizer_node)

    # === Iterative Research ===
    g.add_node("gap_detector", gap_detector_node)
    g.add_node("backtrack", backtrack_handler_node)

    # === Fallback: Single-agent workers (when multi-agent disabled) ===
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    # === Trust Engine ===
    g.add_node("credibility", credibility_scorer_node)
    g.add_node("ranker", ranker_node)
    g.add_node("claims", claims_node)
    g.add_node("cite", cite_node)
    g.add_node("span_verify", span_verify_node)
    g.add_node("cross_validate", cross_validate_node)
    g.add_node("confidence_score", claim_confidence_scorer_node)
    g.add_node("write", writer_node_v8)

    # ========================================================================
    # EDGES
    # ========================================================================

    # --- Discovery Flow ---
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "discovery")
    g.add_edge("discovery", "confidence_check")

    # Route based on confidence level
    g.add_conditional_edges(
        "confidence_check",
        route_after_confidence,
        {
            "clarify": "clarify",
            "auto_refine": "auto_refine",
            "planner": "planner",
        }
    )

    # After clarification or auto-refine, go to planner
    g.add_edge("clarify", "planner")

    # Auto-refine routes back to confidence check or proceeds
    g.add_conditional_edges(
        "auto_refine",
        route_after_confidence,
        {
            "clarify": "clarify",
            "planner": "planner",
            "auto_refine": "auto_refine",  # Can loop for more refinement
        }
    )

    # --- Research Flow (Multi-Agent vs Single-Agent) ---
    if use_multi_agent:
        # Planner → Orchestrator
        g.add_edge("planner", "orchestrator")

        # Orchestrator fans out to parallel subagents
        g.add_conditional_edges("orchestrator", fanout_subagents)

        # Subagents feed into synthesizer
        g.add_edge("subagent", "synthesizer")

        # Synthesizer → Gap Detector
        g.add_conditional_edges(
            "synthesizer",
            route_after_synthesis,
            {
                "gap_detector": "gap_detector",
            }
        )

        # Gap detector decides: continue research, backtrack, or proceed
        g.add_conditional_edges(
            "gap_detector",
            route_after_gaps,
            {
                "orchestrator": "orchestrator",  # More research needed
                "backtrack": "backtrack",        # Handle dead ends
                "reduce": "reduce",              # Sufficient coverage
            }
        )

        # Backtrack → Orchestrator (try alternative approaches)
        g.add_conditional_edges(
            "backtrack",
            route_after_backtrack,
            {
                "orchestrator": "orchestrator",
            }
        )

        # Reduce collects all evidence → Credibility
        g.add_conditional_edges(
            "reduce",
            route_after_reduce_v8,
            {
                "credibility": "credibility",
            }
        )

    else:
        # Fallback: Single-agent mode (like v7)
        g.add_conditional_edges("planner", fanout_workers_v8)
        g.add_edge("worker", "reduce")
        g.add_conditional_edges(
            "reduce",
            route_after_reduce_v8,
            {
                "credibility": "credibility",
            }
        )

    # --- Trust Engine Flow ---
    g.add_edge("credibility", "ranker")
    g.add_edge("ranker", "claims")
    g.add_edge("claims", "cite")
    g.add_edge("cite", "span_verify")
    g.add_edge("span_verify", "cross_validate")
    g.add_edge("cross_validate", "confidence_score")
    g.add_edge("confidence_score", "write")
    g.add_edge("write", END)

    # ========================================================================
    # COMPILE
    # ========================================================================

    compile_kwargs = {}

    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


def build_v8_graph_with_memory(
    interrupt_on_clarify: bool = True,
    use_multi_agent: bool = True,
):
    """
    Build v8 graph with in-memory checkpointer (for human-in-the-loop).

    Returns:
        Tuple of (compiled_graph, checkpointer)
    """
    checkpointer = MemorySaver()
    graph = build_v8_graph(
        checkpointer=checkpointer,
        interrupt_on_clarify=interrupt_on_clarify,
        use_multi_agent=use_multi_agent,
    )
    return graph, checkpointer


def build_v8_simple_graph(use_multi_agent: bool = True):
    """
    Build v8 graph without human-in-the-loop.

    Use this for automated pipelines that shouldn't wait for human input.
    """
    return build_v8_graph(
        checkpointer=None,
        interrupt_on_clarify=False,
        use_multi_agent=use_multi_agent,
    )
