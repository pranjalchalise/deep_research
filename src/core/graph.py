# src/core/graph.py
"""
LangGraph state machine for v8 research pipeline.

This builds the graph that research_v8.py runs. It wires together all the
nodes (discovery, planning, multi-agent orchestration, trust engine, etc.)
into a single compiled LangGraph.

Three graph variants:
  - build_v8_graph:           Full multi-agent with gap detection + backtracking
  - build_v8_optimized_graph: Same but with batched trust engine + complexity routing
  - Each has _with_memory (for HITL) and _simple (no interrupt) helpers
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.core.state import AgentState
from src.core.config import V8Config

# -- Discovery phase: analyze query, identify entities, ask for clarification --
from src.nodes.discovery import (
    analyzer_node,
    discovery_node,
    clarify_node,
    route_after_discovery,
)

# -- Shared nodes used by v8 in both multi-agent and fallback single-agent modes --
from src.nodes.search_worker import worker_node
from src.nodes.reducer import reducer_node
from src.nodes.ranker import ranker_node
from src.nodes.claims import claims_node
from src.nodes.cite import cite_node

# -- v8 planning and writing --
from src.nodes.planner_v8 import planner_node_v8
from src.nodes.writer_v8 import writer_node_v8

# -- v8 iterative research: confidence checks, gap detection, backtracking --
from src.nodes.iterative import (
    confidence_check_node,
    auto_refine_node,
    gap_detector_node,
    backtrack_handler_node,
    route_after_confidence,
    route_after_gaps,
    complexity_router_node,
    early_termination_check_node,
    route_by_complexity,
    route_after_termination_check,
)

# -- v8 multi-agent orchestration: break query into sub-tasks, fan out, synthesize --
from src.nodes.orchestrator import (
    orchestrator_node,
    subagent_node,
    synthesizer_node,
    fanout_subagents,
    route_after_synthesis,
)

# -- v8 trust engine: source credibility, span verification, cross-validation --
from src.nodes.trust_engine import (
    credibility_scorer_node,
    span_verify_node,
    cross_validate_node,
    claim_confidence_scorer_node,
)

# -- v8 batched trust engine: fewer LLM calls by combining credibility+claims, verify+cross --
from src.nodes.trust_engine_batched import (
    batched_credibility_claims_node,
    batched_verification_node,
)


# ============================================================================
# V8 GRAPH - Multi-Agent Orchestration with Iterative Research
# ============================================================================

def fanout_workers_v8(state: AgentState):
    """Send each planned query to a parallel worker node (single-agent fallback)."""
    plan = state.get("plan") or {}
    queries = plan.get("queries") or []
    return [Send("worker", {"query_item": qi}) for qi in queries]


def route_after_reduce_v8(state: AgentState) -> str:
    """Wait for all workers to finish, then move to credibility scoring."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "credibility" if done >= total else END


def route_after_backtrack(state: AgentState) -> str:
    """After handling a dead end, send the query back to the orchestrator."""
    return "orchestrator"


def build_v8_graph(
    checkpointer: Optional[MemorySaver] = None,
    interrupt_on_clarify: bool = True,
    use_multi_agent: bool = True,
):
    """
    Build the full v8 research pipeline as a LangGraph.

    Flow:
      Analyzer → Discovery → Confidence Check → Clarify/Refine? → Planner
        → Orchestrator → [Subagents in parallel] → Synthesizer
        → Gap Detector → (loop or proceed) → Reduce → Trust Engine → Writer

    If use_multi_agent=False, falls back to single-agent workers instead
    of the orchestrator-subagent pattern.
    """
    g = StateGraph(AgentState)

    # -- Discovery: figure out what the query is about --
    g.add_node("analyzer", analyzer_node)
    g.add_node("discovery", discovery_node)
    g.add_node("confidence_check", confidence_check_node)
    g.add_node("clarify", clarify_node)
    g.add_node("auto_refine", auto_refine_node)

    # -- Planning: decide what to search for --
    g.add_node("planner", planner_node_v8)

    # -- Multi-agent research: orchestrator breaks query, subagents research in parallel --
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("subagent", subagent_node)
    g.add_node("synthesizer", synthesizer_node)

    # -- Iterative loop: detect gaps, backtrack on dead ends --
    g.add_node("gap_detector", gap_detector_node)
    g.add_node("backtrack", backtrack_handler_node)

    # -- Single-agent fallback (when multi-agent is disabled) --
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    # -- Trust engine: score credibility, extract claims, verify, cross-validate --
    g.add_node("credibility", credibility_scorer_node)
    g.add_node("ranker", ranker_node)
    g.add_node("claims", claims_node)
    g.add_node("cite", cite_node)
    g.add_node("span_verify", span_verify_node)
    g.add_node("cross_validate", cross_validate_node)
    g.add_node("confidence_score", claim_confidence_scorer_node)
    g.add_node("write", writer_node_v8)

    # -- Edges: discovery flow --
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "discovery")
    g.add_edge("discovery", "confidence_check")

    g.add_conditional_edges(
        "confidence_check",
        route_after_confidence,
        {"clarify": "clarify", "auto_refine": "auto_refine", "planner": "planner"},
    )
    g.add_edge("clarify", "planner")
    g.add_conditional_edges(
        "auto_refine",
        route_after_confidence,
        {"clarify": "clarify", "planner": "planner", "auto_refine": "auto_refine"},
    )

    # -- Edges: research flow --
    if use_multi_agent:
        g.add_edge("planner", "orchestrator")
        g.add_conditional_edges("orchestrator", fanout_subagents)
        g.add_edge("subagent", "synthesizer")
        g.add_conditional_edges(
            "synthesizer", route_after_synthesis, {"gap_detector": "gap_detector"},
        )
        g.add_conditional_edges(
            "gap_detector", route_after_gaps,
            {"orchestrator": "orchestrator", "backtrack": "backtrack", "reduce": "reduce"},
        )
        g.add_conditional_edges(
            "backtrack", route_after_backtrack, {"orchestrator": "orchestrator"},
        )
        g.add_conditional_edges(
            "reduce", route_after_reduce_v8, {"credibility": "credibility"},
        )
    else:
        g.add_conditional_edges("planner", fanout_workers_v8)
        g.add_edge("worker", "reduce")
        g.add_conditional_edges(
            "reduce", route_after_reduce_v8, {"credibility": "credibility"},
        )

    # -- Edges: trust engine pipeline --
    g.add_edge("credibility", "ranker")
    g.add_edge("ranker", "claims")
    g.add_edge("claims", "cite")
    g.add_edge("cite", "span_verify")
    g.add_edge("span_verify", "cross_validate")
    g.add_edge("cross_validate", "confidence_score")
    g.add_edge("confidence_score", "write")
    g.add_edge("write", END)

    # -- Compile --
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


def build_v8_graph_with_memory(interrupt_on_clarify: bool = True, use_multi_agent: bool = True):
    """Build v8 graph with in-memory checkpointer for human-in-the-loop."""
    checkpointer = MemorySaver()
    graph = build_v8_graph(
        checkpointer=checkpointer,
        interrupt_on_clarify=interrupt_on_clarify,
        use_multi_agent=use_multi_agent,
    )
    return graph, checkpointer


def build_v8_simple_graph(use_multi_agent: bool = True):
    """Build v8 graph without HITL interrupt (for automated pipelines)."""
    return build_v8_graph(
        checkpointer=None, interrupt_on_clarify=False, use_multi_agent=use_multi_agent,
    )


# ============================================================================
# OPTIMIZED V8 GRAPH
# Same pipeline but with batched trust engine (fewer LLM calls) and
# complexity-based routing (simple queries skip multi-agent overhead).
# ============================================================================

def route_after_reduce_optimized(state: AgentState) -> str:
    """Wait for all workers, then move to the batched trust engine."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "batched_trust" if done >= total else END


def build_v8_optimized_graph(
    checkpointer: Optional[MemorySaver] = None,
    interrupt_on_clarify: bool = True,
    use_multi_agent: bool = True,
):
    """
    Build the optimized v8 pipeline.

    Differences from build_v8_graph:
      - Complexity router skips multi-agent for simple queries
      - Batched trust engine: credibility+claims in one LLM call,
        verify+cross-validate+confidence in one LLM call
      - Early termination when evidence stops improving
    """
    g = StateGraph(AgentState)

    # -- Discovery --
    g.add_node("analyzer", analyzer_node)
    g.add_node("discovery", discovery_node)
    g.add_node("complexity_router", complexity_router_node)
    g.add_node("clarify", clarify_node)
    g.add_node("auto_refine", auto_refine_node)

    # -- Planning --
    g.add_node("planner", planner_node_v8)

    # -- Multi-agent research --
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("subagent", subagent_node)
    g.add_node("synthesizer", synthesizer_node)

    # -- Iteration control --
    g.add_node("early_termination", early_termination_check_node)
    g.add_node("gap_detector", gap_detector_node)
    g.add_node("backtrack", backtrack_handler_node)

    # -- Single-agent fallback --
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    # -- Batched trust engine (the optimization) --
    g.add_node("batched_cred_claims", batched_credibility_claims_node)
    g.add_node("ranker", ranker_node)
    g.add_node("batched_verify", batched_verification_node)
    g.add_node("write", writer_node_v8)

    # -- Discovery edges --
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "discovery")
    g.add_edge("discovery", "complexity_router")

    def route_complexity_to_confidence(state: AgentState) -> str:
        """Simple queries skip straight to planner; complex ones may need clarification."""
        if state.get("_fast_path", False):
            return "planner"
        discovery = state.get("discovery") or {}
        return "clarify" if discovery.get("confidence", 0) < 0.7 else "planner"

    g.add_conditional_edges(
        "complexity_router", route_complexity_to_confidence,
        {"clarify": "clarify", "planner": "planner"},
    )
    g.add_edge("clarify", "planner")
    g.add_edge("auto_refine", "planner")

    # -- Research edges --
    if use_multi_agent:
        def route_planner_by_complexity(state: AgentState) -> str:
            """Simple queries use single worker; medium/complex use orchestrator."""
            return "fast_workers" if state.get("query_complexity", "medium") == "simple" else "orchestrator"

        g.add_conditional_edges(
            "planner", route_planner_by_complexity,
            {"fast_workers": "worker", "orchestrator": "orchestrator"},
        )
        g.add_edge("worker", "reduce")
        g.add_conditional_edges("orchestrator", fanout_subagents)
        g.add_edge("subagent", "synthesizer")
        g.add_edge("synthesizer", "early_termination")
        g.add_conditional_edges(
            "early_termination", route_after_termination_check,
            {"reduce": "reduce", "gap_detector": "gap_detector"},
        )
        g.add_conditional_edges(
            "gap_detector", route_after_gaps,
            {"orchestrator": "orchestrator", "backtrack": "backtrack", "reduce": "reduce"},
        )
        g.add_conditional_edges(
            "backtrack", route_after_backtrack, {"orchestrator": "orchestrator"},
        )
    else:
        g.add_conditional_edges("planner", fanout_workers_v8)
        g.add_edge("worker", "reduce")

    # -- Batched trust engine edges --
    g.add_conditional_edges(
        "reduce", route_after_reduce_optimized, {"batched_trust": "batched_cred_claims"},
    )
    g.add_edge("batched_cred_claims", "ranker")
    g.add_edge("ranker", "batched_verify")
    g.add_edge("batched_verify", "write")
    g.add_edge("write", END)

    # -- Compile --
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


def build_v8_optimized_graph_with_memory(interrupt_on_clarify: bool = True, use_multi_agent: bool = True):
    """Build optimized v8 graph with checkpointer for HITL."""
    checkpointer = MemorySaver()
    graph = build_v8_optimized_graph(
        checkpointer=checkpointer,
        interrupt_on_clarify=interrupt_on_clarify,
        use_multi_agent=use_multi_agent,
    )
    return graph, checkpointer


def build_v8_optimized_simple_graph(use_multi_agent: bool = True):
    """Build optimized v8 graph without HITL (automated mode)."""
    return build_v8_optimized_graph(
        checkpointer=None, interrupt_on_clarify=False, use_multi_agent=use_multi_agent,
    )
