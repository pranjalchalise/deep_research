# src/core/graph_v9.py
"""
Research Studio V9 - LangGraph Workflow

This is the main graph definition that implements the V9 architecture:

1. QUERY ANALYZER → Semantic understanding (no patterns)
2. AMBIGUITY ROUTER → Clarify or proceed
3. CLARIFY (HITL) → Human clarification with enriched context
4. PLANNER → LLM-generated research plan
5. ORCHESTRATOR → Assign questions to parallel subagents
6. SUBAGENTS → Parallel research execution
7. SYNTHESIZER → Aggregate findings
8. GAP DETECTOR → Check coverage, decide: loop or continue
9. TRUST ENGINE → Credibility, claims, verification
10. WRITER → Generate final report

Key Features:
- Human-in-the-loop with interrupt before clarify node
- Iterative research loop (gap detection → orchestrator)
- Parallel subagent execution via Send()
- Batched trust engine for efficiency
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from src.core.state import AgentState
from src.core.config import V8Config


# ============================================================================
# IMPORT NODES
# ============================================================================

# V9 Nodes (new)
from src.nodes.analyzer import query_analyzer_node, route_after_analysis
from src.nodes.clarify import clarify_node
from src.nodes.planner_v9 import planner_node
from src.nodes.gap_detector import gap_detector_node, route_after_gap_detection

# Existing nodes (reused)
from src.nodes.orchestrator import (
    orchestrator_node,
    subagent_node,
    synthesizer_node,
)
from src.nodes.reducer import reducer_node
from src.nodes.trust_engine_batched import full_trust_engine_batched
from src.nodes.writer import writer_node


# ============================================================================
# FANOUT FUNCTION FOR PARALLEL SUBAGENTS
# ============================================================================

def fanout_subagents(state: AgentState) -> List[Send]:
    """
    Fan out to parallel subagents based on assignments.

    Uses LangGraph's Send() to create parallel execution branches.
    Each subagent runs independently and results are aggregated.
    """
    assignments = state.get("subagent_assignments") or []

    if not assignments:
        # No assignments - skip to synthesizer
        return [Send("synthesizer", state)]

    # Create a Send for each subagent
    sends = []
    for assignment in assignments:
        # Each subagent gets the full state plus its specific assignment
        sends.append(Send("subagent", {
            **state,
            "subagent_assignment": assignment,
        }))

    return sends


# ============================================================================
# INCREMENT ITERATION NODE
# ============================================================================

def increment_iteration_node(state: AgentState) -> Dict[str, Any]:
    """
    Increment the research iteration counter when looping back.

    This node runs when gap detection decides to continue researching.
    """
    current = state.get("research_iteration", 0)
    return {
        "research_iteration": current + 1,
    }


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_v9_graph(
    checkpointer: Optional[Any] = None,
    interrupt_on_clarify: bool = True,
) -> StateGraph:
    """
    Build the V9 research graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence
        interrupt_on_clarify: Whether to interrupt for human clarification

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # ─────────────────────────────────────────────────────────
    # ADD NODES
    # ─────────────────────────────────────────────────────────

    # Phase 1: Query Understanding
    graph.add_node("query_analyzer", query_analyzer_node)

    # Phase 2: Clarification (HITL)
    graph.add_node("clarify", clarify_node)

    # Phase 3: Planning
    graph.add_node("planner", planner_node)

    # Phase 4: Research Loop
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("subagent", subagent_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reducer", reducer_node)  # Normalizes raw sources/evidence
    graph.add_node("gap_detector", gap_detector_node)
    graph.add_node("increment_iteration", increment_iteration_node)

    # Phase 5: Trust & Verification
    graph.add_node("trust_engine", full_trust_engine_batched)

    # Phase 6: Report Generation
    graph.add_node("writer", writer_node)

    # ─────────────────────────────────────────────────────────
    # ADD EDGES
    # ─────────────────────────────────────────────────────────

    # Entry point
    graph.set_entry_point("query_analyzer")

    # Query Analyzer → Conditional: Clarify or Plan
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {
            "clarify": "clarify",
            "planner": "planner",
        }
    )

    # Clarify → Planner (always proceeds after clarification)
    graph.add_edge("clarify", "planner")

    # Planner → Orchestrator
    graph.add_edge("planner", "orchestrator")

    # Orchestrator → Parallel Subagents (fan-out)
    graph.add_conditional_edges(
        "orchestrator",
        fanout_subagents,
    )

    # Subagents → Synthesizer (fan-in)
    graph.add_edge("subagent", "synthesizer")

    # Synthesizer → Reducer (normalize raw sources/evidence)
    graph.add_edge("synthesizer", "reducer")

    # Reducer → Gap Detector
    graph.add_edge("reducer", "gap_detector")

    # Gap Detector → Conditional: Loop or Continue
    graph.add_conditional_edges(
        "gap_detector",
        route_after_gap_detection,
        {
            "orchestrator": "increment_iteration",  # Loop back via increment
            "trust_engine": "trust_engine",         # Continue to verification
        }
    )

    # Increment Iteration → Orchestrator (completes the loop)
    graph.add_edge("increment_iteration", "orchestrator")

    # Trust Engine → Writer
    graph.add_edge("trust_engine", "writer")

    # Writer → END
    graph.add_edge("writer", END)

    # ─────────────────────────────────────────────────────────
    # COMPILE
    # ─────────────────────────────────────────────────────────

    compile_kwargs = {}

    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return graph.compile(**compile_kwargs)


def build_v9_graph_with_memory(
    interrupt_on_clarify: bool = True,
) -> StateGraph:
    """
    Build the V9 graph with in-memory checkpointing.

    This is the recommended way to create the graph for most use cases.
    """
    checkpointer = MemorySaver()
    return build_v9_graph(
        checkpointer=checkpointer,
        interrupt_on_clarify=interrupt_on_clarify,
    )


# ============================================================================
# SIMPLE GRAPH (NO MULTI-AGENT)
# ============================================================================

def build_v9_simple_graph(
    checkpointer: Optional[Any] = None,
    interrupt_on_clarify: bool = True,
) -> StateGraph:
    """
    Build a simplified V9 graph without multi-agent orchestration.

    Uses direct search instead of parallel subagents.
    Good for simpler queries or testing.
    """
    from src.nodes.search_worker import worker_node
    from src.nodes.ranker import ranker_node
    from src.nodes.reducer import reducer_node

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("planner", planner_node)
    graph.add_node("search_worker", worker_node)
    graph.add_node("reducer", reducer_node)
    graph.add_node("ranker", ranker_node)
    graph.add_node("trust_engine", full_trust_engine_batched)
    graph.add_node("writer", writer_node)

    # Entry
    graph.set_entry_point("query_analyzer")

    # Edges
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {
            "clarify": "clarify",
            "planner": "planner",
        }
    )

    graph.add_edge("clarify", "planner")
    graph.add_edge("planner", "search_worker")
    graph.add_edge("search_worker", "reducer")
    graph.add_edge("reducer", "ranker")
    graph.add_edge("ranker", "trust_engine")
    graph.add_edge("trust_engine", "writer")
    graph.add_edge("writer", END)

    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return graph.compile(**compile_kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_research(
    query: str,
    config: Optional[Dict[str, Any]] = None,
    use_multi_agent: bool = True,
    interrupt_on_clarify: bool = True,
) -> Dict[str, Any]:
    """
    Run a research query through the V9 pipeline.

    Args:
        query: The user's research query
        config: Optional configuration overrides
        use_multi_agent: Whether to use multi-agent orchestration
        interrupt_on_clarify: Whether to interrupt for clarification

    Returns:
        Final state dict with report and metadata
    """
    from langchain_core.messages import HumanMessage

    # Build appropriate graph
    if use_multi_agent:
        graph = build_v9_graph_with_memory(interrupt_on_clarify=interrupt_on_clarify)
    else:
        checkpointer = MemorySaver()
        graph = build_v9_simple_graph(
            checkpointer=checkpointer,
            interrupt_on_clarify=interrupt_on_clarify,
        )

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "use_cache": True,
        "cache_dir": ".cache_v9",
    }

    # Add config overrides
    if config:
        initial_state.update(config)

    # Run the graph
    thread_config = {"configurable": {"thread_id": "research-1"}}

    result = graph.invoke(initial_state, config=thread_config)

    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "build_v9_graph",
    "build_v9_graph_with_memory",
    "build_v9_simple_graph",
    "run_research",
]
