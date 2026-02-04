"""
LangGraph StateGraph wiring for the research pipeline.

One graph handles both research modes, chosen at runtime by the
"mode" field in state:

  - "single": iterative search -> gap detection -> loop or verify -> write
  - "multi":  orchestrate -> parallel workers (Send) -> collect -> synthesize
              -> loop or verify -> write

Both paths share understanding, clarification, planning, verification,
and report writing.

Quick start (no HITL):

    from src.pipeline import build_graph
    graph = build_graph(interrupt_on_clarify=False)
    result = graph.invoke(
        {"query": "What is quantum computing?", "mode": "single",
         "iteration": 0, "evidence": [], "worker_results": [],
         "done_workers": 0, "searches_done": [], "sources": {}},
        config={"configurable": {"model": "gpt-4o", "fast_model": "gpt-4o-mini"}},
    )
    print(result["report"])
"""
from __future__ import annotations

from typing import List

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline.state import ResearchState, Configuration
from src.pipeline.nodes import (
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


# ---------------------------------------------------------------------------
# Routing functions -- each returns the name of the next node.
# ---------------------------------------------------------------------------

def route_after_understand(state: ResearchState) -> str:
    """Ambiguous query -> ask user. Clear query -> start planning."""
    return "clarify" if state.get("is_ambiguous") else "plan"


def route_after_plan(state: ResearchState) -> str:
    """Pick single-agent or multi-agent path based on the mode field."""
    return "orchestrate" if state.get("mode") == "multi" else "search_and_extract"


def fanout_workers(state: ResearchState) -> List[Send]:
    """Spawn one parallel worker per sub-question via Send()."""
    return [
        Send("search_worker", {
            "sub_question": sq,
            "query": state["query"],
            "user_clarification": state.get("user_clarification", ""),
        })
        for sq in state.get("sub_questions", [])
    ]


def route_after_gaps(state: ResearchState) -> str:
    """Decide if the single-agent loop should keep going or move to writing.

    We stop when any of these is true:
    - gap detector says we're ready
    - coverage is above the threshold (default 70%)
    - no more searches to run
    - hit the iteration limit
    """
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
    """Decide if multi-agent needs another round or can move to writing.

    We stop when synthesis says we're good, we've used up iterations,
    or there are no follow-up searches to run."""
    syn = state.get("synthesis", {})
    if not syn.get("needs_more_research"):
        return "verify"
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        return "verify"
    if not syn.get("additional_searches"):
        return "verify"
    return "orchestrate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None, interrupt_on_clarify: bool = True):
    """Assemble and compile the research graph.

    Pass a checkpointer (e.g. MemorySaver) if you need state persistence
    or HITL support. For simple fire-and-forget runs, leave it as None.
    Set interrupt_on_clarify=False to skip the HITL pause entirely.
    """
    g = StateGraph(ResearchState, config_schema=Configuration)

    # -- Register all nodes --
    g.add_node("understand", understand_node)
    g.add_node("clarify", clarify_node)
    g.add_node("plan", plan_node)

    g.add_node("orchestrate", orchestrate_node)       # multi-agent
    g.add_node("search_worker", search_worker_node)   # multi-agent (parallel via Send)
    g.add_node("collect", collect_node)                # multi-agent
    g.add_node("synthesize", synthesize_node)          # multi-agent

    g.add_node("search_and_extract", search_and_extract_node)  # single-agent
    g.add_node("detect_gaps", detect_gaps_node)                # single-agent

    g.add_node("verify", verify_node)
    g.add_node("write_report", write_report_node)

    # -- Understanding & clarification --
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_after_understand,
                            {"clarify": "clarify", "plan": "plan"})
    g.add_edge("clarify", "plan")

    # -- Planning splits into single vs multi --
    g.add_conditional_edges("plan", route_after_plan,
                            {"search_and_extract": "search_and_extract",
                             "orchestrate": "orchestrate"})

    # -- Single-agent loop: search -> gaps -> maybe loop --
    g.add_edge("search_and_extract", "detect_gaps")
    g.add_conditional_edges("detect_gaps", route_after_gaps,
                            {"search_and_extract": "search_and_extract",
                             "verify": "verify"})

    # -- Multi-agent loop: orchestrate -> parallel workers -> collect -> synthesize --
    g.add_conditional_edges("orchestrate", fanout_workers)
    g.add_edge("search_worker", "collect")
    g.add_edge("collect", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesis,
                            {"orchestrate": "orchestrate", "verify": "verify"})

    # -- Both paths converge here --
    g.add_edge("verify", "write_report")
    g.add_edge("write_report", END)

    # -- Compile with optional checkpointer and interrupt --
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)


# Exposed at module level for LangGraph Studio / langgraph.json
graph = build_graph(checkpointer=None, interrupt_on_clarify=True)
