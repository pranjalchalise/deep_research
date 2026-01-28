from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.core.state import AgentState
from src.nodes.planner import planner_node
from src.nodes.worker import worker_node
from src.nodes.reducer import reducer_node
from src.nodes.claims import claims_node
from src.nodes.cite import cite_node
from src.nodes.verify import verify_node
from src.nodes.writer import writer_node


def fanout_workers(state: AgentState):
    plan = state.get("plan") or {}
    queries = plan.get("queries") or []
    sends = []
    for qi in queries:
        sends.append(Send("worker", {"query_item": qi}))
    return sends


def route_after_reduce(state: AgentState) -> str:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "claims" if done >= total else END


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)
    g.add_node("claims", claims_node)
    g.add_node("cite", cite_node)
    g.add_node("verify", verify_node)
    g.add_node("write", writer_node)

    g.add_edge(START, "planner")

    # Planner fans out to workers
    g.add_conditional_edges("planner", fanout_workers)

    # Each worker triggers reduce
    g.add_edge("worker", "reduce")

    # Reduce runs multiple times; only final time continues
    g.add_conditional_edges("reduce", route_after_reduce)

    g.add_edge("claims", "cite")
    g.add_edge("cite", "verify")
    g.add_edge("verify", "write")
    g.add_edge("write", END)

    return g.compile()
