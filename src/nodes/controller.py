from __future__ import annotations

from typing import Any, Dict
from langchain_core.messages import AIMessage

from src.core.state import AgentState

def controller_node(state: AgentState) -> Dict[str, Any]:
    # No-op node: routing happens in conditional edges.
    # But we can emit a tiny debug message if you want.
    return {}

def route_from_controller(state: AgentState) -> str:
    # First time: go plan
    if state.get("plan") is None:
        return "planner"

    # After assess: decide to iterate or write
    assess = state.get("last_assess") or {}
    coverage_ok = bool(assess.get("coverage_ok", False))
    issues = state.get("issues") or []
    blocked = any(i.get("level") == "block" for i in issues)

    r = int(state.get("round") or 0)
    maxr = int(state.get("max_rounds") or 1)

    if blocked:
        return "writer"  # will output refusal + issues
    if (coverage_ok and not blocked) or (r >= maxr):
        return "writer"
    return "planner"  # iterate
