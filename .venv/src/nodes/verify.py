from __future__ import annotations

from typing import Any, Dict, List

from src.core.state import AgentState, Issue

def verify_node(state: AgentState) -> Dict[str, Any]:
    claims = state.get("claims") or []
    citations = state.get("citations") or []

    cite_map = {c["cid"]: c.get("eids", []) for c in citations}

    issues: List[Issue] = []
    blocked: List[str] = []
    for cl in claims:
        cid = cl["cid"]
        if not cite_map.get(cid):
            blocked.append(cid)

    if blocked:
        issues.append({
            "level": "block",
            "message": f"{len(blocked)} claims have no evidence citations. Refusing to write report.",
            "related_cids": blocked,
        })

    return {"issues": issues}
