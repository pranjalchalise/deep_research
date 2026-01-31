# src/nodes/verify.py
from __future__ import annotations

from typing import Any, Dict, List

from src.core.state import AgentState, Issue


def verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify that claims have supporting evidence.

    Note: "No evidence found" is NOT a blocking error - the writer will
    handle this case gracefully with a "no information found" message.
    Only blocks when there's evidence of actual problems (not absence of data).
    """
    claims = state.get("claims") or []
    citations = state.get("citations") or []
    evidence = state.get("evidence") or []

    issues: List[Issue] = []

    # No evidence is not a "block" - writer will handle this gracefully
    # by showing "no information found" message
    if not evidence:
        issues.append(
            {
                "level": "warn",  # Changed from "block" to "warn"
                "message": "No evidence items were found for this entity.",
                "related_cids": [],
            }
        )
        return {"issues": issues}

    cite_map = {c["cid"]: (c.get("eids") or []) for c in citations}

    missing: List[str] = []
    has_any = 0
    for cl in claims:
        cid = cl["cid"]
        eids = cite_map.get(cid, [])
        if not eids:
            missing.append(cid)
        else:
            has_any += 1

    # No citations is also not a "block" - writer will handle gracefully
    if has_any == 0 and claims:
        issues.append(
            {
                "level": "warn",  # Changed from "block" to "warn"
                "message": f"No claims could be verified with citations.",
                "related_cids": [c["cid"] for c in claims],
            }
        )
        return {"issues": issues}

    if missing:
        issues.append(
            {
                "level": "warn",
                "message": f"{len(missing)} claims have no citations and will be dropped from the final report.",
                "related_cids": missing,
            }
        )

    return {"issues": issues}
