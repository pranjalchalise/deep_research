from __future__ import annotations

from typing import Any, Dict, List

from src.advanced.state import AgentState
from src.utils.ids import dedup_sources, assign_source_ids, assign_evidence_ids


def reducer_node(state: AgentState) -> Dict[str, Any]:
    """Waits for all workers to finish, then normalizes sources and evidence."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0

    # Workers still running -- nothing to do yet
    if done < total:
        return {}

    raw_sources = state.get("raw_sources") or []
    raw_evidence = state.get("raw_evidence") or []

    deduped = dedup_sources(raw_sources)
    sources = assign_source_ids(deduped)

    url_to_sid = {s["url"]: s["sid"] for s in sources}
    evidence = assign_evidence_ids(raw_evidence, url_to_sid)

    return {"sources": sources, "evidence": evidence}
