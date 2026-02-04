from __future__ import annotations

from typing import Any, Dict, List

from src.core.state import AgentState
from src.utils.ids import dedup_sources, assign_source_ids, assign_evidence_ids

def reducer_node(state: AgentState) -> Dict[str, Any]:
    """
    Called after each worker. Only on the FINAL worker do we normalize and proceed.
    """
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0

    # Not ready yet → no-op
    if done < total:
        return {}

    raw_sources = state.get("raw_sources") or []
    raw_evidence = state.get("raw_evidence") or []

    deduped = dedup_sources(raw_sources)
    sources = assign_source_ids(deduped)

    url_to_sid = {s["url"]: s["sid"] for s in sources}
    evidence = assign_evidence_ids(raw_evidence, url_to_sid)

    return {"sources": sources, "evidence": evidence}
