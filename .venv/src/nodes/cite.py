from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, ClaimCitations
from src.utils.format import evidence_block

CITE_SYSTEM = """You attach evidence to claims.

Input: claims and evidence items.
Output: ONLY JSON list:
{"cid":"C1","eids":["E2","E5"]}

Rules:
- Every claim must have at least 1 evidence id.
- Choose the most directly supporting evidence.
- Only use evidence ids that exist.
"""

def cite_node(state: AgentState) -> Dict[str, Any]:
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    ev_block = evidence_block(evidence)

    resp = llm.invoke([
        SystemMessage(content=CITE_SYSTEM),
        HumanMessage(content=f"CLAIMS:\n{json.dumps(claims, ensure_ascii=False)}\n\nEVIDENCE:\n{ev_block}")
    ])

    txt = resp.content.strip()
    cites: List[ClaimCitations] = []
    try:
        cites = json.loads(txt)
    except Exception:
        cites = []

    # basic cleanup
    valid_eids = {e["eid"] for e in evidence}
    cleaned: List[ClaimCitations] = []
    for item in cites:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cid", "")).strip()
        eids = item.get("eids", [])
        if not cid or not isinstance(eids, list):
            continue
        eids2 = [str(e).strip() for e in eids if str(e).strip() in valid_eids]
        cleaned.append({"cid": cid, "eids": eids2})

    return {"citations": cleaned}
