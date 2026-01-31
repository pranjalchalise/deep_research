# src/nodes/cite.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, ClaimCitations
from src.utils.format import evidence_block
from src.utils.json_utils import parse_json_array
from src.utils.llm import create_chat_model

CITE_SYSTEM = """You attach evidence to claims.

Input: claims and evidence items.
Output: ONLY valid JSON list:
{"cid":"C1","eids":["E2","E5"]}

Rules:
- Every claim must have at least 1 evidence id IF any evidence exists.
- Choose the most directly supporting evidence.
- Only use evidence ids that exist.
- Do NOT include brackets around IDs (use E1 not [E1]).
"""


def _normalize_eids(raw: Any) -> List[str]:
    """
    Accepts eids as:
    - ["E1","E2"]
    - "E1, E2"
    - "[E1]"
    - ["[E1]", "E2"]
    Returns cleaned tokens (still need validation).
    """
    out: List[str] = []

    def add_token(t: str):
        t = t.strip()
        if not t:
            return
        # strip brackets/quotes/punctuation around
        t = re.sub(r"^[^A-Za-z0-9]+", "", t)
        t = re.sub(r"[^A-Za-z0-9]+$", "", t)
        if not t:
            return
        out.append(t)

    if raw is None:
        return out

    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, str):
                # sometimes "E1, E2"
                parts = re.split(r"[,\s]+", x.strip())
                for p in parts:
                    add_token(p)
            else:
                add_token(str(x))
        return out

    if isinstance(raw, str):
        parts = re.split(r"[,\s]+", raw.strip())
        for p in parts:
            add_token(p)
        return out

    add_token(str(raw))
    return out


def cite_node(state: AgentState) -> Dict[str, Any]:
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []

    # If no evidence exists, return empty citations (verifier will decide)
    if not evidence:
        return {"citations": []}

    valid_eids = {e["eid"] for e in evidence}
    ev_block = evidence_block(evidence)

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.0)

    resp = llm.invoke(
        [
            SystemMessage(content=CITE_SYSTEM),
            HumanMessage(
                content=(
                    f"CLAIMS:\n{json.dumps(claims, ensure_ascii=False)}\n\n"
                    f"EVIDENCE:\n{ev_block}\n\n"
                    "Return JSON only."
                )
            ),
        ]
    )

    txt = resp.content.strip()
    cites: List[ClaimCitations] = parse_json_array(txt, default=[])

    # Build a map cid -> cleaned valid eids
    cid_to_eids: Dict[str, List[str]] = {}
    for item in cites:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cid", "")).strip()
        if not cid:
            continue
        raw_eids = _normalize_eids(item.get("eids", []))
        eids = [e for e in raw_eids if e in valid_eids]
        cid_to_eids[cid] = list(dict.fromkeys(eids))  # dedup preserve order

    # Fallback heuristic: if model failed to cite for a claim, attach best evidence by section
    # We prefer evidence with same section; else first evidence.
    section_to_eids: Dict[str, List[str]] = {}
    for e in evidence:
        sec = (e.get("section") or "General").strip()
        section_to_eids.setdefault(sec, []).append(e["eid"])

    cleaned: List[ClaimCitations] = []
    for cl in claims:
        cid = cl["cid"]
        sec = (cl.get("section") or "General").strip()
        eids = cid_to_eids.get(cid, [])

        if not eids:
            # try same-section evidence
            eids = section_to_eids.get(sec, [])[:2]
        if not eids:
            # last resort: pick any evidence
            eids = [next(iter(valid_eids))]

        cleaned.append({"cid": cid, "eids": eids})

    return {"citations": cleaned}
