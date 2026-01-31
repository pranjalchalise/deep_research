from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Claim


SYSTEM = """You are a careful research analyst.

You will be given:
- an outline (section names)
- a list of evidence items with ids (E#), each with section + text + URL + title

Your job:
Produce 8-20 ATOMIC claims and attach the best supporting evidence ids.

Return ONLY valid JSON array:
[
  {"cid":"C1","section":"...","text":"one factual sentence","eids":["E1","E3"]},
  ...
]

Rules:
- Use ONLY the evidence provided.
- Every claim MUST have 1-3 evidence ids.
- Evidence ids must exist in the provided evidence list.
- Claims must be short, factual, and specific (one sentence).
- Prefer primary/official sources when possible.
- If evidence is insufficient, produce fewer claims rather than guessing.
"""


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return toks


def _fallback_assign_eids(claim_text: str, evidence: List[Dict[str, Any]], max_eids: int = 2) -> List[str]:
    """
    Deterministic backup citer: pick evidence with highest token overlap.
    """
    ctoks = set(_tokenize(claim_text))
    if not ctoks:
        return []

    scored = []
    for e in evidence:
        etoks = set(_tokenize((e.get("text") or "") + " " + (e.get("title") or "")))
        overlap = len(ctoks & etoks)
        if overlap > 0:
            scored.append((overlap, e.get("eid")))
    scored.sort(reverse=True, key=lambda x: x[0])

    out = []
    for _, eid in scored[:max_eids]:
        if eid:
            out.append(str(eid))
    return out


def claims_and_cite_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []
    evidence = state.get("evidence") or []

    # If we have no evidence, we cannot proceed.
    if not evidence:
        return {"claims": []}

    # Build an evidence block where EIDs are explicit.
    ev_lines = []
    for e in evidence:
        ev_lines.append(
            f"{e['eid']} | section={e.get('section','General')} | sid={e.get('sid','')} | "
            f"title={e.get('title','')} | url={e.get('url','')}\n"
            f"text={str(e.get('text',''))[:700]}"
        )
    ev_block = "\n\n".join(ev_lines)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    resp = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"OUTLINE:\n{outline}\n\nEVIDENCE:\n{ev_block}")
    ])

    txt = (resp.content or "").strip()

    # Parse JSON, with fallback to outermost array.
    parsed: List[Any] = []
    try:
        parsed = json.loads(txt)
    except Exception:
        start = txt.find("[")
        end = txt.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(txt[start:end + 1])
            except Exception:
                parsed = []

    valid_eids = {e["eid"] for e in evidence if "eid" in e}

    cleaned: List[Claim] = []
    cid_counter = 1

    for item in parsed:
        if not isinstance(item, dict):
            continue

        section = str(item.get("section", "")).strip() or "General"
        text = str(item.get("text", "")).strip()
        eids = item.get("eids", [])

        if not text:
            continue

        # normalize eids
        if not isinstance(eids, list):
            eids = []
        eids = [str(x).strip() for x in eids if str(x).strip() in valid_eids]

        # fallback if LLM forgot eids
        if not eids:
            eids = _fallback_assign_eids(text, evidence, max_eids=2)

        cleaned.append({
            "cid": f"C{cid_counter}",
            "section": section,
            "text": text,
            "eids": eids,
        })
        cid_counter += 1

        if cid_counter > 20:
            break

    return {"claims": cleaned}
