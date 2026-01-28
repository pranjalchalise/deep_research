from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Claim
from src.utils.format import evidence_block

CLAIMS_SYSTEM = """You are generating atomic claims for a report.

Using ONLY the evidence items provided, produce 8-20 atomic claims grouped by section.

Return ONLY JSON list:
{"cid":"C1","section":"...","text":"..."} ...

Rules:
- Each claim must be directly supported by one or more evidence items.
- Keep claims factual, specific, and short (one sentence).
- Do not invent facts not present in evidence.
"""

def claims_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []
    evidence = state.get("evidence") or []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    ev_block = evidence_block(evidence)

    resp = llm.invoke([
        SystemMessage(content=CLAIMS_SYSTEM),
        HumanMessage(content=f"Outline sections:\n{outline}\n\nEvidence:\n{ev_block}")
    ])

    txt = resp.content.strip()
    claims: List[Claim] = []
    try:
        claims = json.loads(txt)
    except Exception:
        claims = []

    # fallback if empty
    cleaned: List[Claim] = []
    i = 1
    for c in claims:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        section = str(c.get("section", "")).strip() or "General"
        if text:
            cleaned.append({"cid": f"C{i}", "section": section, "text": text})
            i += 1

    return {"claims": cleaned}
