from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from src.advanced.state import AgentState, Claim
from src.utils.format import evidence_block
from src.utils.json_utils import parse_json_array
from src.utils.llm import create_chat_model

CLAIMS_SYSTEM = """You generate atomic claims grounded ONLY in evidence.

Return ONLY JSON list:
{"cid":"C1","section":"...","text":"..."} ...

Rules:
- 8-24 claims total.
- Each claim must be one sentence and directly supported by evidence.
- Avoid vague wording; prefer specific statements.
"""


def claims_node(state: AgentState) -> Dict[str, Any]:
    """Turn evidence + outline into a flat list of atomic, citable claims."""
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []
    evidence = state.get("evidence") or []

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)
    ev = evidence_block(evidence)

    resp = llm.invoke(
        [
            SystemMessage(content=CLAIMS_SYSTEM),
            HumanMessage(content=f"Outline:\n{outline}\n\nEvidence:\n{ev}"),
        ]
    )

    txt = resp.content.strip()
    claims_raw: List[dict] = parse_json_array(txt, default=[])

    out: List[Claim] = []
    i = 1
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        section = str(c.get("section", "")).strip() or "General"
        if text:
            out.append({"cid": f"C{i}", "section": section, "text": text})
            i += 1

    return {"claims": out}
