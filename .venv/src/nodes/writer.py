from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.core.state import AgentState, Claim, Evidence, Source

WRITER_SYSTEM = """You write a research report ONLY from the provided claims and citations.

Rules:
- Every paragraph must include at least one citation like [S1].
- Citations must correspond to the sources actually provided.
- If verifier issues include a 'block', output a short refusal and list missing-claim IDs.

Output markdown with:
- Title
- Executive Summary
- Sections following outline
- Sources section with [S#] Title - URL
"""

def writer_node(state: AgentState) -> Dict[str, Any]:
    issues = state.get("issues") or []
    if any(i.get("level") == "block" for i in issues):
        msg = "Blocked: cannot write report because some claims have no evidence.\n\nIssues:\n"
        for i in issues:
            msg += f"- {i['message']} (claims: {i['related_cids']})\n"
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    plan = state.get("plan") or {}
    topic = plan.get("topic") or "Research Report"
    outline = plan.get("outline") or []

    sources: List[Source] = state.get("sources") or []
    evidence: List[Evidence] = state.get("evidence") or []
    claims: List[Claim] = state.get("claims") or []
    citations = state.get("citations") or []

    # Build helper maps for the prompt
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}
    cid_to_sids = {}
    for c in citations:
        sids = []
        for eid in c.get("eids", []):
            sid = eid_to_sid.get(eid)
            if sid:
                sids.append(sid)
        cid_to_sids[c["cid"]] = sorted(set(sids))

    # Compact “claim packets” writer can use
    claim_packets = []
    for cl in claims:
        c_sids = cid_to_sids.get(cl["cid"], [])
        claim_packets.append({
            "cid": cl["cid"],
            "section": cl["section"],
            "text": cl["text"],
            "cite": [f"[{sid}]" for sid in c_sids],
        })

    sources_list = "\n".join([f"[{s['sid']}] {s['title']} — {s['url']}" for s in sources])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"TITLE: {topic}\n\n"
            f"OUTLINE:\n{outline}\n\n"
            f"CLAIMS (each includes allowed citations):\n{claim_packets}\n\n"
            f"SOURCES:\n{sources_list}\n"
        ))
    ])

    report = resp.content.strip()
    return {"report": report, "messages": [AIMessage(content=report)]}
