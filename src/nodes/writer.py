# src/nodes/writer.py
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.core.state import AgentState, Claim, Evidence, Source
from src.utils.llm import create_chat_model


WRITER_SYSTEM = """You write a research report ONLY from the provided claims and citations.

Rules:
- Use ONLY the claims provided.
- Every paragraph must include at least one citation like [S1].
- Citations must correspond to the sources provided.
- If the provided claims are insufficient, say so explicitly and keep the report short.


Output markdown with:
- Title
- Executive Summary
- Sections following outline
- Sources section with [S#] Title — URL
"""


def writer_node(state: AgentState) -> Dict[str, Any]:
    issues = state.get("issues") or []
    if any(i.get("level") == "block" for i in issues):
        msg = "Blocked: cannot write report.\n\nIssues:\n"
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

    # Get entity info for "no results" message
    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []
    original_query = state.get("original_query") or ""

    # ---- Handle "no relevant information found" ----
    if not sources or not evidence or not claims:
        entity_desc = primary_anchor or original_query

        if anchor_terms:
            context = ", ".join(anchor_terms[:3])
            entity_desc = f"{primary_anchor} ({context})"

        no_info_report = f"""# Research Report: {topic}

## No Information Found

I was unable to find relevant, verifiable information about **{entity_desc}**.

This could mean:
1. **Limited online presence**: The person/entity may not have a significant public digital footprint
2. **Privacy**: Information about this person may not be publicly available
3. **Name ambiguity**: Search results returned information about different people/entities with similar names, which were filtered out to avoid providing incorrect information

### What was searched:
- Primary entity: {primary_anchor or "Not specified"}
- Context: {", ".join(anchor_terms) if anchor_terms else "None provided"}
- Original query: {original_query}

### Suggestions:
- Try providing more specific identifying information (full name, institution, role, field)
- Check if the person has a LinkedIn, personal website, or institutional profile
- Consider that the information you're looking for may not be publicly available

---
*No sources were used in this report as no relevant information was found.*
"""
        return {"report": no_info_report, "messages": [AIMessage(content=no_info_report)]}

    # ---- Normal report generation ----

    # CID -> EIDs (from citations)
    cid_to_eids = {c["cid"]: c.get("eids", []) for c in citations}

    # EID -> SID
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}

    # Build claim packets writer can use (each claim has allowed SIDs)
    claim_packets = []
    for cl in claims:
        sids = []
        for eid in cid_to_eids.get(cl["cid"], []):
            sid = eid_to_sid.get(eid)
            if sid:
                sids.append(sid)
        sids = sorted(set(sids))

        claim_packets.append({
            "cid": cl["cid"],
            "section": cl["section"],
            "text": cl["text"],
            "cite": [f"[{sid}]" for sid in sids],
        })

    # Check if we have claims with actual citations
    claims_with_citations = [cp for cp in claim_packets if cp["cite"]]
    if not claims_with_citations:
        # We have claims but no citations - likely all filtered out as irrelevant
        no_cite_report = f"""# Research Report: {topic}

## Insufficient Verified Information

While some search results were found, I could not verify that they are about the correct entity: **{primary_anchor or original_query}**.

To avoid providing potentially incorrect information about a different person/entity with a similar name, I am unable to generate a detailed report.

### Suggestions:
- Provide more specific identifying information
- Confirm the person's full name and current affiliation
- Check if they have a public profile (LinkedIn, institutional page, etc.)

---
*No verified sources could be used for this report.*
"""
        return {"report": no_cite_report, "messages": [AIMessage(content=no_cite_report)]}

    sources_list = "\n".join([f"[{s['sid']}] {s['title']} — {s['url']}" for s in sources])

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"TITLE: {topic}\n\n"
            f"OUTLINE:\n{outline}\n\n"
            f"CLAIMS (each includes allowed citations):\n{claim_packets}\n\n"
            f"SOURCES:\n{sources_list}\n"
        ))
    ])

    report = (resp.content or "").strip()
    return {"report": report, "messages": [AIMessage(content=report)]}
