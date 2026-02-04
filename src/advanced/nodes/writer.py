"""Writer node -- produces the final research report with confidence indicators and source citations."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.advanced.state import (
    AgentState,
    Claim,
    Evidence,
    Source,
    ResearchMetadata,
)
from src.utils.llm import create_chat_model


WRITER_SYSTEM = """You write a research report with confidence indicators.

Rules:
1. Use ONLY the verified claims provided - do NOT add any information not in the claims
2. Include confidence indicators after each statement:
   - [✓✓] = High confidence (cross-validated by multiple sources)
   - [✓] = Verified (single source, but verified)
   - [⚠] = Lower confidence (use "according to [source]" phrasing)
3. Every paragraph must include at least one source citation like [S1]
4. If information is limited, acknowledge this honestly
5. Include a "Research Quality" section at the end summarizing confidence levels

Structure:
- Title
- Executive Summary (2-3 sentences)
- Main sections following the outline
- Research Quality section
- Sources section with [S#] Title — URL

IMPORTANT: Only write what the evidence supports. If a section has no verified claims,
write "No verified information available for this section."
"""


WRITER_SIMPLE_SYSTEM = """You write a research report from the provided claims and citations.

Rules:
- Use ONLY the claims provided
- Every paragraph must include at least one citation like [S1]
- If claims are insufficient, say so explicitly
- Keep it factual and grounded

Output markdown with:
- Title
- Executive Summary
- Sections following outline
- Sources section with [S#] Title — URL
"""


def writer_node(state: AgentState) -> Dict[str, Any]:
    """Build the final markdown report, tagging every statement with a confidence indicator."""
    issues = state.get("issues") or []

    if any(i.get("level") == "block" for i in issues):
        msg = "Blocked: cannot write report.\n\nIssues:\n"
        for i in issues:
            msg += f"- {i['message']} (claims: {i.get('related_cids', [])})\n"
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    plan = state.get("plan") or {}
    topic = plan.get("topic") or "Research Report"
    outline = plan.get("outline") or []

    sources: List[Source] = state.get("sources") or []
    evidence: List[Evidence] = state.get("evidence") or []
    claims: List[Claim] = state.get("claims") or []
    citations = state.get("citations") or []

    verified_citations = state.get("verified_citations") or []
    unverified_claims = state.get("unverified_claims") or []
    cross_validated_claims = state.get("cross_validated_claims") or []
    claim_confidence = state.get("claim_confidence") or {}
    overall_confidence = state.get("overall_confidence", 0.5)
    knowledge_gaps = state.get("knowledge_gaps") or []
    source_credibility = state.get("source_credibility") or {}

    primary_anchor = state.get("primary_anchor") or ""
    anchor_terms = state.get("anchor_terms") or []
    original_query = state.get("original_query") or ""

    if not sources or not evidence or not claims:
        return _generate_no_info_report(state, topic, primary_anchor, anchor_terms, original_query)

    verified_cids = {vc["cid"] for vc in verified_citations}
    verified_claims = [c for c in claims if c["cid"] in verified_cids]

    if claims and not verified_claims:
        return _generate_insufficient_report(state, topic, primary_anchor, original_query)

    cid_to_eids = {c["cid"]: c.get("eids", []) for c in citations}
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}
    cv_cids = {vc["cid"] for vc in cross_validated_claims}

    high_conf_threshold = state.get("high_confidence_threshold", 0.8)
    med_conf_threshold = state.get("medium_confidence_threshold", 0.6)

    claim_packets = []
    for cl in verified_claims:
        cid = cl["cid"]
        conf = claim_confidence.get(cid, 0.5)

        if cid in cv_cids and conf >= high_conf_threshold:
            indicator = "✓✓"
        elif conf >= med_conf_threshold:
            indicator = "✓"
        else:
            indicator = "⚠"

        sids = []
        for eid in cid_to_eids.get(cid, []):
            sid = eid_to_sid.get(eid)
            if sid:
                sids.append(sid)
        sids = sorted(set(sids))

        claim_packets.append({
            "cid": cid,
            "section": cl.get("section", "General"),
            "text": cl["text"],
            "cite": [f"[{sid}]" for sid in sids],
            "confidence": conf,
            "indicator": indicator,
            "cross_validated": cid in cv_cids,
        })

    gaps_summary = ""
    if knowledge_gaps:
        gaps_list = []
        for gap in knowledge_gaps[:5]:
            gaps_list.append(f"- **{gap.get('section', 'General')}**: {gap.get('description', 'Information limited')}")
        gaps_summary = "\n".join(gaps_list)

    total_claims = len(claims)
    verified_count = len(verified_claims)
    cv_count = len(cross_validated_claims)
    high_conf_count = sum(1 for c in claim_confidence.values() if c >= high_conf_threshold)

    quality_metrics = {
        "overall_confidence": f"{overall_confidence:.0%}",
        "verified_claims": f"{verified_count}/{total_claims}",
        "cross_validated": f"{cv_count}",
        "high_confidence": f"{high_conf_count}",
        "sources_used": len(sources),
    }

    sources_list_items = []
    for s in sources:
        sid = s["sid"]
        cred = source_credibility.get(sid, {}).get("overall", 0.5)
        cred_indicator = "●●●" if cred >= 0.7 else ("●●○" if cred >= 0.5 else "●○○")
        sources_list_items.append(f"[{sid}] {s.get('title', 'Untitled')} — {s.get('url', '')}")

    sources_list = "\n".join(sources_list_items)

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.2)

    prompt = f"""TITLE: {topic}

OUTLINE: {outline}

CLAIMS (with confidence indicators):
{_format_claim_packets(claim_packets)}

SOURCES:
{sources_list}

RESEARCH QUALITY:
- Overall Confidence: {quality_metrics['overall_confidence']}
- Verified Claims: {quality_metrics['verified_claims']}
- Cross-Validated Claims: {quality_metrics['cross_validated']}
- Sources Used: {quality_metrics['sources_used']}

KNOWLEDGE GAPS:
{gaps_summary if gaps_summary else "No significant gaps identified."}

Write a comprehensive research report. Include the confidence indicators [✓✓], [✓], or [⚠] after statements.
Add a "Research Quality" section at the end summarizing the confidence metrics.
"""

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=prompt)
    ])

    report = (resp.content or "").strip()

    research_metadata: ResearchMetadata = {
        "overall_confidence": overall_confidence,
        "verified_claims": verified_count,
        "total_claims": total_claims,
        "knowledge_gaps": len(knowledge_gaps),
        "sources_used": len(sources),
        "research_iterations": state.get("research_iteration", 0) + 1,
        "total_searches": len(state.get("research_trajectory") or []),
        "time_elapsed_seconds": 0,  # TODO: wire up actual timing
    }

    return {
        "report": report,
        "messages": [AIMessage(content=report)],
        "research_metadata": research_metadata,
    }


def _format_claim_packets(claim_packets: List[Dict]) -> str:
    """Format claim packets into a readable block for the LLM prompt."""
    lines = []
    for cp in claim_packets:
        cite_str = " ".join(cp.get("cite", []))
        lines.append(
            f"[{cp['indicator']}] ({cp['section']}) {cp['text']} {cite_str}"
        )
    return "\n".join(lines)


def _generate_no_info_report(
    state: AgentState,
    topic: str,
    primary_anchor: str,
    anchor_terms: List[str],
    original_query: str
) -> Dict[str, Any]:
    """Fallback report when we found nothing usable."""
    entity_desc = primary_anchor or original_query
    if anchor_terms:
        context = ", ".join(anchor_terms[:3])
        entity_desc = f"{primary_anchor} ({context})"

    report = f"""# Research Report: {topic}

## No Verified Information Found

I was unable to find relevant, verifiable information about **{entity_desc}**.

### Possible Reasons

1. **Limited online presence**: The subject may not have a significant public digital footprint
2. **Privacy**: Information may not be publicly available
3. **Name disambiguation**: Search results about different entities with similar names were filtered out to ensure accuracy

### Search Parameters

- **Primary entity**: {primary_anchor or "Not specified"}
- **Context**: {", ".join(anchor_terms) if anchor_terms else "None provided"}
- **Original query**: {original_query}

### Recommendations

- Provide more specific identifying information (full name, institution, role, field of work)
- Check if the person/entity has public profiles (LinkedIn, personal website, institutional page)
- Consider that some information may simply not be publicly available

---

## Research Quality

| Metric | Value |
|--------|-------|
| Overall Confidence | 0% |
| Verified Claims | 0 |
| Sources Found | 0 |

---

*No sources were used as no verified information was found.*
"""

    return {"report": report, "messages": [AIMessage(content=report)]}


def _generate_insufficient_report(
    state: AgentState,
    topic: str,
    primary_anchor: str,
    original_query: str
) -> Dict[str, Any]:
    """Fallback report when we have sources but couldn't verify any claims."""
    report = f"""# Research Report: {topic}

## Insufficient Verified Information

While some search results were found, I could not verify that they accurately describe **{primary_anchor or original_query}**.

### Why This Happened

To maintain accuracy and avoid providing incorrect information, claims must be:
1. Directly supported by source text (span verification)
2. Not conflating information about different entities

The available evidence did not meet these verification standards.

### Recommendations

- Provide more specific identifying information
- Confirm the full name and current affiliation
- Check for official public profiles

---

## Research Quality

| Metric | Value |
|--------|-------|
| Overall Confidence | Low |
| Verified Claims | 0 |
| Reason | Claims could not be verified against source text |

---

*No verified sources could be cited in this report.*
"""

    return {"report": report, "messages": [AIMessage(content=report)]}
