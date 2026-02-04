"""Batched trust engine that combines multiple LLM calls to reduce API costs (~60% savings)."""
from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib.parse import urlparse

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import (
    AgentState,
    SourceCredibility,
    VerifiedCitation,
)
from src.core.config import V8Config
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model

from src.nodes.trust_engine import (
    DOMAIN_TRUST_HIGH,
    DOMAIN_TRUST_MEDIUM,
    DOMAIN_TRUST_LOW,
    _calculate_domain_trust,
)


BATCHED_CREDIBILITY_CLAIMS_SYSTEM = """You are an expert at analyzing sources and extracting factual claims.

TASK 1: Assess source credibility
For each source, evaluate:
- AUTHORITY (0-1): Author credentials, citations, official status
- CONTENT_QUALITY (0-1): Detailed, specific, well-structured content

TASK 2: Extract factual claims
From the evidence, extract 8-15 distinct factual claims organized by section.
Each claim should be:
- A single, verifiable statement
- Directly supported by the evidence
- Labeled with which section it belongs to

Return JSON with this structure:
{
  "source_assessments": [
    {"sid": "S1", "authority": 0.7, "content_quality": 0.8}
  ],
  "claims": [
    {"cid": "C1", "text": "Factual claim here", "section": "Background", "supporting_eids": ["E1", "E3"]}
  ]
}
"""


def batched_credibility_claims_node(state: AgentState) -> Dict[str, Any]:
    """Scores source credibility and extracts claims in a single LLM call."""
    cfg = V8Config()
    sources = state.get("sources") or []
    evidence = state.get("evidence") or []
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []

    if not sources:
        return {"claims": [], "source_credibility": {}}

    domain_scores = {}
    for s in sources:
        url = s.get("url", "")
        sid = s.get("sid", "")
        domain_scores[sid] = _calculate_domain_trust(url)

    sources_text = []
    for s in sources[:15]:
        sources_text.append(
            f"[{s['sid']}] {s.get('title', 'Untitled')}\n"
            f"URL: {s.get('url', '')}\n"
            f"Snippet: {s.get('snippet', '')[:300]}"
        )

    evidence_text = []
    for e in evidence[:20]:
        evidence_text.append(
            f"[{e['eid']}] (from {e.get('sid', 'unknown')})\n"
            f"{e['text'][:400]}"
        )

    sections_str = ", ".join(outline[:8]) if outline else "General"

    llm = create_chat_model(model=cfg.get_model_for_node("credibility"), temperature=0.1)

    prompt = f"""SOURCES:
{chr(10).join(sources_text)}

EVIDENCE:
{chr(10).join(evidence_text)}

SECTIONS to organize claims: {sections_str}

Perform both tasks and return the combined JSON."""

    resp = llm.invoke([
        SystemMessage(content=BATCHED_CREDIBILITY_CLAIMS_SYSTEM),
        HumanMessage(content=prompt)
    ])

    result = parse_json_object(resp.content, default={})

    source_assessments = result.get("source_assessments", [])
    llm_scores = {}
    for item in source_assessments:
        if isinstance(item, dict):
            sid = item.get("sid", "")
            llm_scores[sid] = {
                "authority": float(item.get("authority", 0.5)),
                "content_quality": float(item.get("content_quality", 0.5)),
            }

    source_credibility: Dict[str, SourceCredibility] = {}
    min_credibility = state.get("min_source_credibility", 0.35)

    for s in sources:
        sid = s.get("sid", "")
        url = s.get("url", "")

        domain_trust = domain_scores.get(sid, 0.5)
        freshness = 0.7
        authority = llm_scores.get(sid, {}).get("authority", 0.5)
        content_quality = llm_scores.get(sid, {}).get("content_quality", 0.5)

        overall = (
            domain_trust * 0.30 +
            freshness * 0.15 +
            authority * 0.25 +
            content_quality * 0.30
        )

        source_credibility[sid] = {
            "sid": sid,
            "url": url,
            "domain_trust": domain_trust,
            "freshness": freshness,
            "authority": authority,
            "content_quality": content_quality,
            "overall": overall,
        }

        s["credibility"] = overall

    filtered_sources = [
        s for s in sources
        if source_credibility.get(s.get("sid", ""), {}).get("overall", 0) >= min_credibility
    ]

    valid_sids = {s["sid"] for s in filtered_sources}
    filtered_evidence = [
        e for e in evidence
        if e.get("sid") in valid_sids
    ]

    raw_claims = result.get("claims", [])
    claims = []
    for i, c in enumerate(raw_claims):
        if isinstance(c, dict) and c.get("text"):
            claims.append({
                "cid": c.get("cid", f"C{i+1}"),
                "text": c["text"][:500],
                "section": c.get("section", "General"),
                "eids": c.get("supporting_eids", []),
            })

    citations = [
        {"cid": c["cid"], "eids": c.get("eids", [])}
        for c in claims
    ]

    return {
        "sources": filtered_sources,
        "evidence": filtered_evidence,
        "source_credibility": source_credibility,
        "claims": claims,
        "citations": citations,
    }


BATCHED_VERIFICATION_SYSTEM = """You are an expert fact-checker. Perform comprehensive verification of claims.

For each claim:
1. SPAN VERIFICATION: Find the exact text in evidence that supports it
2. CROSS-VALIDATION: Check if multiple sources support it
3. CONFIDENCE SCORING: Rate overall confidence (0-1)

Return JSON array:
[
  {
    "cid": "C1",
    "verified": true,
    "evidence_span": "Exact quote from evidence supporting this",
    "source_eid": "E3",
    "match_confidence": 0.9,
    "cross_validated": true,
    "supporting_sids": ["S1", "S3", "S5"],
    "final_confidence": 0.85
  },
  {
    "cid": "C2",
    "verified": false,
    "reason": "No evidence supports this claim",
    "final_confidence": 0.0
  }
]

Guidelines:
- verified=true: Claim is directly stated or strongly implied in evidence
- match_confidence: 0.9+ for exact matches, 0.7-0.9 for paraphrases, 0.5-0.7 for inferences
- cross_validated=true: At least 2 independent sources support the claim
- final_confidence: Overall reliability considering verification, cross-validation, and source quality
"""


def batched_verification_node(state: AgentState) -> Dict[str, Any]:
    """Performs span verification, cross-validation, and confidence scoring in a single LLM call."""
    cfg = V8Config()
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []
    citations = state.get("citations") or []
    source_credibility = state.get("source_credibility") or {}

    if not claims or not evidence:
        return {
            "verified_citations": [],
            "unverified_claims": [c["cid"] for c in claims],
            "cross_validated_claims": [],
            "single_source_claims": [],
            "claim_confidence": {},
            "overall_confidence": 0.0,
            "hallucination_score": 1.0 if claims else 0.0,
        }

    eid_to_evidence = {e["eid"]: e for e in evidence}
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}
    cid_to_eids = {c["cid"]: c.get("eids", []) for c in citations}

    claims_text = json.dumps([
        {"cid": c["cid"], "text": c["text"], "section": c.get("section", "General")}
        for c in claims
    ], indent=2)

    evidence_text = "\n\n".join([
        f"[{e['eid']}] (Source: {e['sid']})\n{e['text'][:400]}"
        for e in evidence[:25]
    ])

    source_info = "\n".join([
        f"{sid}: credibility={cred.get('overall', 0.5):.2f}"
        for sid, cred in list(source_credibility.items())[:15]
    ])

    llm = create_chat_model(model=cfg.get_model_for_node("cite"), temperature=0)

    prompt = f"""CLAIMS TO VERIFY:
{claims_text}

EVIDENCE:
{evidence_text}

SOURCE CREDIBILITY SCORES:
{source_info}

Verify each claim, checking for span matches and cross-validation."""

    resp = llm.invoke([
        SystemMessage(content=BATCHED_VERIFICATION_SYSTEM),
        HumanMessage(content=prompt)
    ])

    verifications = parse_json_array(resp.content, default=[])

    verified_citations: List[VerifiedCitation] = []
    unverified_claims: List[str] = []
    cross_validated_claims: List[VerifiedCitation] = []
    single_source_claims: List[VerifiedCitation] = []
    claim_confidence: Dict[str, float] = {}
    section_confidence: Dict[str, List[float]] = {}

    for v in verifications:
        if not isinstance(v, dict):
            continue

        cid = v.get("cid", "")
        verified = v.get("verified", False)
        confidence = float(v.get("final_confidence", 0.0))

        claim = next((c for c in claims if c["cid"] == cid), {})
        section = claim.get("section", "General")

        claim_confidence[cid] = confidence

        if section not in section_confidence:
            section_confidence[section] = []
        section_confidence[section].append(confidence)

        if verified:
            vc: VerifiedCitation = {
                "cid": cid,
                "eid": v.get("source_eid", ""),
                "claim_text": claim.get("text", ""),
                "evidence_span": v.get("evidence_span", ""),
                "match_score": float(v.get("match_confidence", 0.5)),
                "verified": True,
                "cross_validated": v.get("cross_validated", False),
                "supporting_sids": v.get("supporting_sids", []),
            }
            verified_citations.append(vc)

            if v.get("cross_validated", False):
                cross_validated_claims.append(vc)
            else:
                single_source_claims.append(vc)
        else:
            unverified_claims.append(cid)

    verified_cids = {vc["cid"] for vc in verified_citations}
    unverified_cids = set(unverified_claims)
    for claim in claims:
        cid = claim["cid"]
        if cid not in verified_cids and cid not in unverified_cids:
            unverified_claims.append(cid)

    total_claims = len(claims)
    verified_count = len(verified_citations)
    hallucination_score = 1.0 - (verified_count / total_claims) if total_claims > 0 else 0.0

    section_conf_avg = {
        section: sum(confs) / len(confs) if confs else 0.0
        for section, confs in section_confidence.items()
    }

    overall = sum(claim_confidence.values()) / len(claim_confidence) if claim_confidence else 0.0

    return {
        "verified_citations": verified_citations,
        "unverified_claims": unverified_claims,
        "cross_validated_claims": cross_validated_claims,
        "single_source_claims": single_source_claims,
        "claim_confidence": claim_confidence,
        "section_confidence": section_conf_avg,
        "overall_confidence": overall,
        "hallucination_score": hallucination_score,
    }


def full_trust_engine_batched(state: AgentState) -> Dict[str, Any]:
    """Run the full trust engine in just 2 LLM calls (credibility+claims, then verification+confidence)."""
    result1 = batched_credibility_claims_node(state)
    updated_state = {**state, **result1}
    result2 = batched_verification_node(updated_state)
    return {**result1, **result2}
