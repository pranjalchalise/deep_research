# src/nodes/trust_engine.py
"""
Enhanced trust engine nodes for v8.

Implements:
- credibility_scorer_node: Score sources using E-E-A-T and domain trust
- span_verify_node: Verify each claim against exact text spans
- cross_validate_node: Check if claims are supported by multiple sources
- claim_confidence_scorer_node: Calculate per-claim and overall confidence
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import (
    AgentState,
    SourceCredibility,
    VerifiedCitation,
)
from src.utils.json_utils import parse_json_object, parse_json_array
from src.utils.llm import create_chat_model


# ============================================================================
# CREDIBILITY SCORER NODE
# ============================================================================

# Domain trust tiers
DOMAIN_TRUST_HIGH = {
    ".edu": 0.9,
    ".gov": 0.9,
    "arxiv.org": 0.9,
    "nature.com": 0.9,
    "science.org": 0.9,
    "ncbi.nlm.nih.gov": 0.9,
    "ieee.org": 0.85,
    "acm.org": 0.85,
    "scholar.google.com": 0.85,
}

DOMAIN_TRUST_MEDIUM = {
    "wikipedia.org": 0.75,
    "github.com": 0.75,
    "linkedin.com": 0.7,
    "microsoft.com": 0.75,
    "google.com": 0.75,
    "amazon.com": 0.7,
    "nytimes.com": 0.7,
    "bbc.com": 0.7,
    "reuters.com": 0.75,
    "forbes.com": 0.65,
    "techcrunch.com": 0.65,
}

DOMAIN_TRUST_LOW = {
    "medium.com": 0.45,
    "quora.com": 0.4,
    "reddit.com": 0.5,
    "pinterest.com": 0.3,
    "buzzfeed.com": 0.35,
    "wikihow.com": 0.4,
}

CREDIBILITY_ASSESSMENT_SYSTEM = """Assess the content quality and authority of sources.

For each source, evaluate:
1. AUTHORITY (0-1): Does it have author credentials, citations, or official status?
2. CONTENT_QUALITY (0-1): Is it detailed, specific, and well-structured?

Return JSON:
[
  {"sid": "S1", "authority": 0.7, "content_quality": 0.8, "notes": "brief explanation"}
]

Guidelines:
- Official company/org pages = high authority (0.8+)
- Academic publications = high authority (0.9)
- News articles = medium authority (0.6-0.7)
- Blog posts = lower authority (0.4-0.6)
- Detailed, specific content = high quality (0.8+)
- Listicles, thin content = low quality (0.3-0.5)
"""


def credibility_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    Score each source's credibility using E-E-A-T principles.

    Components:
    1. Domain trust (rule-based)
    2. Freshness (from publish date if available)
    3. Authority (LLM-assessed)
    4. Content quality (LLM-assessed)
    """
    sources = state.get("sources") or []
    evidence = state.get("evidence") or []

    if not sources:
        return {}

    # Step 1: Domain trust (rule-based)
    domain_scores = {}
    for s in sources:
        url = s.get("url", "")
        sid = s.get("sid", "")
        domain_scores[sid] = _calculate_domain_trust(url)

    # Step 2: LLM assessment of authority and content quality
    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    # Prepare sources for LLM
    sources_text = []
    for s in sources[:15]:  # Limit to avoid token issues
        sources_text.append(
            f"[{s['sid']}] {s.get('title', 'Untitled')}\n"
            f"URL: {s.get('url', '')}\n"
            f"Snippet: {s.get('snippet', '')[:300]}"
        )

    resp = llm.invoke([
        SystemMessage(content=CREDIBILITY_ASSESSMENT_SYSTEM),
        HumanMessage(content=f"Assess these sources:\n\n" + "\n\n".join(sources_text))
    ])

    llm_scores = {}
    for item in parse_json_array(resp.content, default=[]):
        if isinstance(item, dict):
            sid = item.get("sid", "")
            llm_scores[sid] = {
                "authority": float(item.get("authority", 0.5)),
                "content_quality": float(item.get("content_quality", 0.5)),
            }

    # Step 3: Combine scores
    source_credibility: Dict[str, SourceCredibility] = {}
    min_credibility = state.get("min_source_credibility", 0.35)

    for s in sources:
        sid = s.get("sid", "")
        url = s.get("url", "")

        domain_trust = domain_scores.get(sid, 0.5)
        freshness = 0.7  # Default if no date available
        authority = llm_scores.get(sid, {}).get("authority", 0.5)
        content_quality = llm_scores.get(sid, {}).get("content_quality", 0.5)

        # Weighted average
        # Domain trust: 30%, Freshness: 15%, Authority: 25%, Content quality: 30%
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

        # Update source with credibility score
        s["credibility"] = overall

    # Filter low-credibility sources
    filtered_sources = [
        s for s in sources
        if source_credibility.get(s.get("sid", ""), {}).get("overall", 0) >= min_credibility
    ]

    # Also filter evidence
    valid_sids = {s["sid"] for s in filtered_sources}
    filtered_evidence = [
        e for e in evidence
        if e.get("sid") in valid_sids
    ]

    return {
        "sources": filtered_sources,
        "evidence": filtered_evidence,
        "source_credibility": source_credibility,
    }


def _calculate_domain_trust(url: str) -> float:
    """Calculate domain trust score based on URL."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        host = host.replace("www.", "")

        # Check high trust domains
        for pattern, score in DOMAIN_TRUST_HIGH.items():
            if pattern.startswith("."):
                # TLD check
                if host.endswith(pattern):
                    return score
            else:
                # Domain check
                if pattern in host:
                    return score

        # Check medium trust domains
        for pattern, score in DOMAIN_TRUST_MEDIUM.items():
            if pattern in host:
                return score

        # Check low trust domains
        for pattern, score in DOMAIN_TRUST_LOW.items():
            if pattern in host:
                return score

        # Default
        return 0.5

    except Exception:
        return 0.5


# ============================================================================
# SPAN VERIFY NODE
# ============================================================================

SPAN_VERIFY_SYSTEM = """Verify if claims are supported by exact evidence spans.

For each claim, find the EXACT TEXT in the evidence that supports it.
If no exact support exists, mark as unverified.

Return JSON array:
[
  {
    "cid": "C1",
    "verified": true,
    "evidence_span": "The exact quote from evidence that supports this claim",
    "source_eid": "E3",
    "match_confidence": 0.95
  },
  {
    "cid": "C2",
    "verified": false,
    "reason": "No evidence directly supports this claim"
  }
]

Guidelines:
- VERIFIED: The claim is directly stated or strongly implied in the evidence
- UNVERIFIED: The claim goes beyond what the evidence states, or contradicts it
- Match confidence: 0.9+ for exact matches, 0.7-0.9 for paraphrases, 0.5-0.7 for inferences
"""


def span_verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify each claim against exact evidence spans.

    This prevents:
    - Hallucinated facts
    - Misattributed claims
    - Over-generalization
    """
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []
    citations = state.get("citations") or []

    if not claims or not evidence:
        return {
            "verified_citations": [],
            "unverified_claims": [c["cid"] for c in claims],
            "hallucination_score": 1.0 if claims else 0.0,
        }

    llm = create_chat_model(model="gpt-4o-mini", temperature=0)

    # Build evidence lookup
    eid_to_evidence = {e["eid"]: e for e in evidence}
    cid_to_eids = {c["cid"]: c.get("eids", []) for c in citations}

    # Process claims in batches to reduce API calls
    batch_size = 5
    all_verifications = []

    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]

        # Gather relevant evidence for this batch
        batch_evidence = []
        for claim in batch_claims:
            cid = claim["cid"]
            eids = cid_to_eids.get(cid, [])
            for eid in eids:
                if eid in eid_to_evidence and eid_to_evidence[eid] not in batch_evidence:
                    batch_evidence.append(eid_to_evidence[eid])

        # If no specific citations, include some evidence
        if not batch_evidence:
            batch_evidence = evidence[:5]

        # Format for LLM
        claims_text = json.dumps([
            {"cid": c["cid"], "text": c["text"]}
            for c in batch_claims
        ], indent=2)

        evidence_text = "\n\n".join([
            f"[{e['eid']}] (from {e.get('sid', 'unknown')})\n{e['text']}"
            for e in batch_evidence
        ])

        resp = llm.invoke([
            SystemMessage(content=SPAN_VERIFY_SYSTEM),
            HumanMessage(content=f"CLAIMS:\n{claims_text}\n\nEVIDENCE:\n{evidence_text}\n\nVerify each claim.")
        ])

        verifications = parse_json_array(resp.content, default=[])
        all_verifications.extend(verifications)

    # Process verifications
    verified_citations: List[VerifiedCitation] = []
    unverified_claims: List[str] = []

    for v in all_verifications:
        if not isinstance(v, dict):
            continue

        cid = v.get("cid", "")
        verified = v.get("verified", False)

        if verified:
            verified_citations.append({
                "cid": cid,
                "eid": v.get("source_eid", ""),
                "claim_text": next((c["text"] for c in claims if c["cid"] == cid), ""),
                "evidence_span": v.get("evidence_span", ""),
                "match_score": float(v.get("match_confidence", 0.5)),
                "verified": True,
                "cross_validated": False,
                "supporting_sids": [],
            })
        else:
            unverified_claims.append(cid)

    # Add claims that weren't in verifications
    verified_cids = {vc["cid"] for vc in verified_citations}
    unverified_cids = set(unverified_claims)
    for claim in claims:
        cid = claim["cid"]
        if cid not in verified_cids and cid not in unverified_cids:
            unverified_claims.append(cid)

    # Calculate hallucination score
    total_claims = len(claims)
    verified_count = len(verified_citations)
    hallucination_score = 1.0 - (verified_count / total_claims) if total_claims > 0 else 0.0

    return {
        "verified_citations": verified_citations,
        "unverified_claims": unverified_claims,
        "hallucination_score": hallucination_score,
    }


# ============================================================================
# CROSS VALIDATE NODE
# ============================================================================

CROSS_VALIDATE_SYSTEM = """Identify which claims are supported by multiple independent sources.

For each verified claim, check if OTHER evidence items (from different sources) also support it.

Return JSON:
[
  {
    "cid": "C1",
    "cross_validated": true,
    "supporting_eids": ["E1", "E4", "E7"],
    "supporting_sids": ["S1", "S3", "S5"]
  },
  {
    "cid": "C2",
    "cross_validated": false,
    "reason": "Only one source mentions this"
  }
]
"""


def cross_validate_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if claims are supported by multiple independent sources.

    Cross-validated claims are more reliable.
    """
    verified_citations = state.get("verified_citations") or []
    evidence = state.get("evidence") or []

    if not verified_citations or not evidence:
        return {
            "cross_validated_claims": [],
            "single_source_claims": verified_citations,
        }

    # Build EID to SID mapping
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}
    eid_to_text = {e["eid"]: e["text"] for e in evidence}

    llm = create_chat_model(model="gpt-4o-mini", temperature=0)

    # Format claims and evidence
    claims_text = json.dumps([
        {"cid": vc["cid"], "claim": vc["claim_text"], "primary_eid": vc["eid"]}
        for vc in verified_citations
    ], indent=2)

    evidence_text = "\n\n".join([
        f"[{e['eid']}] (Source: {e['sid']})\n{e['text'][:300]}"
        for e in evidence
    ])

    resp = llm.invoke([
        SystemMessage(content=CROSS_VALIDATE_SYSTEM),
        HumanMessage(content=f"VERIFIED CLAIMS:\n{claims_text}\n\nALL EVIDENCE:\n{evidence_text}\n\nIdentify cross-validated claims.")
    ])

    cross_validation = parse_json_array(resp.content, default=[])

    # Build lookup
    cv_lookup = {}
    for cv in cross_validation:
        if isinstance(cv, dict):
            cv_lookup[cv.get("cid", "")] = cv

    cross_validated_claims: List[VerifiedCitation] = []
    single_source_claims: List[VerifiedCitation] = []

    for vc in verified_citations:
        cid = vc["cid"]
        cv_info = cv_lookup.get(cid, {})

        if cv_info.get("cross_validated", False):
            supporting_sids = cv_info.get("supporting_sids", [])
            if not supporting_sids:
                # Derive from EIDs
                supporting_eids = cv_info.get("supporting_eids", [])
                supporting_sids = list(set(eid_to_sid.get(eid, "") for eid in supporting_eids))
                supporting_sids = [s for s in supporting_sids if s]

            cross_validated_claims.append({
                **vc,
                "cross_validated": True,
                "supporting_sids": supporting_sids,
            })
        else:
            single_source_claims.append({
                **vc,
                "cross_validated": False,
                "supporting_sids": [eid_to_sid.get(vc.get("eid", ""), "")],
            })

    return {
        "cross_validated_claims": cross_validated_claims,
        "single_source_claims": single_source_claims,
    }


# ============================================================================
# CLAIM CONFIDENCE SCORER NODE
# ============================================================================

def claim_confidence_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    Calculate per-claim and overall confidence scores.

    Factors:
    - Span verification match score
    - Cross-validation (multiple sources)
    - Source credibility of supporting sources
    """
    verified_citations = state.get("verified_citations") or []
    cross_validated_claims = state.get("cross_validated_claims") or []
    single_source_claims = state.get("single_source_claims") or []
    source_credibility = state.get("source_credibility") or {}
    claims = state.get("claims") or []

    # Create lookup for cross-validated status
    cv_cids = {vc["cid"] for vc in cross_validated_claims}

    claim_confidence: Dict[str, float] = {}
    section_confidence: Dict[str, List[float]] = {}

    for vc in verified_citations:
        cid = vc["cid"]
        claim_text = vc.get("claim_text", "")

        # Find the claim to get its section
        claim = next((c for c in claims if c["cid"] == cid), {})
        section = claim.get("section", "General")

        # Base: span match score
        base_score = vc.get("match_score", 0.5)

        # Bonus: cross-validation
        is_cross_validated = cid in cv_cids
        cross_bonus = 0.15 if is_cross_validated else 0

        # Weight by source credibility
        supporting_sids = vc.get("supporting_sids", [])
        if supporting_sids:
            avg_source_cred = sum(
                source_credibility.get(sid, {}).get("overall", 0.5)
                for sid in supporting_sids
            ) / len(supporting_sids)
        else:
            avg_source_cred = 0.5

        # Final confidence
        confidence = min(1.0, base_score * 0.5 + avg_source_cred * 0.35 + cross_bonus)
        claim_confidence[cid] = confidence

        # Track section confidence
        if section not in section_confidence:
            section_confidence[section] = []
        section_confidence[section].append(confidence)

    # Calculate section-level confidence (average)
    section_conf_avg = {}
    for section, confs in section_confidence.items():
        section_conf_avg[section] = sum(confs) / len(confs) if confs else 0.0

    # Overall confidence
    if claim_confidence:
        overall = sum(claim_confidence.values()) / len(claim_confidence)
    else:
        overall = 0.0

    return {
        "claim_confidence": claim_confidence,
        "section_confidence": section_conf_avg,
        "overall_confidence": overall,
    }
