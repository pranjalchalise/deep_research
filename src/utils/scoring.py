"""
Heuristic quality scoring for search results.

Combines domain reputation, lane alignment, recency, and SEO penalties
into a 0-1 score that the ranker uses to prioritize which sources to
actually fetch and read.
"""
from __future__ import annotations

from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, Tuple

PRIMARY_BONUS_DOMAINS = {
    "docs.langchain.com": 0.25,
    "langchain.com": 0.20,
    "github.com": 0.15,
    "arxiv.org": 0.25,
    "acm.org": 0.25,
    "ieee.org": 0.25,
    "nature.com": 0.25,
    "sciencedirect.com": 0.20,
}

SEO_PENALTY_DOMAINS = {
    "medium.com": -0.10,
    "towardsdatascience.com": -0.08,
    "substack.com": -0.06,
    "blogspot.com": -0.10,
}

FORUM_HINT_DOMAINS = {"reddit.com", "news.ycombinator.com"}

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def parse_date(d: str) -> Optional[datetime]:
    """Best-effort date parsing for the inconsistent formats Tavily returns."""
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(d[:len(fmt)], fmt)
        except Exception:
            continue
    return None

def recency_score(published_date: str) -> float:
    dt = parse_date(published_date or "")
    if not dt:
        return 0.0
    age_days = (datetime.utcnow() - dt.replace(tzinfo=None)).days
    if age_days <= 7:
        return 0.15
    if age_days <= 30:
        return 0.10
    if age_days <= 180:
        return 0.05
    return 0.0

def lane_bonus(domain: str, lane: str) -> float:
    if lane == "docs" and ("docs." in domain or "reference." in domain):
        return 0.12
    if lane == "papers" and ("arxiv.org" in domain or "acm." in domain or "ieee." in domain):
        return 0.12
    if lane == "news" and ("news" in domain or "press" in domain):
        return 0.06
    if lane == "forums" and any(d in domain for d in FORUM_HINT_DOMAINS):
        return 0.06
    return 0.0

def quality_score(url: str, title: str, snippet: str, lane: str, published_date: str = "") -> float:
    """Return a 0-1 heuristic quality score based on domain trust, lane fit, and recency."""
    domain = _domain(url)
    score = 0.50  # base

    for dom, bonus in PRIMARY_BONUS_DOMAINS.items():
        if domain.endswith(dom):
            score += bonus
            break

    for dom, pen in SEO_PENALTY_DOMAINS.items():
        if domain.endswith(dom):
            score += pen
            break

    score += lane_bonus(domain, lane)
    score += recency_score(published_date)

    if "pdf" in url.lower():
        score += 0.03
    if "documentation" in (title or "").lower():
        score += 0.03

    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score
