"""
Query dedup, cost tracking, and early-termination for the research loop.

Keeps the planner from issuing near-duplicate searches and lets us bail
out early when confidence plateaus or the budget runs dry.
"""
from __future__ import annotations

import hashlib
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher


def normalize_query(query: str) -> str:
    """Lowercase, drop filler words, collapse whitespace -- just for comparison purposes."""
    normalized = " ".join(query.lower().strip().split())
    fillers = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to"}
    words = [w for w in normalized.split() if w not in fillers]
    return " ".join(words)


def query_similarity(q1: str, q2: str) -> float:
    """How similar are two queries? Returns 0-1 via SequenceMatcher."""
    n1 = normalize_query(q1)
    n2 = normalize_query(q2)
    return SequenceMatcher(None, n1, n2).ratio()


def deduplicate_queries(
    queries: List[str],
    threshold: float = 0.85
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Drop near-duplicate queries. Returns (unique, list of (removed, kept_instead) pairs)."""
    if not queries:
        return [], []

    unique_queries: List[str] = []
    duplicates: List[Tuple[str, str]] = []

    for query in queries:
        is_duplicate = False
        for existing in unique_queries:
            sim = query_similarity(query, existing)
            if sim >= threshold:
                is_duplicate = True
                duplicates.append((query, existing))
                break

        if not is_duplicate:
            unique_queries.append(query)

    return unique_queries, duplicates


def deduplicate_query_items(
    query_items: List[Dict],
    threshold: float = 0.85
) -> Tuple[List[Dict], int]:
    """Same as deduplicate_queries but for dicts with a 'query' key. Merges priority upward on collision."""
    if not query_items:
        return [], 0

    unique_items: List[Dict] = []
    removed_count = 0

    for item in query_items:
        query = item.get("query", "")
        is_duplicate = False

        for existing in unique_items:
            existing_query = existing.get("query", "")
            sim = query_similarity(query, existing_query)
            if sim >= threshold:
                is_duplicate = True
                if item.get("priority", 0) > existing.get("priority", 0):
                    existing["priority"] = item["priority"]
                break

        if not is_duplicate:
            unique_items.append(item)
        else:
            removed_count += 1

    return unique_items, removed_count


class CostTracker:
    """
    Tracks estimated API spend so we can stop before blowing the budget.

    The per-token costs are rough approximations -- good enough for
    deciding when to stop, not for actual billing.
    """

    MODEL_COSTS = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    TAVILY_COST_PER_SEARCH = 0.01

    def __init__(self, budget_usd: float = 0.50):
        self.budget_usd = budget_usd
        self.total_cost = 0.0
        self.llm_calls = 0
        self.tavily_searches = 0
        self.tokens_used = {"input": 0, "output": 0}

    def add_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Record an LLM call and return its estimated cost."""
        costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["gpt-4o-mini"])
        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])
        self.total_cost += cost
        self.llm_calls += 1
        self.tokens_used["input"] += input_tokens
        self.tokens_used["output"] += output_tokens
        return cost

    def add_tavily_search(self, count: int = 1) -> float:
        """Record Tavily searches and return their estimated cost."""
        cost = count * self.TAVILY_COST_PER_SEARCH
        self.total_cost += cost
        self.tavily_searches += count
        return cost

    def is_over_budget(self) -> bool:
        return self.total_cost >= self.budget_usd

    def remaining_budget(self) -> float:
        return max(0, self.budget_usd - self.total_cost)

    def get_summary(self) -> Dict:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "budget_usd": self.budget_usd,
            "remaining_usd": round(self.remaining_budget(), 4),
            "llm_calls": self.llm_calls,
            "tavily_searches": self.tavily_searches,
            "tokens": self.tokens_used,
            "over_budget": self.is_over_budget(),
        }


def should_terminate_early(
    current_confidence: float,
    previous_confidence: float,
    new_sources_found: int,
    iteration: int,
    max_iterations: int,
    min_confidence_delta: float = 0.05,
    min_new_sources: int = 2,
    cost_tracker: Optional[CostTracker] = None,
) -> Tuple[bool, str]:
    """
    Decide if the research loop should stop early.

    We check several conditions: hard iteration cap, budget exceeded,
    confidence not improving, not finding new sources, or already
    confident enough. Returns (should_stop, reason).
    """
    if iteration >= max_iterations:
        return True, f"max_iterations_reached ({max_iterations})"

    if cost_tracker and cost_tracker.is_over_budget():
        return True, f"cost_budget_exceeded (${cost_tracker.total_cost:.2f})"

    # If confidence barely moved, more iterations probably won't help
    confidence_delta = current_confidence - previous_confidence
    if confidence_delta < min_confidence_delta and iteration > 0:
        return True, f"diminishing_returns (delta={confidence_delta:.2%})"

    if new_sources_found < min_new_sources and iteration > 0:
        return True, f"insufficient_new_sources ({new_sources_found})"

    # 90%+ confidence is good enough -- diminishing returns from here
    if current_confidence >= 0.9:
        return True, f"high_confidence_achieved ({current_confidence:.0%})"

    return False, ""
