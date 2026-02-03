# src/utils/optimization.py
"""
Optimization utilities for the research pipeline.

Features:
- Query deduplication using embedding similarity
- Cost tracking
- Early termination logic
"""
from __future__ import annotations

import hashlib
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher


def normalize_query(query: str) -> str:
    """Normalize a query for comparison."""
    # Lowercase, strip, remove extra whitespace
    normalized = " ".join(query.lower().strip().split())
    # Remove common filler words
    fillers = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to"}
    words = [w for w in normalized.split() if w not in fillers]
    return " ".join(words)


def query_similarity(q1: str, q2: str) -> float:
    """
    Calculate similarity between two queries using SequenceMatcher.
    Returns float between 0 and 1.
    """
    n1 = normalize_query(q1)
    n2 = normalize_query(q2)
    return SequenceMatcher(None, n1, n2).ratio()


def deduplicate_queries(
    queries: List[str],
    threshold: float = 0.85
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Deduplicate a list of queries based on similarity.

    Args:
        queries: List of query strings
        threshold: Similarity threshold (0-1) above which queries are considered duplicates

    Returns:
        Tuple of (deduplicated_queries, removed_duplicates_with_kept_query)
    """
    if not queries:
        return [], []

    unique_queries: List[str] = []
    duplicates: List[Tuple[str, str]] = []  # (removed, kept)

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
    """
    Deduplicate query items (dicts with 'query' key) based on similarity.

    Args:
        query_items: List of dicts with 'query' key
        threshold: Similarity threshold

    Returns:
        Tuple of (deduplicated_items, count_removed)
    """
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
                # Merge priorities if the duplicate has higher priority
                if item.get("priority", 0) > existing.get("priority", 0):
                    existing["priority"] = item["priority"]
                break

        if not is_duplicate:
            unique_items.append(item)
        else:
            removed_count += 1

    return unique_items, removed_count


class CostTracker:
    """Track API costs for early termination."""

    # Approximate costs per 1K tokens (as of 2024)
    MODEL_COSTS = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    TAVILY_COST_PER_SEARCH = 0.01  # Approximate

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
        """Record an LLM call and return cost."""
        costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["gpt-4o-mini"])
        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])
        self.total_cost += cost
        self.llm_calls += 1
        self.tokens_used["input"] += input_tokens
        self.tokens_used["output"] += output_tokens
        return cost

    def add_tavily_search(self, count: int = 1) -> float:
        """Record Tavily searches and return cost."""
        cost = count * self.TAVILY_COST_PER_SEARCH
        self.total_cost += cost
        self.tavily_searches += count
        return cost

    def is_over_budget(self) -> bool:
        """Check if we've exceeded the budget."""
        return self.total_cost >= self.budget_usd

    def remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.budget_usd - self.total_cost)

    def get_summary(self) -> Dict:
        """Get cost summary."""
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
    Determine if research should terminate early.

    Returns:
        Tuple of (should_terminate, reason)
    """
    # Hard limit on iterations
    if iteration >= max_iterations:
        return True, f"max_iterations_reached ({max_iterations})"

    # Cost budget exceeded
    if cost_tracker and cost_tracker.is_over_budget():
        return True, f"cost_budget_exceeded (${cost_tracker.total_cost:.2f})"

    # Diminishing returns on confidence
    confidence_delta = current_confidence - previous_confidence
    if confidence_delta < min_confidence_delta and iteration > 0:
        return True, f"diminishing_returns (delta={confidence_delta:.2%})"

    # Not finding new sources
    if new_sources_found < min_new_sources and iteration > 0:
        return True, f"insufficient_new_sources ({new_sources_found})"

    # High confidence already achieved
    if current_confidence >= 0.9:
        return True, f"high_confidence_achieved ({current_confidence:.0%})"

    return False, ""
