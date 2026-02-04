from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class V8Config:
    """All tunables for the research pipeline in one place. Frozen so nodes
    can't accidentally mutate config mid-run. The defaults here reflect
    cost/quality tradeoffs learned from production runs."""

    # Planning
    max_rounds: int = 2
    queries_per_round: int = 10

    # Iterative research loop
    max_research_iterations: int = 2
    min_confidence_to_proceed: float = 0.7
    max_research_time_minutes: float = 10.0
    enable_backtracking: bool = True

    # Multi-agent orchestration
    use_multi_agent: bool = True
    max_subagents: int = 3
    subagent_model: str = "gpt-4o-mini"
    orchestrator_model: str = "gpt-4o"
    subagent_max_iterations: int = 2

    # Per-node model selection -- planner and writer use the expensive model
    # because output quality matters most there; everything else uses mini.
    model_routing: Tuple[Tuple[str, str], ...] = (
        ("analyzer", "gpt-4o-mini"),
        ("discovery", "gpt-4o-mini"),
        ("planner", "gpt-4o"),
        ("orchestrator", "gpt-4o-mini"),
        ("subagent", "gpt-4o-mini"),
        ("claims", "gpt-4o-mini"),
        ("cite", "gpt-4o-mini"),
        ("credibility", "gpt-4o-mini"),
        ("writer", "gpt-4o"),
    )

    # Complexity-based routing -- simple queries skip multi-agent overhead
    enable_complexity_routing: bool = True
    simple_query_keywords: Tuple[str, ...] = (
        "what is", "who is", "define", "when was", "where is",
    )
    complex_query_keywords: Tuple[str, ...] = (
        "compare", "analyze", "evaluate", "research", "investigate",
        "comprehensive", "in-depth", "detailed analysis",
    )

    # Query deduplication (cuts ~30% of redundant Tavily calls)
    enable_query_dedup: bool = True
    query_similarity_threshold: float = 0.85

    # Batching reduces LLM round-trips by combining verification steps
    batch_trust_engine: bool = True
    batch_claims_credibility: bool = True

    # Early termination -- stop iterating when gains flatline
    enable_early_termination: bool = True
    min_confidence_delta: float = 0.05
    min_new_sources_threshold: int = 2
    cost_budget_usd: float = 0.50

    # Retrieval
    tavily_max_results: int = 6
    rerank_top_n: int = 15
    select_sources_k: int = 10

    # Page reading / chunking
    chunk_chars: int = 3500
    chunk_overlap: int = 350
    max_chunks_per_source: int = 4
    evidence_per_source: int = 3

    # Source credibility (E-E-A-T inspired)
    enable_credibility_scoring: bool = True
    min_source_credibility: float = 0.35

    trusted_domains: tuple = (
        ".edu", ".gov", ".org",
        "arxiv.org", "nature.com", "science.org",
        "wikipedia.org", "github.com", "scholar.google.com",
        "ncbi.nlm.nih.gov", "ieee.org", "acm.org",
    )

    low_trust_domains: tuple = (
        "pinterest.com", "quora.com",
        "buzzfeed.com", "wikihow.com",
    )

    # Citation verification
    enable_span_verification: bool = True
    enable_cross_validation: bool = True
    span_match_threshold: float = 0.6
    cross_validation_threshold: int = 2        # min sources for cross-validation
    hallucination_threshold: float = 0.3       # flag report if >30% claims unsupported

    # Confidence display thresholds
    enable_confidence_scoring: bool = True
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6

    # Trust / termination
    min_source_score_for_strong: float = 0.55
    require_min_sources: int = 4
    allow_news: bool = True

    # Runtime
    fast_mode: bool = False
    request_timeout_s: float = 15.0
    cache_dir: str = ".cache_v8"
    use_cache: bool = True

    # Async (not yet used in main flow)
    enable_async_mode: bool = False
    notify_on_complete: bool = True

    def get_model_for_node(self, node_name: str) -> str:
        """Look up which model a graph node should use."""
        routing_dict = dict(self.model_routing)
        return routing_dict.get(node_name, "gpt-4o-mini")

    def get_query_complexity(self, query: str) -> str:
        """Classify query as simple/medium/complex using keyword matching
        and word count as a rough heuristic. The planner can override this
        with the LLM-based analysis."""
        query_lower = query.lower()

        for keyword in self.complex_query_keywords:
            if keyword in query_lower:
                return "complex"

        for keyword in self.simple_query_keywords:
            if query_lower.startswith(keyword):
                return "simple"

        word_count = len(query.split())
        if word_count <= 6:
            return "simple"
        elif word_count >= 15:
            return "complex"

        return "medium"


V7Config = V8Config  # backwards compat
