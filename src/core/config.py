from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class V8Config:
    """
    Research Studio v8 Configuration (Optimized).

    Features:
    - Iterative research with gap detection
    - Multi-agent orchestrator-worker architecture
    - Source credibility scoring (E-E-A-T)
    - Span-level citation verification
    - Confidence scoring per claim
    - Backtracking on dead ends

    Optimizations:
    - Query deduplication (reduces Tavily calls ~30%)
    - Batched trust engine (reduces LLM calls ~6)
    - Complexity-based routing (fast path for simple queries)
    - Model tiering (40% cost reduction)
    - Early termination on diminishing returns
    """

    # === Planning / Iteration ===
    max_rounds: int = 2
    queries_per_round: int = 10

    # === Iterative Research ===
    max_research_iterations: int = 2           # Reduced from 3
    min_confidence_to_proceed: float = 0.7     # Below this, do another iteration
    max_research_time_minutes: float = 10.0    # Hard timeout
    enable_backtracking: bool = True           # Pivot on dead ends

    # === Multi-Agent ===
    use_multi_agent: bool = True               # Enable orchestrator pattern
    max_subagents: int = 3                     # Reduced from 5 for efficiency
    subagent_model: str = "gpt-4o-mini"        # Cheaper model for workers
    orchestrator_model: str = "gpt-4o"         # Smarter model for lead
    subagent_max_iterations: int = 2           # Internal loops per subagent

    # === OPTIMIZATION: Model Tiering ===
    # Route different models per node for cost efficiency
    model_routing: Tuple[Tuple[str, str], ...] = (
        ("analyzer", "gpt-4o-mini"),
        ("discovery", "gpt-4o-mini"),
        ("planner", "gpt-4o"),                 # Critical - keep smart
        ("orchestrator", "gpt-4o-mini"),       # Decomposition is templated
        ("subagent", "gpt-4o-mini"),
        ("claims", "gpt-4o-mini"),
        ("cite", "gpt-4o-mini"),
        ("credibility", "gpt-4o-mini"),
        ("writer", "gpt-4o"),                  # Critical - keep smart
    )

    # === OPTIMIZATION: Complexity Routing ===
    enable_complexity_routing: bool = True     # Route by query complexity
    simple_query_keywords: Tuple[str, ...] = (
        "what is", "who is", "define", "when was", "where is",
    )
    complex_query_keywords: Tuple[str, ...] = (
        "compare", "analyze", "evaluate", "research", "investigate",
        "comprehensive", "in-depth", "detailed analysis",
    )

    # === OPTIMIZATION: Query Deduplication ===
    enable_query_dedup: bool = True            # Deduplicate similar queries
    query_similarity_threshold: float = 0.85   # Cosine similarity threshold

    # === OPTIMIZATION: Batched Processing ===
    batch_trust_engine: bool = True            # Combine verification steps
    batch_claims_credibility: bool = True      # Combine claims + credibility

    # === OPTIMIZATION: Early Termination ===
    enable_early_termination: bool = True      # Stop on diminishing returns
    min_confidence_delta: float = 0.05         # Stop if improvement < 5%
    min_new_sources_threshold: int = 2         # Stop if < 2 new sources found
    cost_budget_usd: float = 0.50              # Stop approaching budget

    # === Retrieval ===
    tavily_max_results: int = 6
    rerank_top_n: int = 15                     # Increased from 10
    select_sources_k: int = 10                 # Increased from 8

    # === Reading ===
    chunk_chars: int = 3500
    chunk_overlap: int = 350
    max_chunks_per_source: int = 4
    evidence_per_source: int = 3

    # === Source Credibility (NEW) ===
    enable_credibility_scoring: bool = True
    min_source_credibility: float = 0.35       # Filter below this

    # Trusted domains get higher scores
    trusted_domains: tuple = (
        ".edu", ".gov", ".org",
        "arxiv.org", "nature.com", "science.org",
        "wikipedia.org", "github.com", "scholar.google.com",
        "ncbi.nlm.nih.gov", "ieee.org", "acm.org",
    )

    # Lower trust domains
    low_trust_domains: tuple = (
        "pinterest.com", "quora.com",
        "buzzfeed.com", "wikihow.com",
    )

    # === Citation Verification (NEW) ===
    enable_span_verification: bool = True      # Match claims to exact spans
    enable_cross_validation: bool = True       # Check multiple sources agree
    span_match_threshold: float = 0.6          # Min similarity for span match
    cross_validation_threshold: int = 2        # Min sources to cross-validate
    hallucination_threshold: float = 0.3       # Flag if >30% unsupported

    # === Confidence Scoring (NEW) ===
    enable_confidence_scoring: bool = True
    high_confidence_threshold: float = 0.8     # ✓✓ indicator
    medium_confidence_threshold: float = 0.6   # ✓ indicator

    # === Trust / Termination ===
    min_source_score_for_strong: float = 0.55
    require_min_sources: int = 4
    allow_news: bool = True

    # === Runtime ===
    fast_mode: bool = False                    # Skip cross_validate, reduce iterations
    request_timeout_s: float = 15.0            # Increased from 12
    cache_dir: str = ".cache_v8"
    use_cache: bool = True

    # === Async Mode ===
    enable_async_mode: bool = False            # Background operation
    notify_on_complete: bool = True            # Callback when done

    def get_model_for_node(self, node_name: str) -> str:
        """Get the configured model for a specific node."""
        routing_dict = dict(self.model_routing)
        return routing_dict.get(node_name, "gpt-4o-mini")

    def get_query_complexity(self, query: str) -> str:
        """
        Determine query complexity for routing.
        Returns: 'simple', 'medium', or 'complex'
        """
        query_lower = query.lower()

        # Check for complex indicators
        for keyword in self.complex_query_keywords:
            if keyword in query_lower:
                return "complex"

        # Check for simple indicators
        for keyword in self.simple_query_keywords:
            if query_lower.startswith(keyword):
                return "simple"

        # Word count heuristic
        word_count = len(query.split())
        if word_count <= 6:
            return "simple"
        elif word_count >= 15:
            return "complex"

        return "medium"


# Backwards compatibility alias
V7Config = V8Config
