from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class V8Config:
    """
    Research Studio v8 Configuration.

    New features:
    - Iterative research with gap detection
    - Multi-agent orchestrator-worker architecture
    - Source credibility scoring (E-E-A-T)
    - Span-level citation verification
    - Confidence scoring per claim
    - Backtracking on dead ends
    """

    # === Planning / Iteration ===
    max_rounds: int = 2
    queries_per_round: int = 10

    # === Iterative Research (NEW) ===
    max_research_iterations: int = 3           # How many refine loops
    min_confidence_to_proceed: float = 0.7     # Below this, do another iteration
    max_research_time_minutes: float = 10.0    # Hard timeout
    enable_backtracking: bool = True           # Pivot on dead ends

    # === Multi-Agent (NEW) ===
    use_multi_agent: bool = True               # Enable orchestrator pattern
    max_subagents: int = 5                     # Parallel subagents
    subagent_model: str = "gpt-4o-mini"        # Cheaper model for workers
    orchestrator_model: str = "gpt-4o"         # Smarter model for lead
    subagent_max_iterations: int = 2           # Internal loops per subagent

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
    fast_mode: bool = False                    # Disabled by default in v8
    request_timeout_s: float = 15.0            # Increased from 12
    cache_dir: str = ".cache_v8"
    use_cache: bool = True

    # === Async Mode (NEW) ===
    enable_async_mode: bool = False            # Background operation
    notify_on_complete: bool = True            # Callback when done


# Backwards compatibility alias
V7Config = V8Config
