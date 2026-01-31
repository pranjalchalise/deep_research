from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class V7Config:
    # planning / iteration
    max_rounds: int = 2
    queries_per_round: int = 10

    # retrieval
    tavily_max_results: int = 6
    rerank_top_n: int = 10  # how many candidates to consider after search merge
    select_sources_k: int = 8  # how many sources to fully fetch/read

    # reading
    chunk_chars: int = 3500
    chunk_overlap: int = 350
    max_chunks_per_source: int = 4
    evidence_per_source: int = 3

    # trust / termination
    min_source_score_for_strong: float = 0.55
    require_min_sources: int = 4
    allow_news: bool = True

    # runtime
    fast_mode: bool = True      # if True: fewer chunks, smaller evidence extraction
    request_timeout_s: float = 12.0
    cache_dir: str = ".cache_v7"
    use_cache: bool = True      # cache search results to avoid redundant API calls
