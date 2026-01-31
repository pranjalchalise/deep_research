# src/cli.py
"""
CLI for the research pipeline with human-in-the-loop support.

Usage:
    python -m src.cli "Your research question here"

The pipeline will:
1. Analyze your query
2. Search for entity candidates
3. If ambiguous, ask for clarification (human-in-the-loop)
4. Conduct deep research with anchored queries
5. Generate a grounded report with citations
"""
from __future__ import annotations

import sys
import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.core.graph import build_graph_with_memory, build_simple_graph
from src.core.config import V7Config


def get_state_value(state: dict, key: str, default=None):
    """Safely get a value from state."""
    return state.get(key, default)


def run_with_clarification(question: str):
    """
    Run the research pipeline with human-in-the-loop clarification.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    cfg = V7Config()

    # Build graph with memory (required for interrupt)
    app, checkpointer = build_graph_with_memory(interrupt_on_clarify=True)

    # Thread ID for this conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state
    init = {
        "messages": [HumanMessage(content=question)],
        "round": 0,
        "max_rounds": cfg.max_rounds,
        "tavily_max_results": cfg.tavily_max_results,
        "rerank_top_n": cfg.rerank_top_n,
        "select_sources_k": cfg.select_sources_k,
        "chunk_chars": cfg.chunk_chars,
        "chunk_overlap": cfg.chunk_overlap,
        "max_chunks_per_source": cfg.max_chunks_per_source,
        "evidence_per_source": cfg.evidence_per_source,
        "min_source_score_for_strong": cfg.min_source_score_for_strong,
        "require_min_sources": cfg.require_min_sources,
        "fast_mode": cfg.fast_mode,
        "request_timeout_s": cfg.request_timeout_s,
        "cache_dir": cfg.cache_dir,
        "use_cache": cfg.use_cache,
    }

    print("\n" + "="*60)
    print("RESEARCH STUDIO v7 - Human-in-the-Loop")
    print("="*60)
    print(f"\nQuery: {question}")
    print("\n[1/5] Analyzing query...")

    # First invocation - runs until interrupt (clarify node) or completion
    result = app.invoke(init, config)

    # Check if we hit the interrupt (needs clarification)
    snapshot = app.get_state(config)

    if snapshot.next and "clarify" in snapshot.next:
        # We're at the clarify node - need human input
        state = snapshot.values
        clarification_request = get_state_value(state, "clarification_request", "")
        discovery = get_state_value(state, "discovery", {})
        candidates = discovery.get("entity_candidates", [])
        confidence = discovery.get("confidence", 0)

        print(f"\n[2/5] Discovery complete (confidence: {confidence:.0%})")
        print("\n" + "-"*60)
        print("CLARIFICATION NEEDED")
        print("-"*60)

        if clarification_request:
            print(f"\n{clarification_request}")
        else:
            print("\nI found some ambiguity in your query.")
            if candidates:
                print("\nPossible matches:")
                for i, c in enumerate(candidates[:4]):
                    print(f"  {i+1}. {c.get('name', 'Unknown')} - {c.get('description', '')[:100]}")

        # Get human input
        print("\n" + "-"*60)
        human_response = input("Your response (or press Enter to continue with best guess): ").strip()

        # Resume with human clarification
        print("\n[3/5] Processing clarification...")

        # Update state with human response and resume
        app.update_state(config, {"human_clarification": human_response})

        # Continue execution
        result = app.invoke(None, config)

    else:
        # High confidence - no clarification needed
        print("\n[2/5] Entity identified with high confidence")
        print("[3/5] Skipping clarification...")

    print("[4/5] Conducting deep research...")
    print("[5/5] Generating report...\n")

    # Get final result
    final_state = app.get_state(config)
    report = get_state_value(final_state.values, "report", "")

    if not report and final_state.values.get("messages"):
        report = final_state.values["messages"][-1].content

    print("="*60)
    print("RESEARCH REPORT")
    print("="*60)
    print(report)
    print("\n" + "="*60)


def run_simple(question: str):
    """
    Run the research pipeline without human-in-the-loop (auto-proceeds).
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    cfg = V7Config()

    # Build simple graph (no interrupt)
    app = build_simple_graph()

    init = {
        "messages": [HumanMessage(content=question)],
        "round": 0,
        "max_rounds": cfg.max_rounds,
        "tavily_max_results": cfg.tavily_max_results,
        "rerank_top_n": cfg.rerank_top_n,
        "select_sources_k": cfg.select_sources_k,
        "chunk_chars": cfg.chunk_chars,
        "chunk_overlap": cfg.chunk_overlap,
        "max_chunks_per_source": cfg.max_chunks_per_source,
        "evidence_per_source": cfg.evidence_per_source,
        "min_source_score_for_strong": cfg.min_source_score_for_strong,
        "require_min_sources": cfg.require_min_sources,
        "fast_mode": cfg.fast_mode,
        "request_timeout_s": cfg.request_timeout_s,
        "cache_dir": cfg.cache_dir,
        "use_cache": cfg.use_cache,
    }

    print(f"\nResearching: {question}\n")

    result = app.invoke(init)
    print(result["messages"][-1].content)


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.cli "your research question"')
        print('       python -m src.cli --simple "your question"  # Skip clarification')
        raise SystemExit(2)

    # Check for --simple flag
    if sys.argv[1] == "--simple":
        if len(sys.argv) < 3:
            print('Usage: python -m src.cli --simple "your question"')
            raise SystemExit(2)
        question = " ".join(sys.argv[2:])
        run_simple(question)
    else:
        question = " ".join(sys.argv[1:])
        run_with_clarification(question)


if __name__ == "__main__":
    main()
