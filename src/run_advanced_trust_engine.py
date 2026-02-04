#!/usr/bin/env python3
"""
CLI for the advanced research pipeline with trust engine.

This is the heavy-duty pipeline. It runs a full 7-node trust engine on top
of the research: credibility scoring, span verification, cross-validation,
dead-end backtracking, and complexity-based routing.

Use this when accuracy matters more than speed. For quick research,
use ``python -m src.run`` instead.

Examples:
    python -m src.run_advanced_trust_engine "Who is Satya Nadella?"
    python -m src.run_advanced_trust_engine --single-agent "Compare Python vs JavaScript"
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.advanced.graph import (
    build_trust_engine_graph_with_memory,
    build_trust_engine_simple_graph,
)
from src.advanced.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(state: dict, key: str, default=None):
    """Safe state accessor — saves us from KeyError everywhere."""
    return state.get(key, default)


def _print_metrics(metadata: Dict[str, Any]) -> None:
    """Print the quality metrics table after a run."""
    print("\n" + "-" * 60)
    print("RESEARCH QUALITY METRICS")
    print("-" * 60)
    print(f"  Overall Confidence: {metadata.get('overall_confidence', 0):.0%}")
    print(f"  Verified Claims:    {metadata.get('verified_claims', 0)}/{metadata.get('total_claims', 0)}")
    print(f"  Knowledge Gaps:     {metadata.get('knowledge_gaps', 0)}")
    print(f"  Sources Used:       {metadata.get('sources_used', 0)}")
    print(f"  Iterations:         {metadata.get('research_iterations', 1)}")
    print(f"  Total Searches:     {metadata.get('total_searches', 0)}")
    elapsed = metadata.get("time_elapsed_seconds", 0)
    if elapsed > 0:
        print(f"  Time Elapsed:       {elapsed:.1f}s")


def _print_subagent_progress(state: Dict) -> None:
    """Show how many sub-agents have finished so far."""
    assignments = state.get("subagent_assignments") or []
    findings = state.get("subagent_findings") or []
    if not assignments:
        return

    print(f"\n  Subagents: {len(findings)}/{len(assignments)} complete")
    for f in findings:
        sid = f.get("subagent_id", "?")
        conf = f.get("confidence", 0)
        q = f.get("question", "")[:40]
        mark = "+" if conf >= 0.7 else "~"
        print(f"    {mark} {sid}: {q}... ({conf:.0%})")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_advanced(
    question: str,
    skip_clarification: bool = False,
    use_multi_agent: bool = True,
    verbose: bool = True,
) -> str:
    """Run the advanced trust-engine pipeline. Returns the final report."""
    cfg = ResearchConfig()
    start_time = time.time()

    # Two modes: with HITL clarification (needs checkpointer) or without
    if skip_clarification:
        app = build_trust_engine_simple_graph(use_multi_agent=use_multi_agent)
        config = {}
    else:
        app, _ = build_trust_engine_graph_with_memory(
            interrupt_on_clarify=True,
            use_multi_agent=use_multi_agent,
        )
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Seed the graph with config values so every node can read them
    init = {
        "messages": [HumanMessage(content=question)],
        "original_query": question,
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
        "max_research_iterations": cfg.max_research_iterations,
        "min_confidence_to_proceed": cfg.min_confidence_to_proceed,
        "enable_backtracking": cfg.enable_backtracking,
        "max_subagents": cfg.max_subagents,
        "subagent_max_iterations": cfg.subagent_max_iterations,
        "enable_credibility_scoring": cfg.enable_credibility_scoring,
        "min_source_credibility": cfg.min_source_credibility,
        "enable_span_verification": cfg.enable_span_verification,
        "enable_cross_validation": cfg.enable_cross_validation,
        "span_match_threshold": cfg.span_match_threshold,
        "cross_validation_threshold": cfg.cross_validation_threshold,
        "high_confidence_threshold": cfg.high_confidence_threshold,
        "medium_confidence_threshold": cfg.medium_confidence_threshold,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("RESEARCH STUDIO — Trust Engine Pipeline")
        print("=" * 60)
        mode = "Multi-Agent" if use_multi_agent else "Single-Agent"
        print(f"\nQuery: {question}")
        print(f"Mode: {mode} | Max Iterations: {cfg.max_research_iterations}")
        print("\n[1/7] Analyzing query...")

    result = app.invoke(init, config) if config else app.invoke(init)

    # If we're using HITL, check whether the graph paused for clarification
    if not skip_clarification:
        snapshot = app.get_state(config)

        if snapshot.next and "clarify" in snapshot.next:
            state = snapshot.values
            clarification = _get(state, "clarification_request", "")
            discovery = _get(state, "discovery", {})
            candidates = discovery.get("entity_candidates", [])
            confidence = discovery.get("confidence", 0)

            if verbose:
                print(f"\n[2/7] Discovery complete (confidence: {confidence:.0%})")
                print("\n" + "-" * 60)
                print("CLARIFICATION NEEDED")
                print("-" * 60)
                if clarification:
                    print(f"\n{clarification}")
                else:
                    print("\nAmbiguity detected in your query.")
                    if candidates:
                        print("\nPossible matches:")
                        for i, c in enumerate(candidates[:5]):
                            desc = c.get("description", "")[:80]
                            conf = c.get("confidence", 0)
                            print(f"  {i+1}. {c.get('name', 'Unknown')} ({conf:.0%})")
                            if desc:
                                print(f"      {desc}")
                print("\n" + "-" * 60)

            human_response = input("Your response (or Enter for best guess): ").strip()

            if verbose:
                print("\n[3/7] Processing clarification...")

            app.update_state(config, {"human_clarification": human_response})
            result = app.invoke(None, config)
        else:
            if verbose:
                print("\n[2/7] Entity identified with high confidence")
                print("[3/7] Skipping clarification...")

    if verbose:
        print("[4/7] Planning research strategy...")
        print("[5/7] Conducting research...")

        # Try to show sub-agent progress if available
        if not skip_clarification:
            try:
                interim = app.get_state(config)
                if interim and interim.values:
                    _print_subagent_progress(interim.values)
            except Exception:
                pass

        print("[6/7] Verifying sources and claims...")
        print("[7/7] Generating report...\n")

    # Pull the report out of the final graph state
    if skip_clarification:
        report = result.get("report", "")
        metadata = result.get("research_metadata", {})
        if not report and result.get("messages"):
            report = result["messages"][-1].content
    else:
        final = app.get_state(config)
        report = _get(final.values, "report", "")
        metadata = _get(final.values, "research_metadata", {})
        if not report and final.values.get("messages"):
            report = final.values["messages"][-1].content

    elapsed = time.time() - start_time
    metadata["time_elapsed_seconds"] = elapsed

    if verbose:
        print("=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(report)
        if metadata:
            _print_metrics(metadata)
        print("\n" + "=" * 60)
        print(f"Completed in {elapsed:.1f}s")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    args = sys.argv[1:]
    use_multi_agent = True

    if "--help" in args or "-h" in args:
        print("""
Research Studio — Trust Engine Pipeline

This is the advanced pipeline with a full trust engine: credibility
scoring, span verification, cross-validation, and dead-end backtracking.

Usage:
    python -m src.run_advanced_trust_engine [OPTIONS] [QUESTION]

Options:
    --single-agent  Use single-agent mode instead of multi-agent
    --help, -h      Show this help message

Examples:
    python -m src.run_advanced_trust_engine "Who is Satya Nadella?"
    python -m src.run_advanced_trust_engine --single-agent "Compare Python vs JavaScript"
""")
        return

    if "--single-agent" in args:
        args.remove("--single-agent")
        use_multi_agent = False

    question = " ".join(args) if args else input("Enter your research question: ").strip()

    if not question:
        print("Error: No question provided.")
        sys.exit(1)

    run_advanced(question, skip_clarification=False, use_multi_agent=use_multi_agent)


if __name__ == "__main__":
    main()
