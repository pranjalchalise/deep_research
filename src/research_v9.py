#!/usr/bin/env python3
# src/research_v9.py
"""
Research Studio v9 - Deep Research Agent

Modes:
    Single-agent: Sequential research with gap detection
    Multi-agent:  Anthropic-style orchestrator-workers (parallel)

Usage:
    python -m src.research_v9 "Your question"
    python -m src.research_v9 --multi "Your question"
    python -m src.research_v9 --compare "Your question"
    python -m src.research_v9 --help
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from src.agents.deep_researcher import DeepResearchAgent
from src.agents.multi_agent_researcher import MultiAgentResearcher


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_banner(mode: str = "Single-Agent"):
    print("\n" + "=" * 60)
    print("RESEARCH STUDIO v9")
    print("=" * 60)


def print_single_metrics(result: dict, elapsed: float):
    meta = result.get("metadata", {})
    print("\n" + "-" * 60)
    print("METRICS (Single-Agent)")
    print("-" * 60)
    print(f"  Sources:     {meta.get('sources_count', 0)}")
    print(f"  Evidence:    {meta.get('evidence_count', 0)}")
    print(f"  Iterations:  {meta.get('iterations', 0)}")
    print(f"  Coverage:    {meta.get('coverage', 0):.0%}")
    print(f"  Confidence:  {meta.get('confidence', 0):.0%}")
    print(f"  Time:        {elapsed:.1f}s")


def print_multi_metrics(result: dict):
    m = result.get("metrics", {})
    print("\n" + "-" * 60)
    print("METRICS (Multi-Agent)")
    print("-" * 60)
    print(f"  Workers:             {m.get('num_workers', 0)}")
    print(f"  Sources:             {m.get('total_sources', 0)}")
    print(f"  Evidence:            {m.get('total_evidence', 0)}")
    print(f"  Confidence:          {m.get('confidence', 0):.0%}")
    print(f"  Total Time:          {m.get('total_time', 0):.1f}s")
    print(f"  Parallel Time:       {m.get('parallel_research_time', 0):.1f}s (wall clock)")
    print(f"  Sequential Estimate: {m.get('sequential_estimate', 0):.1f}s")
    print(f"  Speedup:             {m.get('speedup_factor', 1):.2f}x")
    print(f"  Time Saved:          {m.get('time_saved_seconds', 0):.1f}s ({m.get('time_saved_percent', 0):.0f}%)")

    # Worker breakdown
    workers = result.get("worker_results", [])
    if workers:
        print("\n  Worker Breakdown:")
        for w in workers:
            print(f"    {w['worker_id']}: {w['question'][:45]}...")
            print(f"           {w['evidence_count']} evidence, {w['confidence']:.0%} conf, {w['time_taken']:.1f}s")


def print_comparison(single_result: dict, single_time: float, multi_result: dict):
    """Print side-by-side comparison."""
    s_meta = single_result.get("metadata", {})
    m = multi_result.get("metrics", {})

    print("\n" + "=" * 60)
    print("COMPARISON: Single-Agent vs Multi-Agent")
    print("=" * 60)
    print(f"{'Metric':<25} {'Single':<15} {'Multi':<15} {'Winner':<10}")
    print("-" * 65)

    # Time
    s_time = single_time
    m_time = m.get("total_time", 0)
    winner = "Multi" if m_time < s_time else "Single"
    print(f"{'Time':<25} {s_time:<15.1f} {m_time:<15.1f} {winner:<10}")

    # Sources
    s_src = s_meta.get("sources_count", 0)
    m_src = m.get("total_sources", 0)
    winner = "Multi" if m_src > s_src else ("Single" if s_src > m_src else "Tie")
    print(f"{'Sources':<25} {s_src:<15} {m_src:<15} {winner:<10}")

    # Evidence
    s_ev = s_meta.get("evidence_count", 0)
    m_ev = m.get("total_evidence", 0)
    winner = "Multi" if m_ev > s_ev else ("Single" if s_ev > m_ev else "Tie")
    print(f"{'Evidence':<25} {s_ev:<15} {m_ev:<15} {winner:<10}")

    # Confidence
    s_conf = s_meta.get("confidence", 0)
    m_conf = m.get("confidence", 0)
    winner = "Multi" if m_conf > s_conf else ("Single" if s_conf > m_conf else "Tie")
    print(f"{'Confidence':<25} {s_conf:<15.0%} {m_conf:<15.0%} {winner:<10}")

    # Coverage
    s_cov = s_meta.get("coverage", 0)
    m_cov = m.get("coverage", 0)
    winner = "Multi" if m_cov > s_cov else ("Single" if s_cov > m_cov else "Tie")
    print(f"{'Coverage':<25} {s_cov:<15.0%} {m_cov:<15.0%} {winner:<10}")

    # Speedup
    speedup = m.get("speedup_factor", 1)
    saved = m.get("time_saved_percent", 0)
    print("-" * 65)
    print(f"\nMulti-agent parallelism speedup: {speedup:.2f}x")
    print(f"Time saved vs sequential workers: {saved:.0f}%")

    if m_time < s_time:
        diff = s_time - m_time
        pct = (diff / s_time) * 100
        print(f"Multi-agent was {diff:.1f}s faster ({pct:.0f}% improvement over single-agent)")
    else:
        diff = m_time - s_time
        print(f"Single-agent was {diff:.1f}s faster (multi-agent overhead)")


# =============================================================================
# RUNNERS
# =============================================================================

def run_single(question: str, skip_clarification: bool, verbose: bool, output_file: str = None) -> tuple:
    """Run single-agent research. Returns (result, elapsed)."""
    start = time.time()

    def no_clarify(q, options):
        if verbose:
            print(f"[AUTO] {q}")
            print(f"[AUTO] Selecting: {options[0] if options else 'default'}")
        return options[0] if options else ""

    agent = DeepResearchAgent(
        max_iterations=5,
        max_sources=15,
        verbose=verbose,
        clarification_callback=no_clarify if skip_clarification else None,
    )

    result = agent.research(question)
    elapsed = time.time() - start

    if verbose:
        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60)
        print(result["report"])
        print_single_metrics(result, elapsed)

    if output_file:
        with open(output_file, "w") as f:
            f.write(result["report"])
        if verbose:
            print(f"\nSaved to: {output_file}")

    return result, elapsed


def run_multi(question: str, skip_clarification: bool, verbose: bool, output_file: str = None) -> dict:
    """Run multi-agent research. Returns result."""
    def no_clarify(q, options):
        if verbose:
            print(f"[AUTO] {q}")
            print(f"[AUTO] Selecting: {options[0] if options else 'default'}")
        return options[0] if options else ""

    agent = MultiAgentResearcher(
        max_workers=4,
        max_sources_per_worker=5,
        verbose=verbose,
        clarification_callback=no_clarify if skip_clarification else None,
    )

    result = agent.research(question)

    if verbose:
        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60)
        print(result["report"])
        print_multi_metrics(result)

    if output_file:
        with open(output_file, "w") as f:
            f.write(result["report"])
        if verbose:
            print(f"\nSaved to: {output_file}")

    return result


# =============================================================================
# CLI
# =============================================================================

def print_help():
    print("""
Research Studio v9 - Deep Research Agent

Usage:
    python -m src.research_v9 [OPTIONS] [QUESTION]

Modes:
    (default)        Multi-agent orchestrator-workers (parallel)
    --single-agent   Single-agent with gap detection
    --compare        Run BOTH and compare metrics side-by-side

Options:
    --simple, -s     Skip human clarification (auto-proceed)
    --quiet, -q      Minimal output
    --output FILE    Save report to file
    -o FILE          Save report to file
    --help, -h       Show this help

Examples:
    python -m src.research_v9 "What are Trump's immigration policies?"
    python -m src.research_v9 --single-agent "Compare React vs Vue"
    python -m src.research_v9 --compare "Latest AI developments"
    python -m src.research_v9 --simple "Quantum computing"
""")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    args = sys.argv[1:]

    # Flags
    skip_clarification = False
    verbose = True
    output_file = None
    mode = "multi"  # multi (default), single, compare

    if "--help" in args or "-h" in args:
        print_help()
        return

    if "--single-agent" in args:
        args.remove("--single-agent")
        mode = "single"

    if "--compare" in args:
        args.remove("--compare")
        mode = "compare"

    if "--simple" in args:
        args.remove("--simple")
        skip_clarification = True

    if "-s" in args:
        args.remove("-s")
        skip_clarification = True

    if "--quiet" in args:
        args.remove("--quiet")
        verbose = False

    if "-q" in args:
        args.remove("-q")
        verbose = False

    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_file = args[idx + 1]
            args.pop(idx)
            args.pop(idx)

    if "-o" in args:
        idx = args.index("-o")
        if idx + 1 < len(args):
            output_file = args[idx + 1]
            args.pop(idx)
            args.pop(idx)

    # Question
    if args:
        question = " ".join(args)
    else:
        print_banner()
        question = input("\nEnter your research question: ").strip()

    if not question:
        print("Error: No question provided.")
        sys.exit(1)

    if verbose:
        print_banner()
        print(f"\nQuery: {question}")
        print(f"Mode: {mode.upper()}")
        print("-" * 60)

    # Run
    if mode == "single":
        run_single(question, skip_clarification, verbose, output_file)

    elif mode == "multi":
        run_multi(question, skip_clarification, verbose, output_file)

    elif mode == "compare":
        print("\n>>> Running SINGLE-AGENT first...")
        print("-" * 60)
        single_result, single_time = run_single(question, True, verbose)

        print("\n\n>>> Running MULTI-AGENT now...")
        print("-" * 60)
        multi_result = run_multi(question, True, verbose)

        print_comparison(single_result, single_time, multi_result)


if __name__ == "__main__":
    main()
