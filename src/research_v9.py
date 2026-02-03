#!/usr/bin/env python3
# src/research_v9.py
"""
Research Studio v9 - Deep Research Agent

A truly general research agent with:
- Human-in-the-loop clarification for ambiguous queries
- Knowledge gap detection with iterative research
- Citation-grounded results with verification

NO hardcoded query types. Works for ANY research question.

Usage:
    python -m src.research_v9 "Your research question"
    python -m src.research_v9 --simple "Skip clarification"
    python -m src.research_v9 --help

Based on architectures from OpenAI, Anthropic, Google, and Perplexity.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from src.agents.deep_researcher import DeepResearchAgent


def print_banner():
    """Print the startup banner."""
    print("\n" + "=" * 60)
    print("RESEARCH STUDIO v9 - Deep Research Agent")
    print("=" * 60)


def print_metrics(result: dict, elapsed: float):
    """Print research quality metrics."""
    meta = result.get("metadata", {})

    print("\n" + "-" * 60)
    print("RESEARCH METRICS")
    print("-" * 60)
    print(f"  Sources: {meta.get('sources_count', 0)}")
    print(f"  Evidence Items: {meta.get('evidence_count', 0)}")
    print(f"  Research Iterations: {meta.get('iterations', 0)}")
    print(f"  Coverage: {meta.get('coverage', 0):.0%}")
    print(f"  Confidence: {meta.get('confidence', 0):.0%}")

    gaps = result.get("gaps_remaining", [])
    if gaps:
        print(f"  Remaining Gaps: {len(gaps)}")

    print(f"  Time Elapsed: {elapsed:.1f}s")


def run_research(
    question: str,
    skip_clarification: bool = False,
    max_iterations: int = 5,
    max_sources: int = 15,
    verbose: bool = True,
    output_file: str = None,
) -> dict:
    """
    Run the v9 research pipeline.

    Args:
        question: The research question
        skip_clarification: If True, skip human clarification
        max_iterations: Maximum research iterations
        max_sources: Maximum sources to consult
        verbose: If True, print progress updates
        output_file: Optional file to save report

    Returns:
        Research result dict
    """
    start_time = time.time()

    if verbose:
        print_banner()
        print(f"\nQuery: {question}")
        print(f"Mode: {'Auto' if skip_clarification else 'Interactive'}")
        print("-" * 60)

    # Create clarification callback
    def no_clarify_callback(q, options):
        """Auto-select first option."""
        if verbose:
            print(f"\n[AUTO] Clarification needed: {q}")
            print(f"[AUTO] Auto-selecting: {options[0] if options else 'default'}")
        return options[0] if options else ""

    # Build agent
    agent = DeepResearchAgent(
        max_iterations=max_iterations,
        max_sources=max_sources,
        verbose=verbose,
        clarification_callback=no_clarify_callback if skip_clarification else None,
    )

    # Run research
    result = agent.research(question)

    elapsed = time.time() - start_time

    # Print report
    if verbose:
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(result["report"])
        print_metrics(result, elapsed)
        print("\n" + "=" * 60)
        print(f"Completed in {elapsed:.1f}s")

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(result["report"])
        if verbose:
            print(f"\nReport saved to: {output_file}")

    return result


def print_help():
    """Print help message."""
    print("""
Research Studio v9 - Deep Research Agent

A general research agent that works for ANY query type.
No hardcoded patterns. Just pure LLM reasoning.

Usage:
    python -m src.research_v9 [OPTIONS] [QUESTION]

Options:
    --simple, -s     Skip human clarification (auto-proceed)
    --quiet, -q      Minimal output (just the report)
    --output FILE    Save report to file
    -o FILE          Save report to file (shorthand)
    --iterations N   Max research iterations (default: 5)
    --sources N      Max sources to consult (default: 15)
    --help, -h       Show this help message

Examples:
    python -m src.research_v9 "What are Trump's immigration policies?"
    python -m src.research_v9 --simple "Compare React vs Vue"
    python -m src.research_v9 "Quantum computing" -o report.md
    python -m src.research_v9 --iterations 3 "Latest AI developments"

Features:
    - Asks for clarification when query is ambiguous
    - Detects knowledge gaps and researches until comprehensive
    - Every claim is grounded in cited sources
""")


def main():
    """Main entry point for v9 research."""
    load_dotenv()

    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    skip_clarification = False
    verbose = True
    output_file = None
    max_iterations = 5
    max_sources = 15

    # Process flags
    if "--help" in args or "-h" in args:
        print_help()
        return

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

    # Process --output or -o
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_file = args[idx + 1]
            args.pop(idx)  # Remove --output
            args.pop(idx)  # Remove filename

    if "-o" in args:
        idx = args.index("-o")
        if idx + 1 < len(args):
            output_file = args[idx + 1]
            args.pop(idx)
            args.pop(idx)

    # Process --iterations
    if "--iterations" in args:
        idx = args.index("--iterations")
        if idx + 1 < len(args):
            max_iterations = int(args[idx + 1])
            args.pop(idx)
            args.pop(idx)

    # Process --sources
    if "--sources" in args:
        idx = args.index("--sources")
        if idx + 1 < len(args):
            max_sources = int(args[idx + 1])
            args.pop(idx)
            args.pop(idx)

    # Get question from remaining args or prompt
    if args:
        question = " ".join(args)
    else:
        if verbose:
            print_banner()
        question = input("\nEnter your research question: ").strip()

    if not question:
        print("Error: No question provided.")
        print("Use --help for usage information.")
        sys.exit(1)

    # Run research
    run_research(
        question=question,
        skip_clarification=skip_clarification,
        max_iterations=max_iterations,
        max_sources=max_sources,
        verbose=verbose,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
