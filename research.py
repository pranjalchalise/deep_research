#!/usr/bin/env python3
"""
Research Studio - General Deep Research Agent

A truly general research agent that works for ANY query.
No hardcoded query types. No templates. Just ReAct reasoning.

Usage:
    python research.py "Your research question here"
    python research.py "What are Trump's immigration policies?" --iterations 3
    python research.py "Compare React vs Vue" -o report.md

Based on architectures from OpenAI, Anthropic, Google, and Perplexity.
"""
import argparse
import os
import sys

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY=your-key")
    sys.exit(1)

from src.agents.researcher import research


def main():
    parser = argparse.ArgumentParser(
        description="General Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python research.py "What are the latest AI regulations in the EU?"
    python research.py "Deep research about quantum computing applications"
    python research.py "Compare PostgreSQL vs MongoDB" --iterations 3
    python research.py "Tell me about Elon Musk's AI safety views" -o report.md
        """
    )

    parser.add_argument("query", help="Your research question")
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Max research iterations (default: 5)"
    )
    parser.add_argument(
        "--sources", "-s",
        type=int,
        default=15,
        help="Max sources to use (default: 15)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save report to file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    # Run research
    result = research(
        query=args.query,
        max_iterations=args.iterations,
        max_sources=args.sources,
        verbose=not args.quiet
    )

    # Output
    print("\n" + "="*60)
    print("RESEARCH COMPLETE")
    print("="*60)
    print(f"Sources: {len(result['sources'])}")
    print(f"Evidence: {result['evidence_count']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Confidence: {result['confidence']:.0%}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result["report"])
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print("REPORT")
        print("="*60)
        print(result["report"])


if __name__ == "__main__":
    main()
