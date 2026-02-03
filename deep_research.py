#!/usr/bin/env python3
"""
Deep Research - Full-Featured General Research Agent

A truly general research agent with:
- HUMAN-IN-THE-LOOP: Asks for clarification when query is ambiguous
- KNOWLEDGE GAP DETECTION: Iteratively fills gaps until comprehensive
- CITATION-GROUNDED: Every claim backed by verifiable sources

Usage:
    python deep_research.py "Your research question here"
    python deep_research.py "Trump's immigration policies" --iterations 5
    python deep_research.py "Compare React vs Vue" -o report.md
    python deep_research.py --interactive

Based on architectures from OpenAI, Anthropic, Google, and Perplexity.
NO hardcoded query types. Works for ANY research question.
"""
import argparse
import os
import sys
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY=your-key")
    print("Or add it to .env file")
    sys.exit(1)

from src.agents.deep_researcher import deep_research, DeepResearchAgent


def interactive_mode(agent: DeepResearchAgent):
    """Run in interactive mode - multiple queries in one session."""
    print("\n" + "="*60)
    print("DEEP RESEARCH - INTERACTIVE MODE")
    print("="*60)
    print("\nCommands:")
    print("  /quit    - Exit")
    print("  /sources - Show sources from last research")
    print("  /gaps    - Show knowledge gaps from last research")
    print("-"*60)

    last_result = None

    while True:
        try:
            query = input("\n> Enter research query: ").strip()

            if not query:
                continue

            if query.lower() == "/quit":
                print("Goodbye!")
                break

            if query.lower() == "/sources" and last_result:
                print("\n--- Sources ---")
                for url, info in last_result.get("sources", {}).items():
                    print(f"  - {info['title']}")
                    print(f"    {url}")
                    print(f"    Evidence items: {info['evidence_count']}")
                continue

            if query.lower() == "/gaps" and last_result:
                print("\n--- Knowledge Gaps ---")
                for gap in last_result.get("gaps_remaining", []):
                    print(f"  - {gap}")
                if not last_result.get("gaps_remaining"):
                    print("  No significant gaps detected.")
                continue

            # Run research
            last_result = agent.research(query)

            print("\n" + "="*60)
            print("REPORT")
            print("="*60)
            print(last_result["report"])

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - General Purpose Research with HITL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python deep_research.py "What are Trump's new immigration policies?"
    python deep_research.py "Deep research about quantum computing applications"
    python deep_research.py "Compare PostgreSQL vs MongoDB" --iterations 5
    python deep_research.py "Elon Musk's views on AI safety" -o report.md
    python deep_research.py --interactive

Features:
    - Asks for clarification when query is ambiguous
    - Detects knowledge gaps and researches until comprehensive
    - Every claim is grounded in cited sources
    - Works for ANY research question (no hardcoded patterns)
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Your research question (optional if using --interactive)"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Max research iterations (default: 5)"
    )

    parser.add_argument(
        "--sources", "-s",
        type=int,
        default=20,
        help="Max sources to consult (default: 20)"
    )

    parser.add_argument(
        "--coverage", "-c",
        type=float,
        default=0.7,
        help="Min coverage threshold 0-1 (default: 0.7)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Save report to file (markdown)"
    )

    parser.add_argument(
        "--output-json",
        help="Save full results to JSON file"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )

    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip clarification (don't ask questions)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Main model (default: gpt-4o)"
    )

    parser.add_argument(
        "--fast-model",
        default="gpt-4o-mini",
        help="Fast model for extraction (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    # Create clarification callback
    def no_clarify_callback(question, options):
        """Skip clarification - just use first option or query as-is."""
        return options[0] if options else ""

    # Build agent
    agent = DeepResearchAgent(
        model=args.model,
        fast_model=args.fast_model,
        max_iterations=args.iterations,
        max_sources=args.sources,
        min_coverage=args.coverage,
        verbose=not args.quiet,
        clarification_callback=no_clarify_callback if args.no_clarify else None
    )

    # Interactive mode
    if args.interactive:
        interactive_mode(agent)
        return

    # Need a query for non-interactive mode
    if not args.query:
        parser.print_help()
        print("\nError: Please provide a query or use --interactive mode")
        sys.exit(1)

    # Run research
    result = agent.research(args.query)

    # Output report
    if not args.quiet:
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        print(result["report"])

    # Save markdown report
    if args.output:
        with open(args.output, "w") as f:
            f.write(result["report"])
            f.write("\n\n---\n\n## Research Metadata\n\n")
            f.write(f"- **Query**: {result['query']}\n")
            if result['clarified_query'] != result['query']:
                f.write(f"- **Clarified Query**: {result['clarified_query']}\n")
            f.write(f"- **Sources**: {result['metadata']['sources_count']}\n")
            f.write(f"- **Evidence Items**: {result['metadata']['evidence_count']}\n")
            f.write(f"- **Research Iterations**: {result['metadata']['iterations']}\n")
            f.write(f"- **Coverage**: {result['metadata']['coverage']:.0%}\n")
            f.write(f"- **Confidence**: {result['metadata']['confidence']:.0%}\n")

            if result.get('gaps_remaining'):
                f.write(f"\n### Remaining Knowledge Gaps\n\n")
                for gap in result['gaps_remaining']:
                    f.write(f"- {gap}\n")

        print(f"\nReport saved to: {args.output}")

    # Save JSON results
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full results saved to: {args.output_json}")

    # Print summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"Query: {result['query']}")
    if result['clarified_query'] != result['query']:
        print(f"Clarified: {result['clarified_query']}")
    print(f"Sources: {result['metadata']['sources_count']}")
    print(f"Evidence: {result['metadata']['evidence_count']}")
    print(f"Iterations: {result['metadata']['iterations']}")
    print(f"Coverage: {result['metadata']['coverage']:.0%}")
    print(f"Confidence: {result['metadata']['confidence']:.0%}")

    if result.get('gaps_remaining'):
        print(f"Remaining gaps: {len(result['gaps_remaining'])}")


if __name__ == "__main__":
    main()
