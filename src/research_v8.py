"""
CLI entry point for v8 of the research agent. Handles argument parsing,
graph construction (with or without human-in-the-loop clarification),
and progress reporting. The actual research logic lives in src/core/graph.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.core.graph import build_v8_graph_with_memory, build_v8_simple_graph
from src.core.config import V8Config


def get_state_value(state: dict, key: str, default=None):
    return state.get(key, default)


def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_research_metadata(metadata: Dict[str, Any]) -> None:
    print("\n" + "-" * 60)
    print("RESEARCH QUALITY METRICS")
    print("-" * 60)

    overall_conf = metadata.get("overall_confidence", 0)
    print(f"  Overall Confidence: {overall_conf:.0%}")
    print(f"  Verified Claims: {metadata.get('verified_claims', 0)}/{metadata.get('total_claims', 0)}")
    print(f"  Knowledge Gaps: {metadata.get('knowledge_gaps', 0)}")
    print(f"  Sources Used: {metadata.get('sources_used', 0)}")
    print(f"  Research Iterations: {metadata.get('research_iterations', 1)}")
    print(f"  Total Searches: {metadata.get('total_searches', 0)}")

    elapsed = metadata.get("time_elapsed_seconds", 0)
    if elapsed > 0:
        print(f"  Time Elapsed: {elapsed:.1f}s")


def print_subagent_progress(state: Dict) -> None:
    assignments = state.get("subagent_assignments") or []
    findings = state.get("subagent_findings") or []

    if not assignments:
        return

    print(f"\n  Subagents: {len(findings)}/{len(assignments)} complete")

    for finding in findings:
        subagent_id = finding.get("subagent_id", "?")
        confidence = finding.get("confidence", 0)
        question = finding.get("question", "")[:40]
        status = "✓" if confidence >= 0.7 else "⚠"
        print(f"    {status} {subagent_id}: {question}... ({confidence:.0%})")


def run_research_v8(
    question: str,
    skip_clarification: bool = False,
    use_multi_agent: bool = True,
    verbose: bool = True,
) -> str:
    """Build and invoke the LangGraph research pipeline, optionally
    pausing for human clarification. Returns the final report text."""
    cfg = V8Config()
    start_time = time.time()

    if skip_clarification:
        app = build_v8_simple_graph(use_multi_agent=use_multi_agent)
        checkpointer = None
        config = {}
    else:
        app, checkpointer = build_v8_graph_with_memory(
            interrupt_on_clarify=True,
            use_multi_agent=use_multi_agent,
        )
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

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
        print("RESEARCH STUDIO v8 - Deep Research Agent")
        print("=" * 60)
        print(f"\nQuery: {question}")
        mode = "Multi-Agent" if use_multi_agent else "Single-Agent"
        print(f"Mode: {mode} | Max Iterations: {cfg.max_research_iterations}")
        print("\n[1/7] Analyzing query...")

    if skip_clarification:
        result = app.invoke(init)
    else:
        result = app.invoke(init, config)

    # If the graph paused at the clarify node, prompt the user and resume
    if not skip_clarification:
        snapshot = app.get_state(config)

        if snapshot.next and "clarify" in snapshot.next:
            state = snapshot.values
            clarification_request = get_state_value(state, "clarification_request", "")
            discovery = get_state_value(state, "discovery", {})
            candidates = discovery.get("entity_candidates", [])
            confidence = discovery.get("confidence", 0)

            if verbose:
                print(f"\n[2/7] Discovery complete (confidence: {confidence:.0%})")
                print("\n" + "-" * 60)
                print("CLARIFICATION NEEDED")
                print("-" * 60)

                if clarification_request:
                    print(f"\n{clarification_request}")
                else:
                    print("\nI found some ambiguity in your query.")
                    if candidates:
                        print("\nPossible matches:")
                        for i, c in enumerate(candidates[:5]):
                            desc = c.get('description', '')[:80]
                            conf = c.get('confidence', 0)
                            print(f"  {i+1}. {c.get('name', 'Unknown')} ({conf:.0%})")
                            if desc:
                                print(f"      {desc}")

                print("\n" + "-" * 60)

            human_response = input("Your response (or press Enter for best guess): ").strip()

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
        print("[5/7] Conducting multi-agent research...")

        if not skip_clarification:
            try:
                interim_state = app.get_state(config)
                if interim_state and interim_state.values:
                    print_subagent_progress(interim_state.values)
            except:
                pass

        print("[6/7] Verifying sources and claims...")
        print("[7/7] Generating report...\n")

    if skip_clarification:
        report = result.get("report", "")
        metadata = result.get("research_metadata", {})
        if not report and result.get("messages"):
            report = result["messages"][-1].content
    else:
        final_state = app.get_state(config)
        report = get_state_value(final_state.values, "report", "")
        metadata = get_state_value(final_state.values, "research_metadata", {})
        if not report and final_state.values.get("messages"):
            report = final_state.values["messages"][-1].content

    elapsed = time.time() - start_time
    metadata["time_elapsed_seconds"] = elapsed

    if verbose:
        print("=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(report)

        if metadata:
            print_research_metadata(metadata)

        print("\n" + "=" * 60)
        print(f"Completed in {elapsed:.1f}s")

    return report


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    args = sys.argv[1:]
    skip_clarification = False
    use_multi_agent = True

    if "--simple" in args:
        args.remove("--simple")
        skip_clarification = True

    if "--single-agent" in args:
        args.remove("--single-agent")
        use_multi_agent = False

    if "--help" in args or "-h" in args:
        print("""
Research Studio v8 - Deep Research Agent

Usage:
    python -m src.research_v8 [OPTIONS] [QUESTION]

Options:
    --simple        Skip human clarification (auto-proceed)
    --single-agent  Use single-agent mode instead of multi-agent
    --help, -h      Show this help message

Examples:
    python -m src.research_v8 "Who is Satya Nadella?"
    python -m src.research_v8 --simple "What is quantum computing?"
    python -m src.research_v8 --single-agent "Compare Python vs JavaScript"
""")
        return

    if args:
        question = " ".join(args)
    else:
        question = input("Enter your research question: ").strip()

    if not question:
        print("No question provided.")
        return

    run_research_v8(
        question,
        skip_clarification=skip_clarification,
        use_multi_agent=use_multi_agent,
    )


if __name__ == "__main__":
    main()
