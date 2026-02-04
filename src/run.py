#!/usr/bin/env python3
"""
CLI for the standard research pipeline.

Supports single-agent (iterative gap detection), multi-agent (parallel
workers via LangGraph Send), human-in-the-loop clarification, and a
--compare mode that runs both and prints metrics side-by-side.

For the advanced pipeline with trust engine, see run_advanced_trust_engine.py.

Examples:
    python -m src.run "What is quantum computing?"
    python -m src.run --single-agent "Compare React vs Vue"
    python -m src.run --compare "Latest AI developments"
"""
from __future__ import annotations

import os
import sys
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline.graph import build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_state(mode: str = "single", max_iterations: int = 5) -> dict:
    """Base state values that every invocation needs."""
    return {
        "mode": mode,
        "max_iterations": max_iterations,
        "min_coverage": 0.7,
        "iteration": 0,
        "evidence": [],
        "worker_results": [],
        "done_workers": 0,
        "searches_done": [],
        "sources": {},
    }


def _print_banner():
    print("\n" + "=" * 60)
    print("RESEARCH STUDIO")
    print("=" * 60)


def _print_metrics(result: dict, elapsed: float, label: str = ""):
    meta = result.get("metadata", {})
    heading = f"METRICS ({label})" if label else "METRICS"
    print("\n" + "-" * 60)
    print(heading)
    print("-" * 60)
    print(f"  Mode:        {meta.get('mode', '?')}")
    print(f"  Sources:     {meta.get('sources_count', 0)}")
    print(f"  Evidence:    {meta.get('evidence_count', 0)}")
    print(f"  Iterations:  {meta.get('iterations', 0)}")
    print(f"  Coverage:    {meta.get('coverage', 0):.0%}")
    print(f"  Confidence:  {meta.get('confidence', 0):.0%}")
    print(f"  Time:        {elapsed:.1f}s")


def _print_comparison(s_result: dict, s_time: float, m_result: dict, m_time: float):
    s = s_result.get("metadata", {})
    m = m_result.get("metadata", {})

    print("\n" + "=" * 60)
    print("COMPARISON: Single-Agent vs Multi-Agent")
    print("=" * 60)
    print(f"  {'Metric':<23} {'Single':<15} {'Multi':<15} {'Winner':<10}")
    print("  " + "-" * 63)

    rows = [
        ("Time (s)", s_time, m_time, "lower"),
        ("Sources", s.get("sources_count", 0), m.get("sources_count", 0), "higher"),
        ("Evidence", s.get("evidence_count", 0), m.get("evidence_count", 0), "higher"),
        ("Confidence", s.get("confidence", 0), m.get("confidence", 0), "higher"),
        ("Coverage", s.get("coverage", 0), m.get("coverage", 0), "higher"),
        ("Iterations", s.get("iterations", 0), m.get("iterations", 0), "info"),
    ]

    for label, sv, mv, direction in rows:
        if direction == "lower":
            winner = "Multi" if mv < sv else ("Single" if sv < mv else "Tie")
        elif direction == "higher":
            winner = "Multi" if mv > sv else ("Single" if sv > mv else "Tie")
        else:
            winner = ""

        # Format percentages for 0-1 metrics (not time or counts)
        if isinstance(sv, float) and sv <= 1.0 and label not in ("Time (s)",):
            print(f"  {label:<23} {sv:<15.0%} {mv:<15.0%} {winner:<10}")
        else:
            sv_fmt = f"{sv:.1f}" if isinstance(sv, float) else str(sv)
            mv_fmt = f"{mv:.1f}" if isinstance(mv, float) else str(mv)
            print(f"  {label:<23} {sv_fmt:<15} {mv_fmt:<15} {winner:<10}")

    print("  " + "-" * 63)
    if m_time < s_time:
        diff = s_time - m_time
        pct = (diff / s_time) * 100 if s_time else 0
        print(f"\n  Multi-agent was {diff:.1f}s faster ({pct:.0f}% improvement)")
    elif s_time < m_time:
        diff = m_time - s_time
        print(f"\n  Single-agent was {diff:.1f}s faster (multi-agent overhead)")
    else:
        print("\n  Both modes took the same time.")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_research(
    question: str,
    mode: str = "multi",
    skip_clarification: bool = False,
    max_iterations: int = 5,
    verbose: bool = True,
    output_file: str | None = None,
    model: str | None = None,
    fast_model: str | None = None,
    max_search_results: int | None = None,
    report_structure: str | None = None,
    system_prompt: str | None = None,
) -> tuple[dict, float]:
    """Run the research graph and return (final_state, elapsed_seconds)."""
    start = time.time()
    init = {
        "query": question,
        "messages": [HumanMessage(content=question)],
        **_default_state(mode=mode, max_iterations=max_iterations),
    }

    # Build the configurable dict from CLI overrides
    configurable: dict = {}
    if model:
        configurable["model"] = model
    if fast_model:
        configurable["fast_model"] = fast_model
    if max_search_results is not None:
        configurable["max_search_results"] = max_search_results
    if report_structure:
        configurable["report_structure"] = report_structure
    if system_prompt:
        configurable["system_prompt"] = system_prompt

    if skip_clarification:
        # No checkpointer, no HITL pause — just run straight through
        graph = build_graph(checkpointer=None, interrupt_on_clarify=False)
        if verbose:
            print("\n[1/5] Analyzing query...")
        run_config = {"configurable": configurable} if configurable else {}
        result = graph.invoke(init, run_config)

    else:
        # With HITL: checkpointer so we can pause at clarify and resume
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer, interrupt_on_clarify=True)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id, **configurable}}

        if verbose:
            print("\n[1/5] Analyzing query...")

        result = graph.invoke(init, config)

        # Did the graph pause at the clarify node?
        snapshot = graph.get_state(config)
        if snapshot.next and "clarify" in snapshot.next:
            cq = result.get("clarification_question", "")
            options = result.get("clarification_options", [])

            if verbose:
                print("\n" + "-" * 60)
                print("CLARIFICATION NEEDED")
                print("-" * 60)
                if cq:
                    print(f"\n{cq}\n")
                if options:
                    print("Options:")
                    for i, opt in enumerate(options, 1):
                        if isinstance(opt, dict):
                            label = opt.get("label", str(opt))
                            desc = opt.get("description", "")
                            print(f"  {i}. {label}")
                            if desc:
                                print(f"     {desc}")
                        else:
                            print(f"  {i}. {opt}")
                    print(f"  {len(options) + 1}. Other (type your own)")
                print("-" * 60)

            user_input = input("\nYour choice (number or text): ").strip()

            # Resolve numbered choice to the option label
            if user_input.isdigit() and options:
                idx = int(user_input) - 1
                if 0 <= idx < len(options):
                    opt = options[idx]
                    user_input = opt.get("label", str(opt)) if isinstance(opt, dict) else str(opt)

            if verbose:
                print(f"\n[2/5] Processing clarification: {user_input}")

            graph.update_state(config, {"user_clarification": user_input})
            result = graph.invoke(None, config)
        else:
            if verbose:
                print("[2/5] Query is clear, skipping clarification...")

    elapsed = time.time() - start

    if verbose:
        print("[3/5] Research complete.")
        print("[4/5] Verification done.")
        print("[5/5] Report generated.\n")

        print("=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(result.get("report", "(no report generated)"))
        _print_metrics(result, elapsed, label=mode.title())

    if output_file:
        with open(output_file, "w") as f:
            f.write(result.get("report", ""))
        if verbose:
            print(f"\nSaved to: {output_file}")

    return result, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_help():
    print("""
Research Studio - LangGraph Research Pipeline

Usage:
    python -m src.run [OPTIONS] [QUESTION]

Modes:
    (default)         Multi-agent (parallel workers via LangGraph Send)
    --single-agent    Single-agent with iterative gap detection
    --compare         Run both modes and compare metrics side-by-side

Options:
    --quiet, -q            Minimal output
    --output FILE          Save report to file
    -o FILE                Save report to file
    --iterations N         Max research iterations (default: 5 single, 2 multi)
    --model MODEL          LLM for planning/writing (default: gpt-4o)
    --fast-model MODEL     LLM for extraction (default: gpt-4o-mini)
    --max-results N        Tavily results per query (default: 5)
    --format FORMAT        Report style: detailed|concise|bullet_points (default: detailed)
    --system-prompt TEXT   Custom instructions prepended to the report writer
    --help, -h             Show this help message

Examples:
    python -m src.run "What is quantum computing?"
    python -m src.run --single-agent "Compare React vs Vue"
    python -m src.run --compare "Latest AI developments"
    python -m src.run --single-agent --iterations 3 "Quantum computing"
    python -m src.run --model gpt-4o-mini --format concise "Explain CRISPR"
    python -m src.run --system-prompt "Write for a technical audience" "Rust vs Go"
""")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    args = sys.argv[1:]
    verbose = True
    output_file = None
    mode = "multi"
    max_iterations = None
    model = None
    fast_model = None
    max_search_results = None
    report_structure = None
    system_prompt = None

    # Parse flags
    if "--help" in args or "-h" in args:
        _print_help()
        return

    if "--single-agent" in args:
        args.remove("--single-agent")
        mode = "single"
    if "--compare" in args:
        args.remove("--compare")
        mode = "compare"
    if "--quiet" in args:
        args.remove("--quiet")
        verbose = False
    if "-q" in args:
        args.remove("-q")
        verbose = False

    # Helper to pop --flag VALUE pairs
    def _pop_flag(flag_name: str) -> str | None:
        if flag_name in args:
            idx = args.index(flag_name)
            if idx + 1 < len(args):
                val = args[idx + 1]
                args.pop(idx)
                args.pop(idx)
                return val
        return None

    # Parse --flag VALUE options
    val = _pop_flag("--iterations")
    if val is not None:
        max_iterations = int(val)

    val = _pop_flag("--model")
    if val is not None:
        model = val

    val = _pop_flag("--fast-model")
    if val is not None:
        fast_model = val

    val = _pop_flag("--max-results")
    if val is not None:
        max_search_results = int(val)

    val = _pop_flag("--format")
    if val is not None:
        report_structure = val

    val = _pop_flag("--system-prompt")
    if val is not None:
        system_prompt = val

    # Parse --output FILE / -o FILE
    for flag in ("--output", "-o"):
        val = _pop_flag(flag)
        if val is not None:
            output_file = val

    # Whatever's left is the question
    if args:
        question = " ".join(args)
    else:
        _print_banner()
        question = input("\nEnter your research question: ").strip()

    if not question:
        print("Error: No question provided.")
        sys.exit(1)

    if verbose:
        _print_banner()
        print(f"\nQuery: {question}")
        print(f"Mode: {mode.upper()}")
        print("-" * 60)

    # Shared config kwargs for run_research
    config_kwargs = dict(
        model=model,
        fast_model=fast_model,
        max_search_results=max_search_results,
        report_structure=report_structure,
        system_prompt=system_prompt,
    )

    if mode == "compare":
        s_iters = max_iterations or 5
        m_iters = max_iterations or 2

        print("\n>>> Running SINGLE-AGENT first...")
        print("-" * 60)
        s_result, s_time = run_research(
            question, mode="single", skip_clarification=True,
            max_iterations=s_iters, verbose=verbose, **config_kwargs,
        )

        print("\n\n>>> Running MULTI-AGENT now...")
        print("-" * 60)
        m_result, m_time = run_research(
            question, mode="multi", skip_clarification=True,
            max_iterations=m_iters, verbose=verbose, **config_kwargs,
        )

        _print_comparison(s_result, s_time, m_result, m_time)
    else:
        iters = max_iterations or (5 if mode == "single" else 2)
        run_research(
            question, mode=mode, skip_clarification=False,
            max_iterations=iters, verbose=verbose, output_file=output_file,
            **config_kwargs,
        )


if __name__ == "__main__":
    main()
