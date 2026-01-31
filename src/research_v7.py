from __future__ import annotations

import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.core.graph import build_graph_with_memory, build_simple_graph
from src.core.config import V7Config


def get_state_value(state: dict, key: str, default=None):
    """Safely get a value from state."""
    return state.get(key, default)


def run_research(question: str, skip_clarification: bool = False):
    """
    Run the research pipeline with optional human-in-the-loop clarification.

    Args:
        question: The research question
        skip_clarification: If True, skip human clarification (auto-proceed)
    """
    cfg = V7Config()

    if skip_clarification:
        # Simple mode - no interrupt
        app = build_simple_graph()
        checkpointer = None
        config = {}
    else:
        # Human-in-the-loop mode
        app, checkpointer = build_graph_with_memory(interrupt_on_clarify=True)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

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

    print("\n" + "=" * 60)
    print("RESEARCH STUDIO v7")
    print("=" * 60)
    print(f"\nQuery: {question}")
    print("\n[1/5] Analyzing query...")

    # First invocation
    if skip_clarification:
        result = app.invoke(init)
    else:
        result = app.invoke(init, config)

    # Check if we hit the interrupt (needs clarification)
    if not skip_clarification:
        snapshot = app.get_state(config)

        if snapshot.next and "clarify" in snapshot.next:
            # We're at the clarify node - need human input
            state = snapshot.values
            clarification_request = get_state_value(state, "clarification_request", "")
            discovery = get_state_value(state, "discovery", {})
            candidates = discovery.get("entity_candidates", [])
            confidence = discovery.get("confidence", 0)

            print(f"\n[2/5] Discovery complete (confidence: {confidence:.0%})")
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
                        print(f"  {i+1}. {c.get('name', 'Unknown')} - {desc}")

            # Get human input
            print("\n" + "-" * 60)
            human_response = input("Your response (or press Enter for best guess): ").strip()

            # Resume with human clarification
            print("\n[3/5] Processing clarification...")

            # Update state with human response
            app.update_state(config, {"human_clarification": human_response})

            # Continue execution
            result = app.invoke(None, config)

        else:
            # High confidence - no clarification needed
            print("\n[2/5] Entity identified with high confidence")
            print("[3/5] Skipping clarification...")

    print("[4/5] Conducting research...")
    print("[5/5] Generating report...\n")

    # Get final result
    if skip_clarification:
        report = result.get("report", "")
        if not report and result.get("messages"):
            report = result["messages"][-1].content
    else:
        final_state = app.get_state(config)
        report = get_state_value(final_state.values, "report", "")
        if not report and final_state.values.get("messages"):
            report = final_state.values["messages"][-1].content

    print("=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(report)
    print("\n" + "=" * 60)


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    import sys

    # Check for --simple flag
    if "--simple" in sys.argv:
        sys.argv.remove("--simple")
        skip_clarification = True
    else:
        skip_clarification = False

    # Get question from args or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your research question: ").strip()

    if not question:
        print("No question provided.")
        return

    run_research(question, skip_clarification=skip_clarification)


if __name__ == "__main__":
    main()
