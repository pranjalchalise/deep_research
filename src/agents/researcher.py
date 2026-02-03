"""
General Deep Research Agent - ReAct Architecture

Inspired by:
- OpenAI: Plan-Act-Observe loop, no hardcoded patterns
- Anthropic: LLM-driven reasoning, orchestrator-workers
- Google: Iterative search-read-reflect
- Perplexity: "Don't say anything not retrieved"

This is a GENERAL research agent that works for ANY query type.
NO hardcoded query classes. NO templates. Just pure LLM reasoning.

The agent follows the ReAct pattern:
1. THINK - Understand what to research, plan approach
2. ACT - Search, read pages, extract information
3. OBSERVE - What did I learn? What's missing?
4. REPEAT - Until sufficient or max iterations

Key principles:
- Let the LLM figure out what to search (no templates)
- Let the LLM decide when it has enough (no hardcoded thresholds)
- Ground everything in retrieved evidence (Perplexity rule)
- Iterate until confident or exhausted
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk


# =============================================================================
# PROMPTS - The only "configuration" in this agent
# =============================================================================

THINK_PROMPT = """You are a research agent. Given a user's question, think about what you need to research.

USER'S QUESTION:
{query}

WHAT YOU ALREADY KNOW (from previous searches):
{context}

Think step by step:
1. What is the user actually asking for?
2. What information do I need to answer this comprehensively?
3. What should I search for next?

Respond with JSON:
{{
    "understanding": "What the user wants to know (1-2 sentences)",
    "information_needed": ["piece of info 1", "piece of info 2", ...],
    "search_queries": ["search query 1", "search query 2", "search query 3"],
    "reasoning": "Why these searches will help"
}}

Generate 2-4 specific, targeted search queries. Be specific to the actual topic.
"""

EXTRACT_PROMPT = """Extract key facts from this content that are relevant to the research question.

RESEARCH QUESTION:
{query}

CONTENT FROM {url}:
{content}

Extract 3-5 key facts. Each fact should be:
- Specific and factual (not vague)
- Directly relevant to the research question
- Something that could be cited

Respond with JSON array:
[
    {{"fact": "The specific fact", "relevance": "Why this matters for the question"}}
]

If the content is not relevant, return an empty array: []
"""

REFLECT_PROMPT = """You are researching: {query}

Here's what you've gathered so far:

{evidence}

Reflect on your research:
1. Do you have enough information to write a comprehensive answer?
2. What aspects are well-covered?
3. What's still missing or unclear?

Respond with JSON:
{{
    "have_enough": true/false,
    "well_covered": ["aspect 1", "aspect 2"],
    "missing": ["missing aspect 1", "missing aspect 2"],
    "confidence": 0.0-1.0,
    "next_searches": ["query 1", "query 2"] // only if have_enough is false
}}

Be honest. If you're missing important information, say so.
"""

WRITE_PROMPT = """Write a comprehensive research report answering this question:

{query}

Use ONLY the following evidence. Every claim must be supported by the evidence.
Include citations like [1], [2] etc.

EVIDENCE:
{evidence}

SOURCES:
{sources}

Write a well-structured report with:
1. A clear answer to the question
2. Supporting details and context
3. Citations for every factual claim
4. Acknowledgment of any limitations or gaps

If the evidence is insufficient to answer the question well, say so honestly.
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Evidence:
    """A piece of evidence extracted from a source."""
    fact: str
    source_url: str
    source_title: str
    relevance: str = ""


@dataclass
class ResearchState:
    """Current state of the research."""
    query: str
    evidence: List[Evidence] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)  # url -> title
    searches_done: List[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 5
    confidence: float = 0.0


# =============================================================================
# THE RESEARCH AGENT
# =============================================================================

class ResearchAgent:
    """
    A general-purpose deep research agent using the ReAct pattern.

    No hardcoded query types. No templates. Just:
    Think → Act → Observe → Repeat → Write
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        max_sources: int = 15,
        verbose: bool = True,
    ):
        self.llm = create_chat_model(model=model, temperature=0.2)
        self.fast_llm = create_chat_model(model=fast_model, temperature=0.1)
        self.max_iterations = max_iterations
        self.max_sources = max_sources
        self.verbose = verbose

    def research(self, query: str) -> Dict[str, Any]:
        """
        Research a query and return a report.

        This is the main entry point. It runs the ReAct loop until
        the agent is confident or reaches max iterations.
        """
        state = ResearchState(query=query, max_iterations=self.max_iterations)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RESEARCHING: {query}")
            print(f"{'='*60}")

        while state.iterations < state.max_iterations:
            state.iterations += 1

            if self.verbose:
                print(f"\n--- Iteration {state.iterations} ---")

            # THINK: What should I search for?
            search_plan = self._think(state)

            if not search_plan.get("search_queries"):
                if self.verbose:
                    print("No more searches needed.")
                break

            # ACT: Execute searches and extract evidence
            new_evidence = self._act(state, search_plan["search_queries"])
            state.evidence.extend(new_evidence)

            if self.verbose:
                print(f"Collected {len(new_evidence)} new pieces of evidence")
                print(f"Total evidence: {len(state.evidence)}")

            # OBSERVE: Do I have enough?
            reflection = self._reflect(state)
            state.confidence = reflection.get("confidence", 0)

            if self.verbose:
                print(f"Confidence: {state.confidence:.0%}")
                if reflection.get("missing"):
                    print(f"Still missing: {reflection['missing']}")

            if reflection.get("have_enough", False):
                if self.verbose:
                    print("Have enough information!")
                break

            # Use the suggested next searches if available
            if reflection.get("next_searches"):
                # Store for next iteration's think phase
                state.searches_done.extend(search_plan["search_queries"])

        # WRITE: Generate the final report
        report = self._write(state)

        return {
            "report": report,
            "sources": state.sources,
            "evidence_count": len(state.evidence),
            "iterations": state.iterations,
            "confidence": state.confidence,
        }

    def _think(self, state: ResearchState) -> Dict[str, Any]:
        """Think about what to search for next."""
        # Build context from existing evidence
        if state.evidence:
            context = "\n".join([
                f"- {e.fact} (from {e.source_title})"
                for e in state.evidence[-10:]  # Last 10 pieces
            ])
        else:
            context = "Nothing yet - this is the first search."

        prompt = THINK_PROMPT.format(
            query=state.query,
            context=context
        )

        response = self.llm.invoke([
            SystemMessage(content="You are a research planning agent. Output valid JSON only."),
            HumanMessage(content=prompt)
        ])

        result = parse_json_object(response.content, default={})

        if self.verbose and result.get("understanding"):
            print(f"Understanding: {result['understanding']}")
            print(f"Searches: {result.get('search_queries', [])}")

        # Filter out searches we've already done
        if result.get("search_queries"):
            result["search_queries"] = [
                q for q in result["search_queries"]
                if q not in state.searches_done
            ]

        return result

    def _act(self, state: ResearchState, queries: List[str]) -> List[Evidence]:
        """Execute searches and extract evidence."""
        new_evidence = []

        for query in queries[:3]:  # Max 3 searches per iteration
            if self.verbose:
                print(f"  Searching: {query}")

            state.searches_done.append(query)

            # Search
            results = cached_search(
                query=query,
                max_results=5,
                use_cache=True,
                cache_dir=".cache_v9/search"
            )

            if not results:
                continue

            # Process top 2 results
            for result in results[:2]:
                url = result["url"]
                title = result["title"]

                # Skip if we already have this source
                if url in state.sources:
                    continue

                # Skip if we have enough sources
                if len(state.sources) >= self.max_sources:
                    break

                state.sources[url] = title

                # Fetch and extract
                chunks = fetch_and_chunk(
                    url=url,
                    chunk_chars=4000,
                    max_chunks=2,
                    timeout_s=10,
                    use_cache=True,
                    cache_dir=".cache_v9/pages"
                )

                if not chunks:
                    continue

                # Extract evidence from content
                evidence = self._extract_evidence(
                    query=state.query,
                    url=url,
                    title=title,
                    content="\n\n".join(chunks[:2])
                )

                new_evidence.extend(evidence)

        return new_evidence

    def _extract_evidence(
        self,
        query: str,
        url: str,
        title: str,
        content: str
    ) -> List[Evidence]:
        """Extract relevant evidence from content."""
        prompt = EXTRACT_PROMPT.format(
            query=query,
            url=url,
            content=content[:6000]  # Limit content length
        )

        response = self.fast_llm.invoke([
            SystemMessage(content="Extract facts as JSON array. Return [] if not relevant."),
            HumanMessage(content=prompt)
        ])

        facts = parse_json_array(response.content, default=[])

        evidence = []
        for fact in facts[:5]:  # Max 5 facts per source
            if isinstance(fact, dict) and fact.get("fact"):
                evidence.append(Evidence(
                    fact=fact["fact"],
                    source_url=url,
                    source_title=title,
                    relevance=fact.get("relevance", "")
                ))

        return evidence

    def _reflect(self, state: ResearchState) -> Dict[str, Any]:
        """Reflect on whether we have enough information."""
        if not state.evidence:
            return {
                "have_enough": False,
                "confidence": 0.0,
                "missing": ["No evidence collected yet"]
            }

        evidence_text = "\n".join([
            f"- {e.fact}"
            for e in state.evidence
        ])

        prompt = REFLECT_PROMPT.format(
            query=state.query,
            evidence=evidence_text
        )

        response = self.llm.invoke([
            SystemMessage(content="Reflect on research progress. Output valid JSON."),
            HumanMessage(content=prompt)
        ])

        return parse_json_object(response.content, default={
            "have_enough": False,
            "confidence": 0.5
        })

    def _write(self, state: ResearchState) -> str:
        """Write the final research report."""
        if not state.evidence:
            return f"# Research Report\n\nI was unable to find relevant information about: {state.query}\n\nPlease try rephrasing your question or providing more context."

        # Format evidence with numbers
        evidence_lines = []
        for i, e in enumerate(state.evidence, 1):
            evidence_lines.append(f"[{i}] {e.fact}")

        # Format sources with numbers
        source_lines = []
        url_to_num = {}
        for i, (url, title) in enumerate(state.sources.items(), 1):
            source_lines.append(f"[{i}] {title} - {url}")
            url_to_num[url] = i

        prompt = WRITE_PROMPT.format(
            query=state.query,
            evidence="\n".join(evidence_lines),
            sources="\n".join(source_lines)
        )

        response = self.llm.invoke([
            SystemMessage(content="Write a research report. Ground every claim in the evidence provided."),
            HumanMessage(content=prompt)
        ])

        report = response.content

        # Add sources section if not already included
        if "Sources" not in report and "References" not in report:
            report += "\n\n---\n\n## Sources\n\n" + "\n".join(source_lines)

        return report


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def research(query: str, **kwargs) -> Dict[str, Any]:
    """
    Research a query and return results.

    This is the main entry point for using the research agent.

    Args:
        query: Any research question
        **kwargs: Additional arguments for ResearchAgent

    Returns:
        Dict with report, sources, evidence_count, iterations, confidence

    Example:
        result = research("What are the latest developments in quantum computing?")
        print(result["report"])
    """
    agent = ResearchAgent(**kwargs)
    return agent.research(query)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python researcher.py 'your research question'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = research(query)

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["report"])
    print("\n" + "="*60)
    print(f"Sources: {len(result['sources'])}")
    print(f"Evidence: {result['evidence_count']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Confidence: {result['confidence']:.0%}")
