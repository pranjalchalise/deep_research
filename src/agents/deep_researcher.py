"""
Deep Research Agent - Full-Featured General Research System

A truly general research agent inspired by OpenAI, Anthropic, Google, and Perplexity.
NO hardcoded query types. NO templates. Just pure LLM reasoning.

Features:
1. HUMAN-IN-THE-LOOP: Asks for clarification when query is ambiguous
2. KNOWLEDGE GAP DETECTION: Iteratively fills gaps until comprehensive
3. CITATION-GROUNDED: Every claim is backed by evidence with source citations

Architecture (ReAct + Enhancements):
┌─────────────────────────────────────────────────────────────────┐
│  USER QUERY                                                      │
│       ↓                                                          │
│  ┌─────────────┐                                                 │
│  │ UNDERSTAND  │ → Is this clear enough to research?             │
│  └─────────────┘                                                 │
│       ↓ No                    ↓ Yes                              │
│  ┌─────────────┐         ┌─────────────┐                         │
│  │ CLARIFY     │ ←HITL→  │ PLAN        │                         │
│  │ (Ask User)  │         │ (Questions) │                         │
│  └─────────────┘         └─────────────┘                         │
│                               ↓                                  │
│                    ┌──────────────────────┐                      │
│                    │   RESEARCH LOOP      │                      │
│                    │  ┌────────────────┐  │                      │
│                    │  │ SEARCH & READ  │  │                      │
│                    │  └────────────────┘  │                      │
│                    │         ↓            │                      │
│                    │  ┌────────────────┐  │                      │
│                    │  │ EXTRACT FACTS  │  │                      │
│                    │  └────────────────┘  │                      │
│                    │         ↓            │                      │
│                    │  ┌────────────────┐  │                      │
│                    │  │ DETECT GAPS   │───┼──→ Loop if gaps      │
│                    │  └────────────────┘  │                      │
│                    └──────────────────────┘                      │
│                               ↓                                  │
│                    ┌──────────────────────┐                      │
│                    │ VERIFY & CITE        │                      │
│                    │ (Cross-reference)    │                      │
│                    └──────────────────────┘                      │
│                               ↓                                  │
│                    ┌──────────────────────┐                      │
│                    │ WRITE REPORT         │                      │
│                    │ (Grounded in sources)│                      │
│                    └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk


# =============================================================================
# PROMPTS - General prompts that work for ANY query type
# =============================================================================

UNDERSTAND_PROMPT = """You are a research assistant. Analyze this query to understand what the user wants.

USER'S QUERY:
{query}

Think carefully:
1. What is the user asking about?
2. Is this query clear enough to research, or is it ambiguous?
3. If ambiguous, what would you need to clarify?

A query is AMBIGUOUS if:
- It could mean multiple different things
- Important context is missing (time period, location, specific aspect)
- The scope is unclear (too broad or unclear focus)
- Key terms are vague or could refer to different things

Respond with JSON:
{{
    "understanding": "What you understand the user wants (1-2 sentences)",
    "is_clear": true/false,
    "ambiguity_reason": "Why it's ambiguous (only if is_clear is false)",
    "clarification_question": "Question to ask user (only if is_clear is false)",
    "clarification_options": ["option 1", "option 2", "option 3"] // suggested answers (only if is_clear is false)
}}
"""

PLAN_PROMPT = """You are a research planner. Create a research plan for this query.

USER'S QUERY:
{query}

{clarification_context}

Think about:
1. What are the key questions that need to be answered?
2. What aspects or dimensions should be covered?
3. What search queries will find the needed information?

Respond with JSON:
{{
    "research_questions": [
        "Key question 1 that needs to be answered",
        "Key question 2 that needs to be answered",
        "Key question 3 that needs to be answered"
    ],
    "aspects_to_cover": [
        "Aspect/dimension 1 to research",
        "Aspect/dimension 2 to research"
    ],
    "initial_searches": [
        "search query 1",
        "search query 2",
        "search query 3",
        "search query 4"
    ]
}}

Generate 3-5 research questions and 4-6 search queries that will comprehensively cover the topic.
"""

EXTRACT_PROMPT = """Extract key facts from this content relevant to the research.

RESEARCH CONTEXT:
{context}

CONTENT FROM {url}:
{content}

Extract specific, factual information. Each fact should be:
- Specific and verifiable (not vague)
- Directly relevant to the research
- Self-contained (understandable on its own)

Respond with JSON array:
[
    {{
        "fact": "The specific factual claim",
        "quote": "Direct quote or paraphrase from the source",
        "confidence": 0.0-1.0
    }}
]

Return [] if no relevant facts found. Be selective - quality over quantity.
"""

GAP_DETECTION_PROMPT = """You are assessing research completeness.

ORIGINAL QUERY:
{query}

RESEARCH QUESTIONS TO ANSWER:
{questions}

EVIDENCE GATHERED SO FAR:
{evidence}

Analyze the coverage:
1. Which research questions are well-answered with strong evidence?
2. Which questions have weak or no evidence?
3. What important aspects are missing?

Respond with JSON:
{{
    "coverage_assessment": [
        {{"question": "question text", "status": "well_covered|partial|missing", "evidence_count": N}},
        ...
    ],
    "overall_coverage": 0.0-1.0,
    "gaps": [
        "Specific gap or missing information 1",
        "Specific gap or missing information 2"
    ],
    "suggested_searches": [
        "search query to fill gap 1",
        "search query to fill gap 2"
    ],
    "ready_to_write": true/false,
    "reasoning": "Why ready or not ready"
}}

Be honest about gaps. It's better to do another search iteration than write an incomplete report.
"""

VERIFY_PROMPT = """You are a fact-checker. Verify claims against the evidence.

CLAIMS TO VERIFY:
{claims}

AVAILABLE EVIDENCE:
{evidence}

For each claim, check if it's supported by the evidence.

Respond with JSON array:
[
    {{
        "claim": "The claim text",
        "supported": true/false,
        "supporting_evidence": [1, 3, 5],  // indices of supporting evidence
        "confidence": 0.0-1.0,
        "notes": "Any caveats or qualifications"
    }}
]
"""

WRITE_PROMPT = """Write a comprehensive research report answering this query.

QUERY:
{query}

VERIFIED EVIDENCE (use these as your ONLY source of facts):
{evidence}

SOURCES:
{sources}

IMPORTANT RULES:
1. ONLY include information that appears in the evidence above
2. EVERY factual claim must have a citation [1], [2], etc.
3. If evidence is contradictory, acknowledge both perspectives
4. If evidence is insufficient for some aspect, acknowledge the limitation
5. Be comprehensive but don't make things up

Structure your report with:
- Clear sections addressing different aspects
- Inline citations for every factual claim
- A Sources section at the end

Write in a professional, informative tone.
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Evidence:
    """A piece of evidence from a source."""
    fact: str
    quote: str
    source_url: str
    source_title: str
    confidence: float = 0.8
    evidence_id: int = 0

    def to_citation_str(self) -> str:
        return f"[{self.evidence_id}] {self.fact}"


@dataclass
class Source:
    """A source that was consulted."""
    url: str
    title: str
    source_id: int = 0
    evidence_count: int = 0
    credibility_score: float = 0.7

    def to_citation_str(self) -> str:
        return f"[{self.source_id}] {self.title} - {self.url}"


@dataclass
class VerifiedClaim:
    """A claim that has been verified against evidence."""
    claim: str
    supported: bool
    supporting_evidence_ids: List[int]
    confidence: float
    notes: str = ""


@dataclass
class ResearchState:
    """Complete state of the research process."""
    query: str
    clarified_query: str = ""
    research_questions: List[str] = field(default_factory=list)
    aspects_to_cover: List[str] = field(default_factory=list)

    # Evidence and sources
    evidence: List[Evidence] = field(default_factory=list)
    sources: Dict[str, Source] = field(default_factory=dict)

    # Tracking
    searches_done: List[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 5

    # Gap detection
    coverage: float = 0.0
    gaps: List[str] = field(default_factory=list)

    # Verification
    verified_claims: List[VerifiedClaim] = field(default_factory=list)

    # Output
    report: str = ""
    confidence: float = 0.0


# =============================================================================
# THE DEEP RESEARCH AGENT
# =============================================================================

class DeepResearchAgent:
    """
    A full-featured general research agent.

    Features:
    - Human-in-the-loop clarification for ambiguous queries
    - Knowledge gap detection with iterative research
    - Citation-grounded results with verification

    No hardcoded query types. Works for ANY research question.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        max_sources: int = 20,
        min_coverage: float = 0.7,
        verbose: bool = True,
        clarification_callback: Optional[Callable[[str, List[str]], str]] = None,
    ):
        """
        Initialize the research agent.

        Args:
            model: Main model for complex reasoning
            fast_model: Fast model for extraction tasks
            max_iterations: Maximum research iterations
            max_sources: Maximum sources to consult
            min_coverage: Minimum coverage before writing (0-1)
            verbose: Print progress to console
            clarification_callback: Function to get user clarification
                                   Called with (question, options) -> user_response
        """
        self.llm = create_chat_model(model=model, temperature=0.2)
        self.fast_llm = create_chat_model(model=fast_model, temperature=0.1)
        self.max_iterations = max_iterations
        self.max_sources = max_sources
        self.min_coverage = min_coverage
        self.verbose = verbose
        self.clarification_callback = clarification_callback or self._default_clarification

    def _log(self, message: str, prefix: str = ""):
        """Log a message if verbose mode is on."""
        if self.verbose:
            if prefix:
                print(f"[{prefix}] {message}")
            else:
                print(message)

    def _default_clarification(self, question: str, options: List[str]) -> str:
        """Default clarification: ask via console input."""
        print(f"\n{'='*60}")
        print("CLARIFICATION NEEDED")
        print('='*60)
        print(f"\n{question}\n")
        if options:
            print("Suggested options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            print(f"  {len(options)+1}. Other (type your own)")

        response = input("\nYour answer: ").strip()

        # Check if user selected a number
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return options[idx]

        return response

    # =========================================================================
    # PHASE 1: UNDERSTAND & CLARIFY
    # =========================================================================

    def _understand_query(self, query: str) -> Dict[str, Any]:
        """Understand the query and check if clarification is needed."""
        self._log("Analyzing query...", "UNDERSTAND")

        response = self.llm.invoke([
            SystemMessage(content="Analyze the query. Output valid JSON only."),
            HumanMessage(content=UNDERSTAND_PROMPT.format(query=query))
        ])

        result = parse_json_object(response.content, default={
            "understanding": query,
            "is_clear": True
        })

        self._log(f"Understanding: {result.get('understanding', 'N/A')}", "UNDERSTAND")

        return result

    def _clarify_if_needed(self, state: ResearchState, understanding: Dict) -> str:
        """Ask for clarification if the query is ambiguous."""
        if understanding.get("is_clear", True):
            return state.query

        self._log(f"Query is ambiguous: {understanding.get('ambiguity_reason', 'unclear')}", "CLARIFY")

        question = understanding.get("clarification_question", "Could you please clarify your question?")
        options = understanding.get("clarification_options", [])

        # Get user clarification
        user_response = self.clarification_callback(question, options)

        # Combine original query with clarification
        clarified = f"{state.query} - Clarification: {user_response}"
        self._log(f"Clarified query: {clarified}", "CLARIFY")

        return clarified

    # =========================================================================
    # PHASE 2: PLAN
    # =========================================================================

    def _create_plan(self, state: ResearchState) -> Dict[str, Any]:
        """Create a research plan with questions and initial searches."""
        self._log("Creating research plan...", "PLAN")

        # Build clarification context if any
        clarification_context = ""
        if state.clarified_query != state.query:
            clarification_context = f"USER CLARIFICATION: {state.clarified_query.replace(state.query + ' - Clarification: ', '')}"

        response = self.llm.invoke([
            SystemMessage(content="Create a research plan. Output valid JSON only."),
            HumanMessage(content=PLAN_PROMPT.format(
                query=state.clarified_query or state.query,
                clarification_context=clarification_context
            ))
        ])

        result = parse_json_object(response.content, default={
            "research_questions": [state.query],
            "aspects_to_cover": [],
            "initial_searches": [state.query]
        })

        self._log(f"Research questions: {len(result.get('research_questions', []))}", "PLAN")
        self._log(f"Initial searches: {result.get('initial_searches', [])}", "PLAN")

        return result

    # =========================================================================
    # PHASE 3: RESEARCH LOOP
    # =========================================================================

    def _search_and_extract(self, state: ResearchState, queries: List[str]) -> List[Evidence]:
        """Execute searches and extract evidence."""
        new_evidence = []

        for query in queries:
            if query in state.searches_done:
                continue

            self._log(f"Searching: {query}", "SEARCH")
            state.searches_done.append(query)

            # Search
            results = cached_search(
                query=query,
                max_results=5,
                use_cache=True,
                cache_dir=".cache_v10/search"
            )

            if not results:
                continue

            # Process top results
            for result in results[:3]:
                url = result["url"]
                title = result["title"]

                if url in state.sources:
                    continue

                if len(state.sources) >= self.max_sources:
                    break

                # Create source
                source_id = len(state.sources) + 1
                source = Source(
                    url=url,
                    title=title,
                    source_id=source_id
                )
                state.sources[url] = source

                # Fetch content
                chunks = fetch_and_chunk(
                    url=url,
                    chunk_chars=4000,
                    max_chunks=2,
                    timeout_s=10,
                    use_cache=True,
                    cache_dir=".cache_v10/pages"
                )

                if not chunks:
                    continue

                # Extract evidence
                evidence = self._extract_evidence(
                    state=state,
                    url=url,
                    title=title,
                    content="\n\n".join(chunks[:2])
                )

                source.evidence_count = len(evidence)
                new_evidence.extend(evidence)

        return new_evidence

    def _extract_evidence(
        self,
        state: ResearchState,
        url: str,
        title: str,
        content: str
    ) -> List[Evidence]:
        """Extract evidence from content."""
        # Build context from research questions
        context = f"Query: {state.clarified_query or state.query}\n"
        if state.research_questions:
            context += "Research questions:\n"
            for q in state.research_questions:
                context += f"- {q}\n"

        response = self.fast_llm.invoke([
            SystemMessage(content="Extract facts as JSON array. Return [] if not relevant."),
            HumanMessage(content=EXTRACT_PROMPT.format(
                context=context,
                url=url,
                content=content[:6000]
            ))
        ])

        facts = parse_json_array(response.content, default=[])

        evidence = []
        for fact in facts[:5]:
            if isinstance(fact, dict) and fact.get("fact"):
                evidence_id = len(state.evidence) + len(evidence) + 1
                evidence.append(Evidence(
                    fact=fact["fact"],
                    quote=fact.get("quote", ""),
                    source_url=url,
                    source_title=title,
                    confidence=fact.get("confidence", 0.8),
                    evidence_id=evidence_id
                ))

        return evidence

    def _detect_gaps(self, state: ResearchState) -> Dict[str, Any]:
        """Detect knowledge gaps and decide if more research is needed."""
        self._log("Detecting knowledge gaps...", "GAPS")

        # Format evidence for analysis
        evidence_text = "\n".join([
            f"[{e.evidence_id}] {e.fact} (from: {e.source_title})"
            for e in state.evidence
        ])

        # Format research questions
        questions_text = "\n".join([
            f"- {q}" for q in state.research_questions
        ])

        response = self.llm.invoke([
            SystemMessage(content="Assess research coverage. Output valid JSON."),
            HumanMessage(content=GAP_DETECTION_PROMPT.format(
                query=state.clarified_query or state.query,
                questions=questions_text,
                evidence=evidence_text
            ))
        ])

        result = parse_json_object(response.content, default={
            "overall_coverage": 0.5,
            "gaps": [],
            "suggested_searches": [],
            "ready_to_write": False
        })

        state.coverage = result.get("overall_coverage", 0.5)
        state.gaps = result.get("gaps", [])

        self._log(f"Coverage: {state.coverage:.0%}", "GAPS")
        if state.gaps:
            self._log(f"Gaps found: {state.gaps}", "GAPS")

        return result

    # =========================================================================
    # PHASE 4: VERIFY
    # =========================================================================

    def _verify_evidence(self, state: ResearchState) -> List[VerifiedClaim]:
        """Verify that claims are supported by evidence."""
        self._log("Verifying evidence...", "VERIFY")

        if not state.evidence:
            return []

        # Get key claims from evidence
        claims = [e.fact for e in state.evidence[:20]]  # Top 20
        claims_text = "\n".join([f"- {c}" for c in claims])

        evidence_text = "\n".join([
            f"[{e.evidence_id}] {e.fact} (source: {e.source_title})"
            for e in state.evidence
        ])

        response = self.llm.invoke([
            SystemMessage(content="Verify claims against evidence. Output valid JSON array."),
            HumanMessage(content=VERIFY_PROMPT.format(
                claims=claims_text,
                evidence=evidence_text
            ))
        ])

        verifications = parse_json_array(response.content, default=[])

        verified = []
        for v in verifications:
            if isinstance(v, dict):
                verified.append(VerifiedClaim(
                    claim=v.get("claim", ""),
                    supported=v.get("supported", False),
                    supporting_evidence_ids=v.get("supporting_evidence", []),
                    confidence=v.get("confidence", 0.5),
                    notes=v.get("notes", "")
                ))

        supported_count = sum(1 for v in verified if v.supported)
        self._log(f"Verified {supported_count}/{len(verified)} claims", "VERIFY")

        return verified

    # =========================================================================
    # PHASE 5: WRITE
    # =========================================================================

    def _write_report(self, state: ResearchState) -> str:
        """Write the final research report with citations."""
        self._log("Writing report...", "WRITE")

        if not state.evidence:
            return f"""# Research Report

I was unable to find sufficient information about: {state.query}

Please try:
- Rephrasing your question
- Being more specific
- Checking if the topic has recent coverage
"""

        # Format evidence with IDs
        evidence_text = "\n".join([
            f"[{e.evidence_id}] {e.fact}\n   Quote: \"{e.quote}\"\n   Source: {e.source_title}"
            for e in state.evidence
        ])

        # Format sources
        sources_text = "\n".join([
            f"[{s.source_id}] {s.title}\n   URL: {s.url}\n   Evidence items: {s.evidence_count}"
            for s in state.sources.values()
        ])

        response = self.llm.invoke([
            SystemMessage(content="Write a comprehensive research report. Every claim must have a citation."),
            HumanMessage(content=WRITE_PROMPT.format(
                query=state.clarified_query or state.query,
                evidence=evidence_text,
                sources=sources_text
            ))
        ])

        report = response.content

        # Ensure sources section exists
        if "## Sources" not in report and "## References" not in report:
            report += "\n\n---\n\n## Sources\n\n"
            for source in state.sources.values():
                report += f"{source.to_citation_str()}\n"

        return report

    # =========================================================================
    # MAIN RESEARCH METHOD
    # =========================================================================

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct deep research on a query.

        This is the main entry point. It orchestrates all phases:
        1. Understand & clarify (HITL if needed)
        2. Plan the research
        3. Research loop (search, extract, detect gaps)
        4. Verify evidence
        5. Write report

        Args:
            query: Any research question

        Returns:
            Dict with report, sources, evidence, metadata
        """
        state = ResearchState(
            query=query,
            max_iterations=self.max_iterations
        )

        self._log(f"\n{'='*60}", "")
        self._log(f"DEEP RESEARCH: {query}", "")
        self._log(f"{'='*60}\n", "")

        # -----------------------------------------------------------------
        # PHASE 1: Understand & Clarify
        # -----------------------------------------------------------------
        understanding = self._understand_query(query)
        state.clarified_query = self._clarify_if_needed(state, understanding)

        # -----------------------------------------------------------------
        # PHASE 2: Plan
        # -----------------------------------------------------------------
        plan = self._create_plan(state)
        state.research_questions = plan.get("research_questions", [query])
        state.aspects_to_cover = plan.get("aspects_to_cover", [])

        # Initial searches
        pending_searches = plan.get("initial_searches", [query])

        # -----------------------------------------------------------------
        # PHASE 3: Research Loop
        # -----------------------------------------------------------------
        while state.iterations < state.max_iterations:
            state.iterations += 1
            self._log(f"\n--- Research Iteration {state.iterations} ---", "")

            # Search and extract
            new_evidence = self._search_and_extract(state, pending_searches)
            state.evidence.extend(new_evidence)

            self._log(f"New evidence: {len(new_evidence)}, Total: {len(state.evidence)}", "PROGRESS")

            # Detect gaps
            gap_result = self._detect_gaps(state)

            # Check if ready to write
            if gap_result.get("ready_to_write", False):
                self._log("Sufficient coverage achieved!", "GAPS")
                break

            if state.coverage >= self.min_coverage:
                self._log(f"Coverage threshold met: {state.coverage:.0%}", "GAPS")
                break

            # Get next searches from gap analysis
            pending_searches = gap_result.get("suggested_searches", [])

            if not pending_searches:
                self._log("No more searches suggested", "GAPS")
                break

            self._log(f"Next searches: {pending_searches}", "GAPS")

        # -----------------------------------------------------------------
        # PHASE 4: Verify
        # -----------------------------------------------------------------
        state.verified_claims = self._verify_evidence(state)

        # Calculate confidence from verification
        if state.verified_claims:
            supported = sum(1 for v in state.verified_claims if v.supported)
            state.confidence = supported / len(state.verified_claims)
        else:
            state.confidence = state.coverage

        # -----------------------------------------------------------------
        # PHASE 5: Write
        # -----------------------------------------------------------------
        state.report = self._write_report(state)

        # -----------------------------------------------------------------
        # Return results
        # -----------------------------------------------------------------
        self._log(f"\n{'='*60}", "")
        self._log("RESEARCH COMPLETE", "")
        self._log(f"{'='*60}", "")
        self._log(f"Sources: {len(state.sources)}", "STATS")
        self._log(f"Evidence: {len(state.evidence)}", "STATS")
        self._log(f"Iterations: {state.iterations}", "STATS")
        self._log(f"Coverage: {state.coverage:.0%}", "STATS")
        self._log(f"Confidence: {state.confidence:.0%}", "STATS")

        return {
            "report": state.report,
            "query": state.query,
            "clarified_query": state.clarified_query,
            "sources": {url: {
                "title": s.title,
                "url": s.url,
                "evidence_count": s.evidence_count
            } for url, s in state.sources.items()},
            "evidence": [{
                "fact": e.fact,
                "quote": e.quote,
                "source": e.source_title,
                "source_url": e.source_url,
                "confidence": e.confidence
            } for e in state.evidence],
            "research_questions": state.research_questions,
            "gaps_remaining": state.gaps,
            "verified_claims": [{
                "claim": v.claim,
                "supported": v.supported,
                "confidence": v.confidence
            } for v in state.verified_claims],
            "metadata": {
                "iterations": state.iterations,
                "coverage": state.coverage,
                "confidence": state.confidence,
                "sources_count": len(state.sources),
                "evidence_count": len(state.evidence),
            }
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def deep_research(
    query: str,
    clarification_callback: Optional[Callable[[str, List[str]], str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Conduct deep research on any query.

    Args:
        query: Any research question
        clarification_callback: Optional function for HITL clarification
                               Called with (question, options) -> user_response
        **kwargs: Additional arguments for DeepResearchAgent

    Returns:
        Dict with report, sources, evidence, metadata

    Example:
        # Basic usage (will ask for clarification via console if needed)
        result = deep_research("What are the latest AI developments?")
        print(result["report"])

        # With custom clarification handler
        def my_clarifier(question, options):
            # Custom logic, e.g., GUI dialog
            return options[0] if options else "default"

        result = deep_research("Tell me about Python", clarification_callback=my_clarifier)
    """
    agent = DeepResearchAgent(
        clarification_callback=clarification_callback,
        **kwargs
    )
    return agent.research(query)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deep_researcher.py 'your research question'")
        print("\nExamples:")
        print("  python deep_researcher.py 'What are Trump's immigration policies?'")
        print("  python deep_researcher.py 'Compare React vs Vue for web development'")
        print("  python deep_researcher.py 'Latest developments in quantum computing'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = deep_research(query)

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["report"])
