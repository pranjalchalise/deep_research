"""
Deep Research Graph - LangGraph Implementation

A LangGraph workflow implementing the general deep research pattern:
1. UNDERSTAND - Analyze the query
2. CLARIFY (HITL) - Ask for clarification if needed
3. PLAN - Create research plan
4. RESEARCH LOOP - Search, extract, detect gaps, repeat
5. VERIFY - Verify claims against evidence
6. WRITE - Generate citation-grounded report

This is the graph-based alternative to src/agents/deep_researcher.py
for integration with LangGraph workflows.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk


# =============================================================================
# STATE DEFINITION
# =============================================================================

class EvidenceItem(TypedDict):
    """A piece of evidence from a source."""
    fact: str
    quote: str
    source_url: str
    source_title: str
    confidence: float
    evidence_id: int


class SourceItem(TypedDict):
    """A source that was consulted."""
    url: str
    title: str
    source_id: int
    evidence_count: int


class VerifiedClaimItem(TypedDict):
    """A verified claim."""
    claim: str
    supported: bool
    supporting_evidence: List[int]
    confidence: float


class DeepResearchState(TypedDict, total=False):
    """State for the deep research graph."""
    # Input
    messages: List[Any]  # Conversation messages
    query: str  # Original query

    # Understanding
    understanding: str
    is_clear: bool
    clarification_question: str
    clarification_options: List[str]

    # HITL
    needs_clarification: bool
    user_clarification: str
    clarified_query: str

    # Planning
    research_questions: List[str]
    aspects_to_cover: List[str]
    pending_searches: List[str]

    # Research
    evidence: Annotated[List[EvidenceItem], operator.add]
    sources: Dict[str, SourceItem]
    searches_done: List[str]

    # Gap Detection
    coverage: float
    gaps: List[str]
    ready_to_write: bool

    # Verification
    verified_claims: List[VerifiedClaimItem]

    # Output
    report: str
    confidence: float

    # Control
    research_iteration: int
    max_iterations: int
    max_sources: int
    min_coverage: float

    # Cache config
    use_cache: bool
    cache_dir: str


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

def understand_node(state: DeepResearchState) -> Dict[str, Any]:
    """Understand the query and check if clarification is needed."""
    # Get query from messages
    messages = state.get("messages", [])
    query = state.get("query", "")

    if not query and messages:
        for msg in messages:
            if hasattr(msg, "content"):
                query = msg.content
                break

    llm = create_chat_model(model="gpt-4o", temperature=0.2)

    prompt = f"""Analyze this research query.

USER'S QUERY: {query}

Determine:
1. What is the user asking about?
2. Is this clear enough to research, or is it ambiguous?
3. If ambiguous, what needs clarification?

Respond with JSON:
{{
    "understanding": "What you understand (1-2 sentences)",
    "is_clear": true/false,
    "ambiguity_reason": "Why ambiguous (if applicable)",
    "clarification_question": "Question to ask (if ambiguous)",
    "clarification_options": ["option 1", "option 2", "option 3"]
}}
"""

    response = llm.invoke([
        SystemMessage(content="Analyze the query. Output valid JSON only."),
        HumanMessage(content=prompt)
    ])

    result = parse_json_object(response.content, default={
        "understanding": query,
        "is_clear": True
    })

    return {
        "query": query,
        "understanding": result.get("understanding", query),
        "is_clear": result.get("is_clear", True),
        "clarification_question": result.get("clarification_question", ""),
        "clarification_options": result.get("clarification_options", []),
        "needs_clarification": not result.get("is_clear", True),
    }


def route_after_understand(state: DeepResearchState) -> Literal["clarify", "plan"]:
    """Route based on whether clarification is needed."""
    if state.get("needs_clarification", False) and not state.get("user_clarification"):
        return "clarify"
    return "plan"


def clarify_node(state: DeepResearchState) -> Dict[str, Any]:
    """Prepare clarification request (will interrupt for HITL)."""
    return {
        "needs_clarification": True,
    }


def plan_node(state: DeepResearchState) -> Dict[str, Any]:
    """Create research plan."""
    query = state.get("query", "")
    user_clarification = state.get("user_clarification", "")

    clarified_query = query
    if user_clarification:
        clarified_query = f"{query} - Clarification: {user_clarification}"

    llm = create_chat_model(model="gpt-4o", temperature=0.2)

    prompt = f"""Create a research plan.

QUERY: {clarified_query}

Create:
1. Key research questions to answer
2. Aspects to cover
3. Initial search queries

Respond with JSON:
{{
    "research_questions": ["question 1", "question 2", ...],
    "aspects_to_cover": ["aspect 1", "aspect 2", ...],
    "initial_searches": ["search 1", "search 2", "search 3", "search 4"]
}}
"""

    response = llm.invoke([
        SystemMessage(content="Create a research plan. Output valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = parse_json_object(response.content, default={
        "research_questions": [clarified_query],
        "aspects_to_cover": [],
        "initial_searches": [clarified_query]
    })

    return {
        "clarified_query": clarified_query,
        "research_questions": result.get("research_questions", []),
        "aspects_to_cover": result.get("aspects_to_cover", []),
        "pending_searches": result.get("initial_searches", []),
        "needs_clarification": False,
        "research_iteration": 0,
        "sources": {},
        "searches_done": [],
    }


def search_node(state: DeepResearchState) -> Dict[str, Any]:
    """Execute searches and extract evidence."""
    pending = state.get("pending_searches", [])
    done = state.get("searches_done", [])
    sources = state.get("sources", {})
    clarified_query = state.get("clarified_query", state.get("query", ""))
    research_questions = state.get("research_questions", [])

    max_sources = state.get("max_sources", 20)
    use_cache = state.get("use_cache", True)
    cache_dir = state.get("cache_dir", ".cache_v10")

    llm = create_chat_model(model="gpt-4o-mini", temperature=0.1)

    new_evidence = []
    new_sources = dict(sources)
    new_done = list(done)

    # Build context for extraction
    context = f"Query: {clarified_query}\n"
    if research_questions:
        context += "Research questions:\n"
        for q in research_questions[:5]:
            context += f"- {q}\n"

    for query in pending[:4]:  # Max 4 searches per iteration
        if query in new_done:
            continue

        new_done.append(query)

        # Search
        results = cached_search(
            query=query,
            max_results=5,
            use_cache=use_cache,
            cache_dir=f"{cache_dir}/search"
        )

        if not results:
            continue

        # Process results
        for result in results[:3]:
            url = result["url"]
            title = result["title"]

            if url in new_sources:
                continue

            if len(new_sources) >= max_sources:
                break

            # Create source
            source_id = len(new_sources) + 1
            new_sources[url] = {
                "url": url,
                "title": title,
                "source_id": source_id,
                "evidence_count": 0
            }

            # Fetch content
            chunks = fetch_and_chunk(
                url=url,
                chunk_chars=4000,
                max_chunks=2,
                timeout_s=10,
                use_cache=use_cache,
                cache_dir=f"{cache_dir}/pages"
            )

            if not chunks:
                continue

            # Extract evidence
            content = "\n\n".join(chunks[:2])

            extract_prompt = f"""Extract key facts relevant to the research.

CONTEXT:
{context}

CONTENT FROM {url}:
{content[:6000]}

Respond with JSON array:
[
    {{"fact": "specific fact", "quote": "supporting quote", "confidence": 0.8}}
]

Return [] if not relevant.
"""

            response = llm.invoke([
                SystemMessage(content="Extract facts as JSON array."),
                HumanMessage(content=extract_prompt)
            ])

            facts = parse_json_array(response.content, default=[])

            evidence_count = 0
            for fact in facts[:5]:
                if isinstance(fact, dict) and fact.get("fact"):
                    evidence_id = len(state.get("evidence", [])) + len(new_evidence) + 1
                    new_evidence.append({
                        "fact": fact["fact"],
                        "quote": fact.get("quote", ""),
                        "source_url": url,
                        "source_title": title,
                        "confidence": fact.get("confidence", 0.8),
                        "evidence_id": evidence_id
                    })
                    evidence_count += 1

            new_sources[url]["evidence_count"] = evidence_count

    return {
        "evidence": new_evidence,
        "sources": new_sources,
        "searches_done": new_done,
    }


def gap_detect_node(state: DeepResearchState) -> Dict[str, Any]:
    """Detect knowledge gaps and decide next steps."""
    evidence = state.get("evidence", [])
    research_questions = state.get("research_questions", [])
    clarified_query = state.get("clarified_query", state.get("query", ""))
    iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    min_coverage = state.get("min_coverage", 0.7)

    if not evidence:
        return {
            "coverage": 0.0,
            "gaps": ["No evidence collected"],
            "ready_to_write": False,
            "pending_searches": [clarified_query],
            "research_iteration": iteration + 1,
        }

    llm = create_chat_model(model="gpt-4o", temperature=0.2)

    evidence_text = "\n".join([
        f"[{e['evidence_id']}] {e['fact']}"
        for e in evidence[:30]
    ])

    questions_text = "\n".join([f"- {q}" for q in research_questions])

    prompt = f"""Assess research coverage.

QUERY: {clarified_query}

QUESTIONS TO ANSWER:
{questions_text}

EVIDENCE:
{evidence_text}

Respond with JSON:
{{
    "overall_coverage": 0.0-1.0,
    "gaps": ["gap 1", "gap 2"],
    "suggested_searches": ["search 1", "search 2"],
    "ready_to_write": true/false
}}
"""

    response = llm.invoke([
        SystemMessage(content="Assess coverage. Output valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = parse_json_object(response.content, default={
        "overall_coverage": 0.5,
        "gaps": [],
        "suggested_searches": [],
        "ready_to_write": False
    })

    coverage = result.get("overall_coverage", 0.5)
    ready = result.get("ready_to_write", False)

    # Force completion if at max iterations or high coverage
    if iteration >= max_iterations - 1 or coverage >= min_coverage:
        ready = True

    return {
        "coverage": coverage,
        "gaps": result.get("gaps", []),
        "ready_to_write": ready,
        "pending_searches": result.get("suggested_searches", []),
        "research_iteration": iteration + 1,
    }


def route_after_gaps(state: DeepResearchState) -> Literal["search", "verify"]:
    """Route based on gap detection results."""
    if state.get("ready_to_write", False):
        return "verify"
    return "search"


def verify_node(state: DeepResearchState) -> Dict[str, Any]:
    """Verify claims against evidence."""
    evidence = state.get("evidence", [])

    if not evidence:
        return {"verified_claims": [], "confidence": 0.0}

    llm = create_chat_model(model="gpt-4o", temperature=0.2)

    # Get key claims
    claims = [e["fact"] for e in evidence[:20]]
    claims_text = "\n".join([f"- {c}" for c in claims])

    evidence_text = "\n".join([
        f"[{e['evidence_id']}] {e['fact']}"
        for e in evidence
    ])

    prompt = f"""Verify claims against evidence.

CLAIMS:
{claims_text}

EVIDENCE:
{evidence_text}

Respond with JSON array:
[
    {{"claim": "claim text", "supported": true/false, "supporting_evidence": [1,2], "confidence": 0.8}}
]
"""

    response = llm.invoke([
        SystemMessage(content="Verify claims. Output valid JSON array."),
        HumanMessage(content=prompt)
    ])

    verifications = parse_json_array(response.content, default=[])

    verified = []
    supported_count = 0
    for v in verifications:
        if isinstance(v, dict):
            verified.append({
                "claim": v.get("claim", ""),
                "supported": v.get("supported", False),
                "supporting_evidence": v.get("supporting_evidence", []),
                "confidence": v.get("confidence", 0.5)
            })
            if v.get("supported"):
                supported_count += 1

    confidence = supported_count / len(verified) if verified else 0.5

    return {
        "verified_claims": verified,
        "confidence": confidence,
    }


def write_node(state: DeepResearchState) -> Dict[str, Any]:
    """Write the final report."""
    evidence = state.get("evidence", [])
    sources = state.get("sources", {})
    clarified_query = state.get("clarified_query", state.get("query", ""))

    if not evidence:
        return {
            "report": f"# Research Report\n\nInsufficient information found for: {clarified_query}"
        }

    llm = create_chat_model(model="gpt-4o", temperature=0.3)

    evidence_text = "\n".join([
        f"[{e['evidence_id']}] {e['fact']}\n   Source: {e['source_title']}"
        for e in evidence
    ])

    sources_text = "\n".join([
        f"[{s['source_id']}] {s['title']} - {s['url']}"
        for s in sources.values()
    ])

    prompt = f"""Write a comprehensive research report.

QUERY: {clarified_query}

EVIDENCE:
{evidence_text}

SOURCES:
{sources_text}

RULES:
1. ONLY use facts from the evidence
2. EVERY claim must have a citation [1], [2], etc.
3. Acknowledge any limitations
4. Include a Sources section

Write a professional, well-structured report.
"""

    response = llm.invoke([
        SystemMessage(content="Write a research report with citations."),
        HumanMessage(content=prompt)
    ])

    report = response.content

    # Ensure sources section
    if "## Sources" not in report and "## References" not in report:
        report += "\n\n---\n\n## Sources\n\n"
        for s in sources.values():
            report += f"[{s['source_id']}] {s['title']} - {s['url']}\n"

    return {"report": report}


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_deep_research_graph(
    checkpointer: Optional[Any] = None,
    interrupt_on_clarify: bool = True,
) -> StateGraph:
    """
    Build the deep research graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence
        interrupt_on_clarify: Whether to interrupt for HITL clarification

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(DeepResearchState)

    # Add nodes
    graph.add_node("understand", understand_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("gap_detect", gap_detect_node)
    graph.add_node("verify", verify_node)
    graph.add_node("write", write_node)

    # Set entry point
    graph.set_entry_point("understand")

    # Add edges
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {"clarify": "clarify", "plan": "plan"}
    )

    graph.add_edge("clarify", "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "gap_detect")

    graph.add_conditional_edges(
        "gap_detect",
        route_after_gaps,
        {"search": "search", "verify": "verify"}
    )

    graph.add_edge("verify", "write")
    graph.add_edge("write", END)

    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return graph.compile(**compile_kwargs)


def build_deep_research_graph_with_memory(
    interrupt_on_clarify: bool = True,
) -> StateGraph:
    """Build the graph with in-memory checkpointing."""
    checkpointer = MemorySaver()
    return build_deep_research_graph(
        checkpointer=checkpointer,
        interrupt_on_clarify=interrupt_on_clarify,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepResearchState",
    "build_deep_research_graph",
    "build_deep_research_graph_with_memory",
]
