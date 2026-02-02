# Deep Research Agent v8 - Comprehensive Improvement Plan

## Executive Summary

This document provides a detailed analysis of how OpenAI, Anthropic, xAI, Google, and Perplexity implement their deep research agents, followed by a comprehensive improvement plan for Research Studio v8.

---

## Part 1: Industry Analysis

### 1.1 OpenAI Deep Research

**Architecture**: ReAct-based agent powered by o3 model optimized for extended reasoning.

**Key Innovations**:
| Component | Implementation |
|-----------|----------------|
| **Model** | o3 model with expanded "attention span" for long chains of thought |
| **Training** | End-to-end RL on simulated research environments with tool access |
| **Loop** | Plan → Act → Observe (ReAct paradigm) with backtracking |
| **Tools** | Web browser, PDF parser, image analyzer, Python code execution |
| **Duration** | Up to 30 minutes, 21+ sources, dozens of search queries |
| **Summarization** | o3-mini used to summarize chains of thought |

**Key Techniques**:
1. **Backtracking**: When paths are unfruitful (paywalls, irrelevant), pivot to alternatives
2. **Iterative Refinement**: Search → Read → Decide next query based on learnings
3. **Unified Architecture**: Deep Research + Operator share state for fluid transitions
4. **Multimodal**: Handles HTML, PDFs, images, charts, CSVs

**Performance**: 26.6% on Humanity's Last Exam (3,000 questions across 100 subjects)

**Source**: [OpenAI Introducing Deep Research](https://openai.com/index/introducing-deep-research/), [How Deep Research Works](https://blog.promptlayer.com/how-deep-research-works/)

---

### 1.2 Anthropic Multi-Agent Research System

**Architecture**: Orchestrator-Worker pattern with parallel subagents.

**Key Innovations**:
| Component | Implementation |
|-----------|----------------|
| **Lead Agent** | Claude Opus 4 - analyzes query, develops strategy, spawns subagents |
| **Subagents** | Claude Sonnet 4 - 3-5 parallel workers with own context windows |
| **Parallelism** | Two levels: subagent spawning + tool calls within subagents |
| **Search** | Brave Search integration via web_search tool |
| **Compression** | Subagents condense findings before returning to lead agent |

**Key Techniques**:
1. **Separation of Concerns**: Each subagent has distinct tools, prompts, trajectories
2. **Parallel Context Windows**: Distribute work to add capacity for reasoning
3. **Progressive Search**: Multiple searches using earlier results to inform queries
4. **Token Efficiency**: Programmatic tool calling reduced tokens by 37%

**Performance**: 90.2% improvement over single-agent Claude Opus 4 on internal evals

**Resource Usage**: Multi-agent uses ~15x more tokens than single chat

**Source**: [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system), [ByteByteGo Analysis](https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent)

---

### 1.3 xAI Grok DeepSearch

**Architecture**: Distributed crawler network + on-demand reasoning agent.

**Key Innovations**:
| Component | Implementation |
|-----------|----------------|
| **Model** | Grok 3 trained on 200K H100 GPUs |
| **Search** | Real-time web + X/Twitter signals |
| **Reasoning** | "Think" mode for slower, deliberate chain-of-thought |
| **Speed** | 67ms average response latency |
| **Context** | 128K token window |

**Key Techniques**:
1. **Dynamic Workflows**: Static vs dynamic based on query complexity
2. **Sub-query Generation**: Generates specific sub-queries from main question
3. **Conflict Resolution**: Reasons about conflicting facts and opinions
4. **DeeperSearch**: Extended version with more reasoning depth

**Training**: RL at unprecedented scale for chain-of-thought refinement, backtracking, error correction

**Source**: [xAI Grok 3 Announcement](https://x.ai/news/grok-3), [Oreate AI Guide](https://www.oreateai.com/blog/unlocking-deep-research-in-grok-xai-a-comprehensive-guide/)

---

### 1.4 Google Gemini Deep Research

**Architecture**: Agentic system combining Gemini + Google Search + web browsing in continuous reasoning loop.

**Key Innovations**:
| Component | Implementation |
|-----------|----------------|
| **Model** | Gemini 3 Pro, trained to reduce hallucinations |
| **Context** | 1M token window + RAG for session memory |
| **Task Manager** | Asynchronous with shared state between planner and task models |
| **Sources** | Web + Gmail + Drive + Chat integrations |
| **Multimodal** | Images, PDFs, audio, video as research inputs |

**Key Techniques**:
1. **Iterative Planning**: Formulate queries → Read → Identify gaps → Search again
2. **Graceful Error Recovery**: Shared state allows recovery without restart
3. **Truly Async**: Works in background, notifies when done
4. **Knowledge Gap Detection**: At each step, identifies missing information

**Performance**: 46.4% on Humanity's Last Exam, 66.1% on DeepSearchQA, 59.2% on BrowseComp

**Source**: [Google AI Deep Research](https://ai.google.dev/gemini-api/docs/deep-research), [Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/deep-research-agent-gemini-api/)

---

### 1.5 Perplexity Deep Research

**Architecture**: Multi-model with dynamic routing + Comet multi-agent framework.

**Key Innovations**:
| Component | Implementation |
|-----------|----------------|
| **Routing** | Dynamic routing to different engines (conversational, research, coding) |
| **Source Ranking** | E-E-A-T scoring (Experience, Expertise, Authority, Trust) |
| **Comet Framework** | Retrieval Agent + Synthesis Agent + Verification Agent |
| **Scale** | 200M daily queries |

**Key Techniques**:
1. **Research Tree**: Branches query into sub-queries (tech viability, case studies, economics)
2. **Multi-Agent Verification**: Verification agent validates citations against live sources
3. **30% Faster**: Than Google for market intelligence tasks

**Source**: [Perplexity Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api), [Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)

---

## Part 2: Key Patterns Across All Systems

### 2.1 Common Architectural Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL PATTERNS                           │
├─────────────────────────────────────────────────────────────────┤
│  1. ReAct Loop (Plan → Act → Observe → Repeat)                 │
│  2. Multi-Agent / Orchestrator-Worker Architecture             │
│  3. Iterative Query Refinement (not single search)             │
│  4. Backtracking on Dead Ends                                  │
│  5. Parallel Execution where possible                          │
│  6. Source Credibility Scoring                                 │
│  7. Citation Verification / Cross-Validation                   │
│  8. Long Context + RAG Hybrid                                  │
│  9. Multimodal Input Support                                   │
│ 10. Asynchronous Operation with Shared State                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Query Strategy Comparison

| System | Query Generation | Refinement | Parallelism |
|--------|-----------------|------------|-------------|
| OpenAI | Implicit in CoT | Based on results read | Sequential with backtrack |
| Anthropic | Lead agent generates | Subagents refine independently | 3-5 subagents parallel |
| xAI | Sub-query decomposition | Think mode for depth | Dynamic workflows |
| Google | Research plan from prompt | Gap detection → new queries | Task-level parallel |
| Perplexity | Research tree branching | Multi-step in Pro Search | Comet multi-agent |

### 2.3 Hallucination Prevention Techniques

1. **Grounding**: Only generate from retrieved evidence (RAG)
2. **Citation Verification**: Match claims to source spans
3. **Cross-Validation**: Multiple sources for same claim
4. **Confidence Scoring**: 0-1 scale with justification
5. **URL Verification**: Check cited URLs actually exist
6. **Multi-Model Ensemble**: Compare outputs, take consensus

---

## Part 3: Gap Analysis - Current v7 vs Industry Best Practices

### 3.1 What v7 Does Well

| Feature | Current Implementation | Industry Alignment |
|---------|----------------------|-------------------|
| Entity Disambiguation | ERA-CoT + pattern matching + discovery search | Unique - most don't do this |
| Human-in-the-Loop | LangGraph interrupt at clarify node | Similar to OpenAI approach |
| Parallel Workers | Fan-out via LangGraph Send | Matches Anthropic pattern |
| Query Templates | 3-2-1 strategy per query type | Good, similar to Perplexity |
| Source Deduplication | URL normalization + longest snippet | Standard practice |
| Relevance Filtering | LLM checks for entity match | Similar to industry |
| Claims → Citations | Trust engine pipeline | Standard grounding approach |
| Caching | Search + page content | Industry standard |

### 3.2 Critical Gaps

| Gap | Industry Practice | v7 Current | Priority |
|-----|------------------|-----------|----------|
| **Backtracking** | Pivot when path fails | No backtracking | 🔴 Critical |
| **Iterative Refinement** | Search → Learn → Re-search | Single round of queries | 🔴 Critical |
| **Knowledge Gap Detection** | Identify missing info | None | 🔴 Critical |
| **Multi-Agent Architecture** | Orchestrator + Workers | Flat worker pattern | 🟡 High |
| **Source Credibility Scoring** | E-E-A-T, domain trust | Basic quality_score | 🟡 High |
| **Citation Verification** | Span-level matching | Section-level fallback | 🟡 High |
| **Async Operation** | Background + notify | Synchronous only | 🟢 Medium |
| **Multimodal Inputs** | PDF, images, video | Text only | 🟢 Medium |
| **Code Execution** | Python for analysis | None | 🟢 Medium |
| **Confidence Scores** | Per-claim 0-1 scores | Binary pass/fail | 🟢 Medium |

---

## Part 4: Comprehensive Improvement Plan

### Phase 0: Configuration & State Enhancements

#### 0.1 Enhanced Configuration (config.py)

```python
@dataclass(frozen=True)
class V8Config:
    # === EXISTING (kept) ===
    max_rounds: int = 2
    queries_per_round: int = 10
    tavily_max_results: int = 6
    # ...

    # === NEW: Iterative Research ===
    max_research_iterations: int = 3           # How many refine loops
    min_confidence_to_proceed: float = 0.7     # Below this, do another iteration
    max_research_time_minutes: float = 10.0    # Hard timeout
    enable_backtracking: bool = True           # Pivot on dead ends

    # === NEW: Multi-Agent ===
    use_multi_agent: bool = True               # Enable orchestrator pattern
    max_subagents: int = 5                     # Parallel subagents
    subagent_model: str = "gpt-4o-mini"        # Cheaper model for workers
    orchestrator_model: str = "gpt-4o"         # Smarter model for lead

    # === NEW: Source Credibility ===
    enable_eeat_scoring: bool = True           # E-E-A-T scoring
    trusted_domains: List[str] = field(default_factory=lambda: [
        ".edu", ".gov", ".org", "arxiv.org", "nature.com",
        "wikipedia.org", "github.com", "scholar.google.com"
    ])
    low_trust_domains: List[str] = field(default_factory=lambda: [
        "pinterest.com", "quora.com", "medium.com"  # Not always reliable
    ])
    min_source_credibility: float = 0.4        # Filter below this

    # === NEW: Citation Verification ===
    enable_span_verification: bool = True      # Match claims to exact spans
    require_multi_source_claims: bool = False  # Require 2+ sources per claim
    hallucination_threshold: float = 0.3       # Flag if >30% unsupported

    # === NEW: Async & Performance ===
    enable_async_mode: bool = False            # Background operation
    notify_on_complete: bool = True            # Callback when done
```

#### 0.2 Enhanced State (state.py)

```python
class AgentState(MessagesState):
    # === EXISTING (kept) ===
    original_query: str
    discovery: DiscoveryResult
    # ...

    # === NEW: Iterative Research State ===
    research_iteration: int                    # Current iteration (0, 1, 2...)
    knowledge_gaps: List[KnowledgeGap]         # What's still missing
    research_trajectory: List[TrajectoryStep]  # History of actions taken
    dead_ends: List[DeadEnd]                   # Paths that failed

    # === NEW: Confidence Tracking ===
    overall_confidence: float                  # 0-1 research confidence
    section_confidence: Dict[str, float]       # Per-section confidence

    # === NEW: Source Credibility ===
    source_credibility: Dict[str, SourceCredibility]  # url -> credibility

    # === NEW: Citation Verification ===
    verified_citations: List[VerifiedCitation]
    unverified_claims: List[str]               # CIDs without verification
    hallucination_score: float                 # % of unsupported claims

# New TypedDicts
class KnowledgeGap(TypedDict):
    section: str
    description: str
    suggested_queries: List[str]
    priority: float  # 0-1

class TrajectoryStep(TypedDict):
    iteration: int
    action: str  # "search", "read", "refine", "backtrack"
    query: str
    result_summary: str
    confidence_delta: float

class DeadEnd(TypedDict):
    query: str
    reason: str  # "no_results", "irrelevant", "paywall", "low_credibility"
    iteration: int

class SourceCredibility(TypedDict):
    url: str
    domain_trust: float      # 0-1 based on domain
    freshness: float         # 0-1 based on publish date
    authority: float         # 0-1 based on citations, author
    content_quality: float   # 0-1 based on depth, structure
    overall: float           # weighted average

class VerifiedCitation(TypedDict):
    cid: str
    eid: str
    claim_text: str
    evidence_span: str       # Exact text from source
    match_score: float       # 0-1 semantic similarity
    verified: bool
```

---

### Phase 1: Discovery & Disambiguation (ENHANCED)

#### Current Flow
```
analyzer → discovery → [clarify] → planner
```

#### New Flow
```
analyzer → discovery → confidence_check → [clarify | auto_refine] → planner
```

#### 1.1 Enhanced Analyzer Node

**Additions**:
1. **Query Complexity Assessment**: Simple (1 search) vs Complex (multi-iteration)
2. **Research Scope Estimation**: Narrow vs Broad
3. **Confidence Prediction**: How confident can we expect to be?

```python
# New output fields from analyzer
{
    "query_complexity": "complex",        # simple | moderate | complex
    "estimated_scope": "narrow",          # narrow | medium | broad
    "predicted_confidence": 0.7,          # Expected achievable confidence
    "suggested_iterations": 2,            # How many research rounds
    "decomposed_questions": [             # Sub-questions to answer
        "What is their educational background?",
        "What projects have they worked on?",
        "What is their current role?"
    ],
}
```

#### 1.2 New Node: `confidence_check_node`

**Purpose**: Decide if we need more discovery or can proceed.

```python
def confidence_check_node(state: AgentState) -> Dict[str, Any]:
    """
    After discovery, assess if confidence is sufficient.

    Routes to:
    - "clarify" if ambiguous AND human-in-loop enabled
    - "auto_refine" if ambiguous but can auto-resolve
    - "planner" if confident
    """
    discovery = state.get("discovery", {})
    confidence = discovery.get("confidence", 0)
    candidates = discovery.get("entity_candidates", [])
    query_type = discovery.get("query_type", "general")

    # Concept/technical queries rarely need refinement
    if query_type in ("concept", "technical"):
        return {"route": "planner"}

    # High confidence - proceed
    if confidence >= 0.85:
        return {"route": "planner"}

    # Multiple candidates with similar confidence - need disambiguation
    if len(candidates) > 1:
        top_two = sorted(candidates, key=lambda c: c.get("confidence", 0), reverse=True)[:2]
        if top_two[0]["confidence"] - top_two[1]["confidence"] < 0.2:
            # Ambiguous - need clarification
            return {"route": "clarify"}

    # Low confidence but single candidate - try auto-refine
    if confidence < 0.7 and len(candidates) == 1:
        return {
            "route": "auto_refine",
            "refinement_queries": _generate_refinement_queries(candidates[0])
        }

    return {"route": "planner"}
```

#### 1.3 New Node: `auto_refine_node`

**Purpose**: Automatically refine discovery without human input.

```python
def auto_refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Do additional discovery searches to increase confidence.

    Strategies:
    1. Search with different query formulations
    2. Look for unique identifiers (LinkedIn, personal site)
    3. Cross-reference with known affiliations
    """
    candidate = state.get("selected_entity") or state["discovery"]["entity_candidates"][0]
    refinement_queries = state.get("refinement_queries", [])

    # Generate targeted refinement queries
    if not refinement_queries:
        name = candidate.get("name", "")
        identifiers = candidate.get("identifiers", [])

        refinement_queries = [
            f'"{name}" LinkedIn profile',
            f'"{name}" site:linkedin.com OR site:github.com',
            f'"{name}" {identifiers[0] if identifiers else ""} biography',
        ]

    # Execute refinement searches
    all_results = []
    for query in refinement_queries:
        results = cached_search(query, max_results=5, lane="general")
        all_results.extend(results)

    # Re-analyze with additional context
    # ... (LLM call to reassess confidence)

    return {
        "discovery": {**state["discovery"], "confidence": new_confidence},
        "selected_entity": refined_entity,
    }
```

---

### Phase 2: Research Planning (ENHANCED)

#### Current Flow
```
planner → [workers in parallel]
```

#### New Flow
```
planner → orchestrator → [subagents in parallel] → synthesizer → gap_detector → [loop or proceed]
```

#### 2.1 Enhanced Planner with Research Tree

**Purpose**: Generate a hierarchical research plan, not just flat queries.

```python
PLANNER_V2_SYSTEM = """You are a research planning expert.

Create a RESEARCH TREE that decomposes the question into:
1. Primary questions (must answer)
2. Secondary questions (should answer if time permits)
3. Tertiary questions (nice to have)

For each question, specify:
- Search queries (3-6 words, quoted names)
- Target sources (academic, news, official, community)
- Expected confidence contribution

Return JSON:
{
  "topic": "...",
  "research_tree": {
    "primary": [
      {
        "question": "What is their educational background?",
        "queries": ["\"Name\" university education", "\"Name\" degree"],
        "target_sources": ["academic", "official"],
        "confidence_weight": 0.3
      }
    ],
    "secondary": [...],
    "tertiary": [...]
  },
  "estimated_total_confidence": 0.85,
  "estimated_search_count": 15
}
"""
```

#### 2.2 New Node: `orchestrator_node`

**Purpose**: Lead agent that coordinates subagents (Anthropic pattern).

```python
def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Lead agent that:
    1. Analyzes the research plan
    2. Assigns questions to subagents
    3. Tracks overall progress
    4. Decides when to stop or continue
    """
    research_tree = state["plan"]["research_tree"]
    primary_questions = research_tree["primary"]

    # Spawn subagents for primary questions (parallel)
    subagent_assignments = []
    for i, question in enumerate(primary_questions[:5]):  # Max 5 subagents
        subagent_assignments.append({
            "subagent_id": f"SA{i+1}",
            "question": question["question"],
            "queries": question["queries"],
            "target_sources": question["target_sources"],
        })

    return {
        "subagent_assignments": subagent_assignments,
        "orchestrator_state": {
            "phase": "primary_research",
            "questions_assigned": len(subagent_assignments),
            "questions_completed": 0,
        }
    }
```

#### 2.3 Enhanced Worker → Subagent

**Key Changes**:
1. Each subagent has its own context window
2. Subagents can do iterative search within their scope
3. Subagents return compressed findings + confidence

```python
def subagent_node(state: AgentState) -> Dict[str, Any]:
    """
    Subagent that independently researches assigned question.

    Can do multiple searches, read pages, and refine queries
    before returning findings to orchestrator.
    """
    assignment = state["subagent_assignment"]
    question = assignment["question"]
    queries = assignment["queries"]

    # Iterative search within subagent
    findings = []
    confidence = 0.0
    max_iterations = 3

    for iteration in range(max_iterations):
        # Execute current queries
        for query in queries:
            results = cached_search(query, max_results=6)
            if not results:
                # Backtrack: try alternative query
                queries = _generate_alternative_queries(query)
                continue

            # Fetch and extract evidence
            evidence = _extract_evidence(results)
            findings.extend(evidence)

            # Assess confidence for this question
            confidence = _assess_question_confidence(question, findings)

            if confidence >= 0.8:
                break  # Sufficient for this question

        if confidence >= 0.8:
            break

        # Generate refined queries based on what we learned
        queries = _refine_queries(question, findings)

    # Compress findings for orchestrator
    compressed = _compress_findings(findings, max_tokens=2000)

    return {
        "subagent_findings": [{
            "subagent_id": assignment["subagent_id"],
            "question": question,
            "findings": compressed,
            "evidence_ids": [f["eid"] for f in findings],
            "confidence": confidence,
            "iterations_used": iteration + 1,
        }],
        "raw_evidence": findings,
        "done_subagents": 1,
    }
```

#### 2.4 New Node: `gap_detector_node`

**Purpose**: After research round, identify what's still missing.

```python
GAP_DETECTOR_SYSTEM = """Analyze research findings and identify knowledge gaps.

Given:
- Original question
- Research outline
- Current findings by section
- Current confidence by section

Identify:
1. Sections with low coverage (< 0.6 confidence)
2. Unanswered sub-questions
3. Conflicting information needing resolution
4. Missing perspectives (e.g., only positive sources, no critiques)

Return JSON:
{
  "overall_confidence": 0.72,
  "section_gaps": [
    {
      "section": "Work & Achievements",
      "current_confidence": 0.45,
      "gap_description": "No specific project details found",
      "suggested_queries": ["\"Name\" projects portfolio", "\"Name\" github contributions"]
    }
  ],
  "conflicts": [
    {
      "topic": "Current role",
      "sources_disagreeing": ["S1", "S3"],
      "resolution_query": "\"Name\" current position 2024"
    }
  ],
  "recommendation": "continue" | "sufficient"
}
"""

def gap_detector_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze current research state and identify gaps.

    Decides whether to:
    - Continue with another research iteration
    - Proceed to synthesis (sufficient coverage)
    """
    evidence = state.get("evidence", [])
    claims = state.get("claims", [])
    plan = state.get("plan", {})
    outline = plan.get("outline", [])

    # Group evidence by section
    section_evidence = defaultdict(list)
    for e in evidence:
        section_evidence[e["section"]].append(e)

    # Calculate per-section confidence
    section_confidence = {}
    for section in outline:
        ev_count = len(section_evidence.get(section, []))
        # Heuristic: 3+ evidence items = high confidence
        section_confidence[section] = min(1.0, ev_count / 3)

    # LLM analysis for gaps
    llm = create_chat_model(model="gpt-4o-mini")
    # ... (prompt with current state)

    gap_analysis = parse_json_object(resp.content, default={})

    # Decision
    overall_confidence = gap_analysis.get("overall_confidence", 0.5)
    recommendation = gap_analysis.get("recommendation", "continue")

    current_iteration = state.get("research_iteration", 0)
    max_iterations = state.get("max_research_iterations", 3)

    if recommendation == "sufficient" or current_iteration >= max_iterations:
        return {
            "knowledge_gaps": [],
            "overall_confidence": overall_confidence,
            "proceed_to_synthesis": True,
        }

    # Generate new queries for gaps
    new_queries = []
    for gap in gap_analysis.get("section_gaps", []):
        for q in gap.get("suggested_queries", []):
            new_queries.append({
                "query": q,
                "section": gap["section"],
                "priority": 1.0 - gap["current_confidence"],
            })

    return {
        "knowledge_gaps": gap_analysis.get("section_gaps", []),
        "overall_confidence": overall_confidence,
        "proceed_to_synthesis": False,
        "refinement_queries": new_queries,
        "research_iteration": current_iteration + 1,
    }
```

---

### Phase 3: Source Credibility & Ranking (NEW)

#### 3.1 New Node: `credibility_scorer_node`

**Purpose**: Score sources using E-E-A-T and domain trust.

```python
CREDIBILITY_SYSTEM = """Score source credibility on multiple dimensions.

For each source, evaluate:

1. DOMAIN TRUST (0-1):
   - .edu, .gov = 0.9
   - Major news (nytimes, bbc) = 0.8
   - Wikipedia = 0.75
   - Official company sites = 0.7
   - Medium, Quora = 0.4
   - Unknown = 0.5

2. FRESHNESS (0-1):
   - < 6 months = 1.0
   - 6-12 months = 0.8
   - 1-2 years = 0.6
   - 2-5 years = 0.4
   - > 5 years = 0.2
   - Unknown = 0.5

3. AUTHORITY (0-1):
   - Has author credentials = +0.2
   - Cites other sources = +0.2
   - Is cited by others = +0.3
   - Official/primary source = +0.3

4. CONTENT QUALITY (0-1):
   - Detailed, specific = 0.9
   - Moderate depth = 0.6
   - Thin/listicle = 0.3

Return JSON array:
[{"sid": "S1", "domain_trust": 0.8, "freshness": 0.9, "authority": 0.7, "content_quality": 0.8, "overall": 0.8}]
"""

def credibility_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    Score each source's credibility.
    Filter out low-credibility sources.
    """
    sources = state.get("sources", [])

    # Rule-based domain scoring
    domain_scores = {}
    for s in sources:
        url = s.get("url", "")
        domain = urlparse(url).hostname or ""

        # Domain trust rules
        if any(d in domain for d in [".edu", ".gov"]):
            domain_scores[s["sid"]] = 0.9
        elif any(d in domain for d in ["wikipedia.org", "arxiv.org"]):
            domain_scores[s["sid"]] = 0.85
        elif any(d in domain for d in ["github.com", "linkedin.com"]):
            domain_scores[s["sid"]] = 0.75
        elif any(d in domain for d in ["medium.com", "quora.com"]):
            domain_scores[s["sid"]] = 0.4
        else:
            domain_scores[s["sid"]] = 0.5

    # LLM scoring for content quality and authority
    llm = create_chat_model(model="gpt-4o-mini")
    # ... (evaluate content quality based on snippets)

    # Combine scores
    source_credibility = {}
    for s in sources:
        sid = s["sid"]
        source_credibility[sid] = {
            "domain_trust": domain_scores.get(sid, 0.5),
            "freshness": _calculate_freshness(s.get("published_date")),
            "authority": llm_scores.get(sid, {}).get("authority", 0.5),
            "content_quality": llm_scores.get(sid, {}).get("content_quality", 0.5),
        }

        # Weighted average
        weights = {"domain_trust": 0.3, "freshness": 0.2, "authority": 0.25, "content_quality": 0.25}
        source_credibility[sid]["overall"] = sum(
            source_credibility[sid][k] * weights[k] for k in weights
        )

    # Filter low-credibility sources
    min_credibility = state.get("min_source_credibility", 0.4)
    filtered_sources = [
        s for s in sources
        if source_credibility[s["sid"]]["overall"] >= min_credibility
    ]

    # Also filter evidence
    valid_sids = {s["sid"] for s in filtered_sources}
    filtered_evidence = [
        e for e in state.get("evidence", [])
        if e.get("sid") in valid_sids
    ]

    return {
        "sources": filtered_sources,
        "evidence": filtered_evidence,
        "source_credibility": source_credibility,
    }
```

---

### Phase 4: Trust Engine (ENHANCED)

#### Current Flow
```
claims → cite → verify → writer
```

#### New Flow
```
claims → cite → span_verify → cross_validate → confidence_score → writer
```

#### 4.1 New Node: `span_verify_node`

**Purpose**: Verify each claim against exact text spans in sources.

```python
SPAN_VERIFY_SYSTEM = """Verify if claims are supported by exact evidence spans.

For each claim, find the EXACT TEXT in the evidence that supports it.
If no exact support exists, mark as unverified.

Return JSON:
[
  {
    "cid": "C1",
    "verified": true,
    "evidence_span": "Pranjal Chalise graduated from Amherst College in 2024",
    "source_eid": "E3",
    "match_confidence": 0.95
  },
  {
    "cid": "C2",
    "verified": false,
    "reason": "No evidence mentions this specific fact"
  }
]
"""

def span_verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify each claim against exact evidence spans.

    This prevents:
    - Hallucinated facts
    - Misattributed claims
    - Over-generalization
    """
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    citations = state.get("citations", [])

    llm = create_chat_model(model="gpt-4o-mini", temperature=0)

    # Build evidence lookup
    eid_to_evidence = {e["eid"]: e for e in evidence}
    cid_to_eids = {c["cid"]: c["eids"] for c in citations}

    verified_citations = []
    unverified_claims = []

    for claim in claims:
        cid = claim["cid"]
        claim_text = claim["text"]
        eids = cid_to_eids.get(cid, [])

        # Get relevant evidence texts
        evidence_texts = []
        for eid in eids:
            if eid in eid_to_evidence:
                evidence_texts.append({
                    "eid": eid,
                    "text": eid_to_evidence[eid]["text"]
                })

        # LLM verification
        resp = llm.invoke([
            SystemMessage(content=SPAN_VERIFY_SYSTEM),
            HumanMessage(content=f"Claim: {claim_text}\n\nEvidence:\n{json.dumps(evidence_texts)}")
        ])

        result = parse_json_object(resp.content, default={})

        if result.get("verified"):
            verified_citations.append({
                "cid": cid,
                "eid": result.get("source_eid"),
                "claim_text": claim_text,
                "evidence_span": result.get("evidence_span"),
                "match_score": result.get("match_confidence", 0.5),
                "verified": True,
            })
        else:
            unverified_claims.append(cid)

    # Calculate hallucination score
    total_claims = len(claims)
    verified_count = len(verified_citations)
    hallucination_score = 1 - (verified_count / total_claims) if total_claims > 0 else 0

    return {
        "verified_citations": verified_citations,
        "unverified_claims": unverified_claims,
        "hallucination_score": hallucination_score,
    }
```

#### 4.2 New Node: `cross_validate_node`

**Purpose**: Check if claims are supported by multiple sources.

```python
def cross_validate_node(state: AgentState) -> Dict[str, Any]:
    """
    For important claims, check if multiple sources agree.
    Flag claims that only have single-source support.
    """
    verified_citations = state.get("verified_citations", [])
    evidence = state.get("evidence", [])

    # Group evidence by SID
    eid_to_sid = {e["eid"]: e["sid"] for e in evidence}

    cross_validated = []
    single_source = []

    for vc in verified_citations:
        cid = vc["cid"]
        claim_text = vc["claim_text"]

        # Find all evidence items that could support this claim
        supporting_sids = set()
        for e in evidence:
            if _semantic_match(claim_text, e["text"]) > 0.6:
                supporting_sids.add(e["sid"])

        if len(supporting_sids) >= 2:
            cross_validated.append({
                **vc,
                "cross_validated": True,
                "supporting_sources": list(supporting_sids),
            })
        else:
            single_source.append({
                **vc,
                "cross_validated": False,
                "supporting_sources": list(supporting_sids),
            })

    return {
        "cross_validated_claims": cross_validated,
        "single_source_claims": single_source,
    }
```

#### 4.3 New Node: `confidence_scorer_node` (for claims)

**Purpose**: Assign confidence scores to each claim.

```python
def claim_confidence_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    Calculate per-claim and overall confidence scores.

    Factors:
    - Span verification match score
    - Cross-validation (multiple sources)
    - Source credibility of supporting sources
    - Recency of evidence
    """
    verified_citations = state.get("verified_citations", [])
    cross_validated = state.get("cross_validated_claims", [])
    single_source = state.get("single_source_claims", [])
    source_credibility = state.get("source_credibility", {})

    claim_confidence = {}

    for vc in verified_citations:
        cid = vc["cid"]

        # Base: span match score
        base_score = vc.get("match_score", 0.5)

        # Bonus: cross-validation
        is_cross_validated = cid in [cv["cid"] for cv in cross_validated]
        cross_bonus = 0.15 if is_cross_validated else 0

        # Weight by source credibility
        supporting_sids = vc.get("supporting_sources", [])
        avg_source_cred = sum(
            source_credibility.get(sid, {}).get("overall", 0.5)
            for sid in supporting_sids
        ) / max(len(supporting_sids), 1)

        # Final confidence
        confidence = min(1.0, base_score * 0.5 + avg_source_cred * 0.35 + cross_bonus)
        claim_confidence[cid] = confidence

    # Overall confidence
    if claim_confidence:
        overall = sum(claim_confidence.values()) / len(claim_confidence)
    else:
        overall = 0.0

    return {
        "claim_confidence": claim_confidence,
        "overall_confidence": overall,
    }
```

---

### Phase 5: Report Generation (ENHANCED)

#### 5.1 Enhanced Writer with Confidence Indicators

```python
WRITER_V2_SYSTEM = """Write a research report with confidence indicators.

Rules:
1. Use ONLY the verified claims provided
2. Include confidence indicators:
   - ✓✓ = High confidence (cross-validated, 0.8+)
   - ✓ = Medium confidence (verified, 0.6-0.8)
   - ⚠ = Low confidence (single source, < 0.6)
3. For low-confidence claims, note "according to [source]"
4. Include a "Research Confidence" section at the end
5. List any knowledge gaps honestly

Output markdown with:
- Title
- Executive Summary
- Sections with citations [S#]
- Research Confidence section
- Knowledge Gaps (if any)
- Sources
"""

def writer_node_v2(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced writer with:
    - Confidence indicators per claim
    - Knowledge gap acknowledgment
    - Source credibility display
    """
    claims = state.get("claims", [])
    claim_confidence = state.get("claim_confidence", {})
    knowledge_gaps = state.get("knowledge_gaps", [])
    overall_confidence = state.get("overall_confidence", 0.5)
    unverified_claims = state.get("unverified_claims", [])

    # Filter out unverified claims
    verified_claims = [c for c in claims if c["cid"] not in unverified_claims]

    # Add confidence indicators
    claims_with_confidence = []
    for claim in verified_claims:
        cid = claim["cid"]
        conf = claim_confidence.get(cid, 0.5)

        if conf >= 0.8:
            indicator = "✓✓"
        elif conf >= 0.6:
            indicator = "✓"
        else:
            indicator = "⚠"

        claims_with_confidence.append({
            **claim,
            "confidence": conf,
            "indicator": indicator,
        })

    # Build knowledge gaps section
    gaps_text = ""
    if knowledge_gaps:
        gaps_text = "### Knowledge Gaps\n\n"
        for gap in knowledge_gaps:
            gaps_text += f"- **{gap['section']}**: {gap['description']}\n"

    # LLM report generation
    # ...

    return {
        "report": report,
        "research_metadata": {
            "overall_confidence": overall_confidence,
            "verified_claims": len(verified_claims),
            "total_claims": len(claims),
            "knowledge_gaps": len(knowledge_gaps),
        }
    }
```

---

### Phase 6: Backtracking & Error Recovery (NEW)

#### 6.1 New Node: `backtrack_handler_node`

**Purpose**: Handle failed searches and pivot to alternatives.

```python
def backtrack_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    When a search path fails, generate alternative approaches.

    Failure types:
    1. No results → broaden query
    2. Irrelevant results → refine with context
    3. Paywall/blocked → try alternative domains
    4. Low credibility → find authoritative sources
    """
    dead_ends = state.get("dead_ends", [])
    research_trajectory = state.get("research_trajectory", [])

    if not dead_ends:
        return {}

    latest_dead_end = dead_ends[-1]
    reason = latest_dead_end["reason"]
    failed_query = latest_dead_end["query"]

    alternative_queries = []

    if reason == "no_results":
        # Broaden: remove quotes, add synonyms
        alternative_queries = [
            failed_query.replace('"', ''),  # Remove quotes
            f"{failed_query} OR background OR profile",  # Add alternatives
        ]

    elif reason == "irrelevant":
        # Add more context
        primary_anchor = state.get("primary_anchor", "")
        anchor_terms = state.get("anchor_terms", [])
        alternative_queries = [
            f'"{primary_anchor}" {anchor_terms[0] if anchor_terms else ""} -unrelated',
            f'"{primary_anchor}" exact person NOT company',
        ]

    elif reason == "paywall":
        # Try alternative domains
        alternative_queries = [
            f'{failed_query} site:wikipedia.org OR site:github.com',
            f'{failed_query} -site:wsj.com -site:nytimes.com',  # Exclude paywalled
        ]

    elif reason == "low_credibility":
        # Target authoritative sources
        alternative_queries = [
            f'{failed_query} site:.edu OR site:.gov',
            f'{failed_query} official',
        ]

    # Log trajectory
    trajectory_step = {
        "iteration": state.get("research_iteration", 0),
        "action": "backtrack",
        "query": failed_query,
        "result_summary": f"Failed: {reason}",
        "alternative_queries": alternative_queries,
    }

    return {
        "research_trajectory": research_trajectory + [trajectory_step],
        "refinement_queries": alternative_queries,
    }
```

---

### Phase 7: Graph Architecture (UPDATED)

#### 7.1 New Graph Structure

```python
def build_v8_graph(checkpointer=None, interrupt_on_clarify=True):
    """
    v8 Graph with:
    - Multi-agent orchestration
    - Iterative research loop
    - Backtracking
    - Enhanced trust engine
    """
    g = StateGraph(AgentState)

    # === Phase 1: Discovery ===
    g.add_node("analyzer", analyzer_node_v2)
    g.add_node("discovery", discovery_node)
    g.add_node("confidence_check", confidence_check_node)
    g.add_node("auto_refine", auto_refine_node)
    g.add_node("clarify", clarify_node)

    # === Phase 2: Research (Multi-Agent) ===
    g.add_node("planner", planner_node_v2)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("subagent", subagent_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("gap_detector", gap_detector_node)
    g.add_node("backtrack", backtrack_handler_node)

    # === Phase 3: Source Processing ===
    g.add_node("reduce", reducer_node)
    g.add_node("credibility", credibility_scorer_node)
    g.add_node("ranker", ranker_node_v2)

    # === Phase 4: Trust Engine ===
    g.add_node("claims", claims_node)
    g.add_node("cite", cite_node)
    g.add_node("span_verify", span_verify_node)
    g.add_node("cross_validate", cross_validate_node)
    g.add_node("claim_confidence", claim_confidence_scorer_node)

    # === Phase 5: Output ===
    g.add_node("writer", writer_node_v2)

    # === Edges ===

    # Discovery flow
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "discovery")
    g.add_conditional_edges("discovery", route_confidence_check, {
        "confidence_check": "confidence_check",
    })
    g.add_conditional_edges("confidence_check", route_after_confidence, {
        "clarify": "clarify",
        "auto_refine": "auto_refine",
        "planner": "planner",
    })
    g.add_edge("auto_refine", "confidence_check")  # Loop back
    g.add_edge("clarify", "planner")

    # Research flow (multi-agent)
    g.add_edge("planner", "orchestrator")
    g.add_conditional_edges("orchestrator", fanout_subagents)  # Parallel subagents
    g.add_edge("subagent", "synthesizer")
    g.add_conditional_edges("synthesizer", route_after_synthesis, {
        "gap_detector": "gap_detector",
    })
    g.add_conditional_edges("gap_detector", route_after_gaps, {
        "orchestrator": "orchestrator",  # Loop for more research
        "reduce": "reduce",              # Proceed to processing
        "backtrack": "backtrack",        # Handle failures
    })
    g.add_edge("backtrack", "orchestrator")  # Retry with alternatives

    # Source processing
    g.add_edge("reduce", "credibility")
    g.add_edge("credibility", "ranker")

    # Trust engine
    g.add_edge("ranker", "claims")
    g.add_edge("claims", "cite")
    g.add_edge("cite", "span_verify")
    g.add_edge("span_verify", "cross_validate")
    g.add_edge("cross_validate", "claim_confidence")

    # Output
    g.add_edge("claim_confidence", "writer")
    g.add_edge("writer", END)

    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)
```

#### 7.2 Visual Graph

```
                                    ┌──────────────┐
                                    │    START     │
                                    └──────┬───────┘
                                           │
                                    ┌──────▼───────┐
                                    │   Analyzer   │
                                    └──────┬───────┘
                                           │
                                    ┌──────▼───────┐
                                    │  Discovery   │
                                    └──────┬───────┘
                                           │
                                    ┌──────▼───────┐
                                    │ Confidence   │
                                    │   Check      │
                                    └──────┬───────┘
                           ┌───────────────┼───────────────┐
                           │               │               │
                    ┌──────▼───────┐ ┌─────▼────┐ ┌───────▼──────┐
                    │   Clarify    │ │Auto Refine│ │   Planner    │
                    │ (interrupt)  │ │  (loop)   │ │              │
                    └──────┬───────┘ └─────┬────┘ └───────┬──────┘
                           │               │               │
                           └───────────────┼───────────────┘
                                           │
                                    ┌──────▼───────┐
                                    │ Orchestrator │
                                    └──────┬───────┘
                           ┌───────────────┼───────────────┐
                           │               │               │
                    ┌──────▼───────┐┌──────▼───────┐┌──────▼───────┐
                    │  Subagent 1  ││  Subagent 2  ││  Subagent N  │
                    │  (parallel)  ││  (parallel)  ││  (parallel)  │
                    └──────┬───────┘└──────┬───────┘└──────┬───────┘
                           └───────────────┼───────────────┘
                                           │
                                    ┌──────▼───────┐
                                    │  Synthesizer │
                                    └──────┬───────┘
                                           │
                                    ┌──────▼───────┐         ┌────────────┐
                                    │ Gap Detector │◄────────│ Backtrack  │
                                    └──────┬───────┘         └────────────┘
                           ┌───────────────┼
                           │ (if gaps)     │ (if sufficient)
                           ▼               ▼
                    ┌──────────────┐ ┌─────────────┐
                    │ Orchestrator │ │   Reduce    │
                    │   (loop)     │ └──────┬──────┘
                    └──────────────┘        │
                                    ┌───────▼──────┐
                                    │ Credibility  │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │    Ranker    │
                                    └───────┬──────┘
                                            │
                              ══════════════╧══════════════
                                     TRUST ENGINE
                              ══════════════╤══════════════
                                            │
                                    ┌───────▼──────┐
                                    │    Claims    │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │     Cite     │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │ Span Verify  │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │Cross Validate│
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │  Confidence  │
                                    │   Scorer     │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │    Writer    │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │     END      │
                                    └──────────────┘
```

---

## Part 5: Implementation Priority

### Sprint 1: Core Improvements (Week 1-2)
1. ✅ Iterative research loop (gap detection + refinement)
2. ✅ Backtracking on dead ends
3. ✅ Enhanced credibility scoring

### Sprint 2: Multi-Agent Architecture (Week 3-4)
1. ✅ Orchestrator-worker pattern
2. ✅ Parallel subagents with own context
3. ✅ Compressed findings handoff

### Sprint 3: Trust Engine (Week 5-6)
1. ✅ Span-level citation verification
2. ✅ Cross-validation for key claims
3. ✅ Per-claim confidence scores

### Sprint 4: Polish & Performance (Week 7-8)
1. ✅ Enhanced writer with confidence indicators
2. ✅ Async mode (optional)
3. ✅ Performance optimization

---

## Part 6: Expected Improvements

| Metric | v7 Current | v8 Target | Industry Best |
|--------|-----------|-----------|---------------|
| Research depth | 1 round | 3 iterations | 5+ (OpenAI) |
| Source coverage | 8 sources | 20+ sources | 21+ (OpenAI) |
| Hallucination rate | Unknown | < 10% | ~17% (best RAG) |
| Confidence accuracy | N/A | Per-claim 0-1 | Standard |
| Dead-end recovery | None | Automatic | Standard |
| Multi-source claims | Rare | 50%+ | Variable |

---

## Sources

- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)
- [How Deep Research Works](https://blog.promptlayer.com/how-deep-research-works/)
- [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system)
- [xAI Grok 3](https://x.ai/news/grok-3)
- [Google Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)
- [Perplexity Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)
- [Hallucination Prevention Guide](https://infomineo.com/artificial-intelligence/stop-ai-hallucinations-detection-prevention-verification-guide-2025/)
