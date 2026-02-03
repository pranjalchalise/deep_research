# Research Studio V9: Final Architecture

## Overview

A LangGraph-based deep research agent with:
- **LLM-driven query understanding** (no hardcoded patterns)
- **Human-in-the-loop clarification** when queries are ambiguous
- **Multi-agent parallel research** (orchestrator-workers pattern)
- **Knowledge gap detection with iterative refinement**
- **Grounded citations with confidence indicators**

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH STUDIO V9                                │
│                         LangGraph State Machine                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  USER QUERY  │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     1. QUERY ANALYZER          │
                    │     ─────────────────          │
                    │  • LLM-based semantic analysis │
                    │  • Detects: intent, subject,   │
                    │    topic focus, temporal scope │
                    │  • Assesses ambiguity level    │
                    │  • NO regex patterns           │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │   2. AMBIGUITY ROUTER          │
                    │   ───────────────────          │
                    │  Check: needs_clarification?   │
                    └────────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
              [ambiguous]                     [clear]
                     │                               │
                     ▼                               │
        ┌────────────────────────┐                  │
        │  3. CLARIFY (HITL)     │                  │
        │  ────────────────      │                  │
        │  ═══════════════════   │                  │
        │  INTERRUPT: Ask user   │                  │
        │  for clarification     │                  │
        │  ═══════════════════   │                  │
        │                        │                  │
        │  User responds with    │                  │
        │  additional context    │                  │
        │                        │                  │
        │  Update state with:    │                  │
        │  • clarified_query     │                  │
        │  • enriched_context    │                  │
        └───────────┬────────────┘                  │
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │     4. RESEARCH PLANNER        │
                    │     ──────────────────         │
                    │  • Generates research plan     │
                    │  • Creates outline sections    │
                    │  • Formulates research Qs      │
                    │  • Has FULL context from       │
                    │    clarification (if any)      │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                    RESEARCH LOOP (max 3 iterations)                       │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
│                                                                           │
│                    ┌────────────────────────────────┐                     │
│                    │     5. ORCHESTRATOR            │                     │
│                    │     ──────────────             │                     │
│                    │  • Assigns questions to        │                     │
│                    │    parallel subagents          │                     │
│                    │  • Deduplicates queries        │                     │
│                    │  • Tracks research state       │                     │
│                    └────────────────┬───────────────┘                     │
│                                     │                                     │
│                     ┌───────────────┼───────────────┐                     │
│                     │               │               │                     │
│                     ▼               ▼               ▼                     │
│              ┌───────────┐  ┌───────────┐  ┌───────────┐                 │
│              │ SUBAGENT  │  │ SUBAGENT  │  │ SUBAGENT  │   PARALLEL      │
│              │    #1     │  │    #2     │  │    #3     │   (LangGraph    │
│              │           │  │           │  │           │    Send)        │
│              │ Question: │  │ Question: │  │ Question: │                 │
│              │ "..."     │  │ "..."     │  │ "..."     │                 │
│              │           │  │           │  │           │                 │
│              │ • Search  │  │ • Search  │  │ • Search  │                 │
│              │ • Fetch   │  │ • Fetch   │  │ • Fetch   │                 │
│              │ • Extract │  │ • Extract │  │ • Extract │                 │
│              └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                 │
│                    │              │              │                        │
│                    └──────────────┼──────────────┘                        │
│                                   │                                       │
│                                   ▼                                       │
│                    ┌────────────────────────────────┐                     │
│                    │     6. SYNTHESIZER             │                     │
│                    │     ─────────────              │                     │
│                    │  • Aggregates all findings     │                     │
│                    │  • Deduplicates sources        │                     │
│                    │  • Calculates coverage         │                     │
│                    └────────────────┬───────────────┘                     │
│                                     │                                     │
│                                     ▼                                     │
│                  ┌──────────────────────────────────────┐                 │
│                  │     7. GAP DETECTOR                  │                 │
│                  │     ──────────────                   │                 │
│                  │  ┌────────────────────────────────┐  │                 │
│                  │  │ Analyzes coverage vs plan:    │  │                 │
│                  │  │                               │  │                 │
│                  │  │ • Section-level confidence    │  │                 │
│                  │  │ • Missing information         │  │                 │
│                  │  │ • Conflicting sources         │  │                 │
│                  │  │ • Unanswered questions        │  │                 │
│                  │  │                               │  │                 │
│                  │  │ Output:                       │  │                 │
│                  │  │ • gaps: [{section, severity}] │  │                 │
│                  │  │ • refinement_queries: [...]   │  │                 │
│                  │  │ • recommendation: continue    │  │                 │
│                  │  │   or sufficient               │  │                 │
│                  │  └────────────────────────────────┘  │                 │
│                  └──────────────────┬───────────────────┘                 │
│                                     │                                     │
│                                     ▼                                     │
│                    ┌────────────────────────────────┐                     │
│                    │     8. ITERATION ROUTER        │                     │
│                    │     ──────────────────         │                     │
│                    │                                │                     │
│                    │  IF gaps AND iteration < max   │                     │
│                    │     AND confidence_delta > 5%: │                     │
│                    │     → Route to ORCHESTRATOR    │                     │
│                    │       (with refinement queries)│                     │
│                    │                                │                     │
│                    │  ELSE:                         │                     │
│                    │     → Route to TRUST ENGINE    │                     │
│                    └────────────────┬───────────────┘                     │
│                                     │                                     │
│           ┌─────────────────────────┼─────────────────────────┐           │
│           │                         │                         │           │
│    [has gaps, iterate]              │              [sufficient coverage]  │
│           │                         │                         │           │
│           │    ┌────────────────────┘                         │           │
│           │    │                                              │           │
│           ▼    │                                              │           │
│   ┌────────────┴───┐                                          │           │
│   │  INCREMENT     │                                          │           │
│   │  iteration++   │                                          │           │
│   │  Update state  │                                          │           │
│   │  with refined  │                                          │           │
│   │  queries       │                                          │           │
│   └───────┬────────┘                                          │           │
│           │                                                   │           │
│           └──────────► ORCHESTRATOR (loop back) ◄─────────────┘           │
│                                                                           │
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                     │
                                     │ [sufficient coverage]
                                     ▼
                    ┌────────────────────────────────┐
                    │     9. TRUST ENGINE            │
                    │     ─────────────              │
                    │  (Batched: 2 LLM calls)        │
                    │                                │
                    │  A. Credibility + Claims:      │
                    │     • Score source credibility │
                    │     • Extract atomic claims    │
                    │                                │
                    │  B. Verification:              │
                    │     • Span verification        │
                    │     • Cross-validation         │
                    │     • Confidence scoring       │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     10. REPORT WRITER          │
                    │     ────────────────           │
                    │  • Uses ONLY verified claims   │
                    │  • Adds confidence indicators  │
                    │     ✓✓ = cross-validated      │
                    │     ✓  = verified             │
                    │     ⚠  = single source        │
                    │  • Inline citations [S1][S2]  │
                    │  • Acknowledges gaps          │
                    │  • Research quality metadata  │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │ FINAL REPORT │
                              └──────────────┘
```

---

## State Schema

```python
from typing import TypedDict, List, Dict, Optional, Annotated
import operator

class ResearchState(TypedDict):
    # ─────────────────────────────────────────────────────────
    # INPUT
    # ─────────────────────────────────────────────────────────
    messages: List[BaseMessage]           # Conversation history

    # ─────────────────────────────────────────────────────────
    # QUERY ANALYSIS (Node 1)
    # ─────────────────────────────────────────────────────────
    original_query: str                   # Raw user query
    query_analysis: QueryAnalysis         # LLM analysis result
    # QueryAnalysis = {
    #   intent: str,
    #   query_class: str,
    #   primary_subject: str,
    #   topic_focus: Optional[str],
    #   temporal_scope: str,
    #   complexity: str,
    #   ambiguity_level: str,
    #   needs_clarification: bool,
    #   clarification_question: Optional[str],
    #   suggested_questions: List[str]
    # }

    # ─────────────────────────────────────────────────────────
    # CLARIFICATION (Node 3) - HITL
    # ─────────────────────────────────────────────────────────
    clarification_request: Optional[str]  # Question to ask user
    user_clarification: Optional[str]     # User's response
    enriched_context: Optional[str]       # Combined context after clarification

    # ─────────────────────────────────────────────────────────
    # PLANNING (Node 4)
    # ─────────────────────────────────────────────────────────
    research_plan: ResearchPlan           # Generated plan
    # ResearchPlan = {
    #   topic: str,
    #   outline: List[str],
    #   research_questions: List[ResearchQuestion],
    #   estimated_complexity: str
    # }

    # ─────────────────────────────────────────────────────────
    # RESEARCH LOOP (Nodes 5-8)
    # ─────────────────────────────────────────────────────────
    research_iteration: int               # Current iteration (0, 1, 2)
    max_iterations: int                   # Hard limit (default: 3)

    # Subagent management
    subagent_assignments: List[SubagentAssignment]
    subagent_findings: Annotated[List[SubagentFindings], operator.add]

    # Accumulated data (grows each iteration)
    raw_sources: Annotated[List[RawSource], operator.add]
    raw_evidence: Annotated[List[RawEvidence], operator.add]

    # Gap detection
    knowledge_gaps: List[KnowledgeGap]    # Identified gaps
    refinement_queries: List[str]         # Queries for next iteration
    section_confidence: Dict[str, float]  # Confidence per outline section
    overall_confidence: float             # 0.0 - 1.0
    previous_confidence: float            # For delta calculation

    # ─────────────────────────────────────────────────────────
    # PROCESSED DATA (After research loop)
    # ─────────────────────────────────────────────────────────
    sources: List[Source]                 # Deduplicated, with SIDs
    evidence: List[Evidence]              # Deduplicated, with EIDs

    # ─────────────────────────────────────────────────────────
    # TRUST ENGINE (Node 9)
    # ─────────────────────────────────────────────────────────
    source_credibility: Dict[str, SourceCredibility]
    claims: List[Claim]                   # Atomic claims extracted
    verified_citations: List[VerifiedCitation]
    unverified_claims: List[str]
    cross_validated_claims: List[str]
    claim_confidence: Dict[str, float]    # Confidence per claim
    hallucination_score: float            # % unsupported claims

    # ─────────────────────────────────────────────────────────
    # OUTPUT (Node 10)
    # ─────────────────────────────────────────────────────────
    report: str                           # Final markdown report
    research_metadata: ResearchMetadata   # Quality metrics
```

---

## LangGraph Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_research_graph():
    # Create the graph
    graph = StateGraph(ResearchState)

    # ─────────────────────────────────────────────────────────
    # ADD NODES
    # ─────────────────────────────────────────────────────────

    # Phase 1: Query Understanding
    graph.add_node("query_analyzer", query_analyzer_node)

    # Phase 2: Clarification (HITL)
    graph.add_node("clarify", clarify_node)

    # Phase 3: Planning
    graph.add_node("planner", planner_node)

    # Phase 4: Research Loop
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("subagent", subagent_node)        # Parallel execution
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("gap_detector", gap_detector_node)

    # Phase 5: Trust & Verification
    graph.add_node("trust_engine", trust_engine_node)

    # Phase 6: Report Generation
    graph.add_node("writer", writer_node)

    # ─────────────────────────────────────────────────────────
    # ADD EDGES
    # ─────────────────────────────────────────────────────────

    # Entry point
    graph.set_entry_point("query_analyzer")

    # Query Analyzer → Conditional: Clarify or Plan
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {
            "clarify": "clarify",
            "planner": "planner",
        }
    )

    # Clarify → Planner (always, after user responds)
    graph.add_edge("clarify", "planner")

    # Planner → Orchestrator
    graph.add_edge("planner", "orchestrator")

    # Orchestrator → Parallel Subagents (fan-out)
    graph.add_conditional_edges(
        "orchestrator",
        fanout_subagents,  # Returns list of Send() objects
    )

    # Subagents → Synthesizer (fan-in)
    graph.add_edge("subagent", "synthesizer")

    # Synthesizer → Gap Detector
    graph.add_edge("synthesizer", "gap_detector")

    # Gap Detector → Conditional: Loop or Continue
    graph.add_conditional_edges(
        "gap_detector",
        route_after_gap_detection,
        {
            "orchestrator": "orchestrator",  # Loop back
            "trust_engine": "trust_engine",  # Continue
        }
    )

    # Trust Engine → Writer
    graph.add_edge("trust_engine", "writer")

    # Writer → END
    graph.add_edge("writer", END)

    # ─────────────────────────────────────────────────────────
    # COMPILE WITH INTERRUPT
    # ─────────────────────────────────────────────────────────

    checkpointer = MemorySaver()

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["clarify"],  # HITL: Pause before clarify node
    )
```

---

## Node Responsibilities

### Node 1: Query Analyzer
**Purpose**: Understand what the user actually wants (LLM-driven, no patterns)

```python
def query_analyzer_node(state: ResearchState) -> dict:
    """
    Analyze user query using LLM to understand intent.

    Input: Raw user query from messages
    Output: {
        original_query: str,
        query_analysis: QueryAnalysis,
        clarification_request: Optional[str]  # If needs clarification
    }
    """
```

### Node 2: Ambiguity Router (Conditional Edge)
**Purpose**: Decide if we need human clarification

```python
def route_after_analysis(state: ResearchState) -> str:
    """Route based on ambiguity level."""
    analysis = state["query_analysis"]

    if analysis["needs_clarification"]:
        return "clarify"
    return "planner"
```

### Node 3: Clarify (HITL Interrupt)
**Purpose**: Get clarification from user, enrich context

```python
def clarify_node(state: ResearchState) -> dict:
    """
    Process user's clarification and enrich context.

    This node runs AFTER the interrupt.
    The user's response is in state["user_clarification"].

    Input: user_clarification from human
    Output: {
        enriched_context: str,  # Combined original + clarification
        query_analysis: QueryAnalysis  # Updated with clarification
    }
    """
```

### Node 4: Research Planner
**Purpose**: Generate research plan with questions and outline

```python
def planner_node(state: ResearchState) -> dict:
    """
    Generate research plan based on query analysis.

    Uses enriched_context if clarification was provided.

    Input: query_analysis (+ enriched_context if available)
    Output: {
        research_plan: ResearchPlan
    }
    """
```

### Node 5: Orchestrator
**Purpose**: Assign questions to parallel subagents

```python
def orchestrator_node(state: ResearchState) -> dict:
    """
    Distribute research questions to subagents.

    On iteration 0: Uses research_plan.research_questions
    On iteration 1+: Uses refinement_queries from gap detection

    Input: research_plan OR refinement_queries
    Output: {
        subagent_assignments: List[SubagentAssignment],
        research_iteration: int  # Incremented
    }
    """
```

### Node 6: Subagent (Parallel)
**Purpose**: Research assigned question independently

```python
def subagent_node(state: ResearchState) -> dict:
    """
    Research a single question independently.

    Runs in parallel via LangGraph Send().

    Input: subagent_assignment (single assignment)
    Output: {
        subagent_findings: [SubagentFindings],  # Accumulated
        raw_sources: [RawSource],               # Accumulated
        raw_evidence: [RawEvidence]             # Accumulated
    }
    """
```

### Node 7: Synthesizer
**Purpose**: Aggregate findings from all subagents

```python
def synthesizer_node(state: ResearchState) -> dict:
    """
    Combine all subagent findings.

    Input: subagent_findings (accumulated from all subagents)
    Output: {
        sources: List[Source],    # Deduplicated, with SIDs
        evidence: List[Evidence]  # Deduplicated, with EIDs
    }
    """
```

### Node 8: Gap Detector
**Purpose**: Check coverage and decide if more research needed

```python
def gap_detector_node(state: ResearchState) -> dict:
    """
    Analyze coverage and identify knowledge gaps.

    Input: evidence, research_plan.outline
    Output: {
        knowledge_gaps: List[KnowledgeGap],
        section_confidence: Dict[str, float],
        overall_confidence: float,
        refinement_queries: List[str],  # For next iteration
        previous_confidence: float      # For delta calculation
    }
    """

def route_after_gap_detection(state: ResearchState) -> str:
    """Decide: iterate or continue to trust engine."""
    iteration = state["research_iteration"]
    max_iter = state.get("max_iterations", 3)
    confidence = state["overall_confidence"]
    prev_confidence = state.get("previous_confidence", 0)
    gaps = state.get("knowledge_gaps", [])

    # Stop conditions
    if iteration >= max_iter:
        return "trust_engine"
    if confidence >= 0.85:
        return "trust_engine"
    if confidence - prev_confidence < 0.05 and iteration > 0:
        return "trust_engine"  # Diminishing returns
    if not gaps:
        return "trust_engine"

    # Continue iterating
    return "orchestrator"
```

### Node 9: Trust Engine
**Purpose**: Score credibility, extract claims, verify citations

```python
def trust_engine_node(state: ResearchState) -> dict:
    """
    Batched trust verification (2 LLM calls instead of 5).

    Call 1: Credibility scoring + Claims extraction
    Call 2: Span verification + Cross-validation + Confidence

    Input: sources, evidence
    Output: {
        source_credibility: Dict[str, SourceCredibility],
        claims: List[Claim],
        verified_citations: List[VerifiedCitation],
        unverified_claims: List[str],
        cross_validated_claims: List[str],
        claim_confidence: Dict[str, float],
        hallucination_score: float
    }
    """
```

### Node 10: Report Writer
**Purpose**: Generate final report with citations

```python
def writer_node(state: ResearchState) -> dict:
    """
    Generate markdown report with confidence indicators.

    Rules:
    1. Only use verified claims
    2. Include ✓✓/✓/⚠ indicators
    3. Inline citations [S1][S2]
    4. Acknowledge gaps

    Input: claims, verified_citations, source_credibility
    Output: {
        report: str,
        research_metadata: ResearchMetadata
    }
    """
```

---

## Implementation Order

### Step 1: Query Analyzer (Core Fix)
Replace pattern matching with LLM-based analysis.
```
Files: src/nodes/analyzer.py (NEW)
```

### Step 2: Clarification Node
HITL interrupt for ambiguous queries.
```
Files: src/nodes/clarify.py (UPDATE)
```

### Step 3: Research Planner
LLM-generated plans (not templates).
```
Files: src/nodes/planner.py (REWRITE)
```

### Step 4: Orchestrator + Subagents
Multi-agent parallel research.
```
Files: src/nodes/orchestrator.py (UPDATE)
```

### Step 5: Gap Detector
Knowledge gap detection and iteration logic.
```
Files: src/nodes/gap_detector.py (NEW/UPDATE)
```

### Step 6: Trust Engine
Batched verification.
```
Files: src/nodes/trust_engine.py (EXISTS - minor updates)
```

### Step 7: Report Writer
Confidence indicators and citations.
```
Files: src/nodes/writer.py (EXISTS - minor updates)
```

### Step 8: Graph Assembly
Wire everything together.
```
Files: src/core/graph.py (REWRITE)
```

---

## Example Flow

**Query**: "Deep research about Trump's new immigrant related policies"

```
1. QUERY ANALYZER
   → intent: "Understand recent Trump immigration policies"
   → query_class: "current_events"
   → primary_subject: "Trump administration immigration policy"
   → ambiguity_level: "none"
   → needs_clarification: false

2. ROUTER → "planner" (no clarification needed)

3. PLANNER
   → research_questions: [
       "What immigration executive orders has Trump signed in 2025?",
       "What are the key policy changes vs previous administration?",
       "What do legal experts say about these policies?",
       "What is the humanitarian impact?"
     ]
   → outline: ["Overview", "Key Policies", "Executive Orders",
               "Legal Analysis", "Humanitarian Impact", "Expert Opinions"]

4. ORCHESTRATOR (Iteration 0)
   → Assigns 4 questions to 4 parallel subagents

5. SUBAGENTS (Parallel)
   → SA1: Searches "Trump immigration executive orders 2025"
   → SA2: Searches "Trump immigration policy changes vs Biden"
   → SA3: Searches "Trump immigration policy legal analysis"
   → SA4: Searches "Trump deportation humanitarian impact"

6. SYNTHESIZER
   → Aggregates 24 sources, 40 evidence items
   → Deduplicates to 18 unique sources

7. GAP DETECTOR
   → section_confidence: {
       "Overview": 0.9,
       "Key Policies": 0.85,
       "Executive Orders": 0.8,
       "Legal Analysis": 0.6,  ← GAP
       "Humanitarian Impact": 0.5,  ← GAP
       "Expert Opinions": 0.4  ← GAP
     }
   → overall_confidence: 0.68
   → gaps: ["Legal Analysis", "Humanitarian Impact", "Expert Opinions"]
   → refinement_queries: [
       "Trump immigration policy court challenges 2025",
       "immigration detention conditions Trump 2025",
       "immigration policy experts analysis Trump"
     ]

8. ROUTER → "orchestrator" (gaps exist, iteration < max)

9. ORCHESTRATOR (Iteration 1)
   → Assigns refinement queries to 3 subagents

10. SUBAGENTS (Parallel) - Round 2
    → Research gaps...

11. SYNTHESIZER - Round 2
    → Now: 28 sources, 55 evidence items

12. GAP DETECTOR - Round 2
    → overall_confidence: 0.82
    → confidence_delta: 0.14 (good improvement)
    → gaps: ["Expert Opinions"] (only 1 remaining)

13. ROUTER → "trust_engine" (confidence >= 0.8)

14. TRUST ENGINE
    → Extracts 15 claims
    → Verifies 13/15 (87%)
    → Cross-validates 8 claims

15. WRITER
    → Generates report with ✓✓/✓/⚠ indicators
    → Includes citations [S1], [S2], etc.
    → Notes: "Expert opinions section has limited coverage"
```

---

## Ready to Implement?

Start with **Step 1: Query Analyzer** - this is the core fix that removes all hardcoded patterns and makes the system general.

```bash
# Implementation order:
1. src/nodes/analyzer.py      # NEW - LLM-based query analysis
2. src/nodes/clarify.py       # UPDATE - HITL with enriched context
3. src/nodes/planner.py       # REWRITE - LLM-generated plans
4. src/nodes/orchestrator.py  # UPDATE - Use new analysis
5. src/nodes/gap_detector.py  # UPDATE - Better gap detection
6. src/core/graph.py          # REWRITE - New workflow
7. src/core/state.py          # UPDATE - New state fields
```
