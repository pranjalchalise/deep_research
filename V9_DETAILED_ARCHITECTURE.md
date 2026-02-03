# Research Studio V9: Complete Node-by-Node Architecture

## Table of Contents
1. [Overview](#overview)
2. [State Schema](#state-schema)
3. [Graph Structure](#graph-structure)
4. [Node-by-Node Deep Dive](#node-by-node-deep-dive)
5. [Data Flow Example](#complete-data-flow-example)
6. [Key Design Decisions](#key-design-decisions)

---

## Overview

Research Studio V9 is a LangGraph-based deep research agent that implements state-of-the-art patterns from OpenAI, Anthropic, Google, and Perplexity. The architecture consists of **12 nodes** organized into 6 phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH STUDIO V9                                │
│                      LangGraph State Machine                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: UNDERSTANDING      Phase 2: CLARIFICATION                     │
│  ┌─────────────────┐         ┌─────────────────┐                        │
│  │ query_analyzer  │────────▶│    clarify      │                        │
│  └─────────────────┘         └─────────────────┘                        │
│          │                           │                                   │
│          └───────────────┬───────────┘                                   │
│                          ▼                                               │
│  Phase 3: PLANNING       ┌─────────────────┐                            │
│                          │    planner      │                            │
│                          └─────────────────┘                            │
│                                  │                                       │
│  Phase 4: RESEARCH LOOP          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  ┌─────────────┐    ┌──────────┐    ┌────────────┐         │        │
│  │  │orchestrator │───▶│subagent  │───▶│synthesizer │         │        │
│  │  └─────────────┘    │(parallel)│    └────────────┘         │        │
│  │         ▲           └──────────┘           │               │        │
│  │         │                                  ▼               │        │
│  │  ┌──────┴──────┐              ┌────────────────┐          │        │
│  │  │ increment_  │◀─────────────│  gap_detector  │          │        │
│  │  │ iteration   │   (if gaps)  └────────────────┘          │        │
│  │  └─────────────┘                      │                   │        │
│  └───────────────────────────────────────┼───────────────────┘        │
│                                          │ (sufficient)                │
│  Phase 5: VERIFICATION                   ▼                             │
│                          ┌─────────────────┐                           │
│                          │  trust_engine   │                           │
│                          └─────────────────┘                           │
│                                  │                                      │
│  Phase 6: OUTPUT                 ▼                                      │
│                          ┌─────────────────┐                           │
│                          │     writer      │                           │
│                          └─────────────────┘                           │
│                                  │                                      │
│                                  ▼                                      │
│                              [REPORT]                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## State Schema

The entire pipeline operates on a shared state object. Here are the key fields:

```python
class AgentState(MessagesState):
    # === Query Understanding ===
    original_query: str                    # Raw user input
    query_analysis: QueryAnalysisResult    # LLM analysis output

    # === Clarification ===
    needs_clarification: bool              # Should we ask user?
    clarification_request: str             # Question to ask
    user_clarification: str                # User's response
    enriched_context: str                  # Combined context after clarification

    # === Planning ===
    plan: Plan                             # Topic, outline, queries
    primary_anchor: str                    # Main subject for all queries

    # === Research Loop ===
    research_iteration: int                # Current iteration (0, 1, 2)
    max_iterations: int                    # Hard limit (default: 3)
    subagent_assignments: List[Assignment] # Work distribution
    subagent_findings: List[Findings]      # Results from workers

    # === Accumulated Data ===
    raw_sources: List[RawSource]           # Accumulated from workers
    raw_evidence: List[RawEvidence]        # Accumulated from workers
    sources: List[Source]                  # Normalized (with SIDs)
    evidence: List[Evidence]               # Normalized (with EIDs)

    # === Trust & Verification ===
    source_credibility: Dict[str, Score]   # Per-source credibility
    claims: List[Claim]                    # Extracted claims
    verified_citations: List[Citation]     # Verified claim-evidence links
    overall_confidence: float              # Research confidence (0-1)

    # === Gap Detection ===
    knowledge_gaps: List[KnowledgeGap]     # Missing information
    refinement_queries: List[Query]        # Queries for next iteration
    proceed_to_synthesis: bool             # Ready for final output?

    # === Output ===
    report: str                            # Final markdown report
```

---

## Graph Structure

### Multi-Agent Graph (12 nodes)

```python
def build_v9_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("planner", planner_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("subagent", subagent_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reducer", reducer_node)
    graph.add_node("gap_detector", gap_detector_node)
    graph.add_node("increment_iteration", increment_iteration_node)
    graph.add_node("trust_engine", full_trust_engine_batched)
    graph.add_node("writer", writer_node)

    # Entry point
    graph.set_entry_point("query_analyzer")

    # Conditional: clarify or plan
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {"clarify": "clarify", "planner": "planner"}
    )

    # Linear flow
    graph.add_edge("clarify", "planner")
    graph.add_edge("planner", "orchestrator")

    # Fan-out to parallel subagents
    graph.add_conditional_edges("orchestrator", fanout_subagents)

    # Fan-in
    graph.add_edge("subagent", "synthesizer")
    graph.add_edge("synthesizer", "reducer")
    graph.add_edge("reducer", "gap_detector")

    # Loop or continue
    graph.add_conditional_edges(
        "gap_detector",
        route_after_gap_detection,
        {"orchestrator": "increment_iteration", "trust_engine": "trust_engine"}
    )

    graph.add_edge("increment_iteration", "orchestrator")
    graph.add_edge("trust_engine", "writer")
    graph.add_edge("writer", END)

    return graph.compile(interrupt_before=["clarify"])
```

---

## Node-by-Node Deep Dive

### Running Example

We'll trace this query through every node:
> **"Deep research about Trump's new immigrant related policies"**

---

## NODE 1: `query_analyzer`

### Purpose
Semantically understand the user's query using LLM (no regex patterns).

### Location
`src/nodes/analyzer.py`

### What It Does

1. **Extracts** the query from the messages
2. **Calls LLM** with a detailed analysis prompt
3. **Classifies** into one of 17 query classes
4. **Detects ambiguity** and decides if clarification is needed
5. **Generates** research questions and suggested outline

### Input State
```python
{
    "messages": [HumanMessage(content="Deep research about Trump's new immigrant related policies")]
}
```

### LLM Prompt (Simplified)
```
You are an expert research query analyst.

Analyze this query: "Deep research about Trump's new immigrant related policies"

## Critical Rules
- NEVER classify queries about recent policies as "concept_explanation"
- "Deep research about X" means COMPREHENSIVE investigation
- Temporal indicators like "new" → current_events

Return JSON with:
- intent: What the user wants to learn
- query_class: Classification (17 options)
- primary_subject: Main entity
- temporal_scope: recent/historical/timeless
- ambiguity_level: none/low/medium/high
- needs_clarification: true/false
- suggested_questions: Research questions
- suggested_outline: Report sections
```

### LLM Response
```json
{
  "intent": "Understand the recent immigration policy changes under the Trump administration",
  "query_class": "current_events",
  "primary_subject": "Trump administration immigration policies",
  "subject_type": "policy",
  "topic_focus": "recent immigration-related policies",
  "temporal_scope": "recent",
  "domain": "politics",
  "complexity": "complex",
  "ambiguity_level": "none",
  "needs_clarification": false,
  "suggested_questions": [
    "What immigration executive orders has Trump signed?",
    "What are the key policy changes from the previous administration?",
    "What legal challenges have been filed?",
    "What is the humanitarian impact?"
  ],
  "suggested_outline": [
    "Executive Summary",
    "Key Policy Changes",
    "Executive Orders",
    "Legal Challenges",
    "Humanitarian Impact"
  ],
  "analysis_confidence": 0.95
}
```

### Output State Updates
```python
{
    "original_query": "Deep research about Trump's new immigrant related policies",
    "query_analysis": { ... },  # Full analysis object
    "needs_clarification": False,
    "clarification_request": None,
    "primary_anchor": "Trump administration immigration policies"
}
```

### Routing Decision
```python
def route_after_analysis(state):
    if state.get("needs_clarification"):
        return "clarify"  # → go to clarify node
    return "planner"      # → skip to planner
```

**Result**: `"planner"` (no clarification needed)

---

## NODE 2: `clarify` (HITL)

### Purpose
Handle human-in-the-loop clarification for ambiguous queries.

### Location
`src/nodes/clarify.py`

### When It Runs
Only when `route_after_analysis` returns `"clarify"`.

### Example: Ambiguous Query

**Query**: `"John Smith professor"`

**Query Analyzer Output**:
```json
{
  "query_class": "person_profile",
  "ambiguity_level": "high",
  "needs_clarification": true,
  "clarification_question": "There are many professors named John Smith. Which university or field?",
  "clarification_options": [
    "Specify the university",
    "Specify the field",
    "Provide other context"
  ],
  "analysis_confidence": 0.30
}
```

### LangGraph Interrupt

The graph **interrupts** before this node:
```python
graph.compile(interrupt_before=["clarify"])
```

The application shows the user:
```
Clarification needed:

There are many professors named John Smith. Could you provide more context?

Suggestions:
1. Specify the university (e.g., "John Smith at MIT")
2. Specify the field (e.g., "John Smith economics professor")
3. Provide other context
```

### User Response
User types: `"MIT, computer science"`

The application resumes the graph with:
```python
graph.invoke(
    {"user_clarification": "MIT, computer science"},
    config={"configurable": {"thread_id": "..."}}
)
```

### What clarify_node Does

1. **Takes** user's clarification
2. **Calls LLM** to merge original query + clarification
3. **Creates enriched_context** for downstream nodes
4. **Updates** query_analysis with clarified information

### LLM Prompt (Clarification Processing)
```
Original Query: "John Smith professor"
Original Intent: Learn about a professor named John Smith
Ambiguity Reasons: Common name, no institution specified

User's Clarification: "MIT, computer science"

Create enriched understanding combining original query with clarification.
```

### LLM Response
```json
{
  "enriched_context": "Research John Smith, a computer science professor at MIT. Find information about his background, research work, publications, and academic contributions.",
  "updated_intent": "Learn about John Smith, the computer science professor at MIT",
  "updated_subject": "John Smith (MIT Computer Science)",
  "updated_class": "person_profile",
  "suggested_questions": [
    "What is John Smith's educational background?",
    "What are his main research areas?",
    "What notable publications has he authored?",
    "What courses does he teach at MIT?"
  ],
  "confidence_boost": 0.35
}
```

### Output State Updates
```python
{
    "enriched_context": "Research John Smith, a computer science professor at MIT...",
    "user_clarification": "MIT, computer science",
    "needs_clarification": False,
    "query_analysis": {
        # Updated with clarified info
        "intent": "Learn about John Smith, the computer science professor at MIT",
        "primary_subject": "John Smith (MIT Computer Science)",
        "analysis_confidence": 0.65  # Was 0.30, boosted by 0.35
    },
    "primary_anchor": "John Smith (MIT Computer Science)"
}
```

### Key Insight
After clarification, **all downstream nodes** have access to:
- `original_query`: The raw query
- `user_clarification`: What the user said
- `enriched_context`: Complete, disambiguated context
- `query_analysis`: Updated with clarified information

---

## NODE 3: `planner`

### Purpose
Generate a comprehensive research plan based on query analysis.

### Location
`src/nodes/planner_v9.py`

### What It Does

1. **Reads** query_analysis and enriched_context
2. **Calls LLM** to generate research plan
3. **Creates**:
   - Topic title
   - Report outline
   - Research tree (primary/secondary questions)
   - Optimized search queries

### Input (From State)
```python
{
    "query_analysis": {
        "intent": "Understand recent Trump immigration policies",
        "query_class": "current_events",
        "primary_subject": "Trump administration immigration policies",
        "temporal_scope": "recent",
        "domain": "politics",
        "suggested_questions": [...],
        "suggested_outline": [...]
    },
    "enriched_context": None  # (or clarification context if applicable)
}
```

### LLM Prompt
```
You are an expert research planner.

## Query Analysis
Intent: Understand recent Trump immigration policies
Query Class: current_events
Primary Subject: Trump administration immigration policies
Temporal Scope: recent
Domain: politics

## Guidelines for current_events queries:
- Include recent time qualifiers (2024, 2025, "latest")
- Search for official announcements, news, expert analysis
- Include multiple perspectives

## Search Query Optimization Rules
- 3-7 words per query
- Use quotes for proper nouns: "Donald Trump"
- Add year for recency: "2025", "latest"

Return JSON with topic, outline, research_tree, search_strategy.
```

### LLM Response
```json
{
  "topic": "Trump Administration Immigration Policies (2024-2025)",
  "outline": [
    "Executive Summary",
    "Key Policy Changes",
    "Executive Orders and Legal Framework",
    "Border Security Measures",
    "Deportation Policies",
    "Legal Challenges and Court Decisions",
    "Humanitarian Impact",
    "Expert Analysis and Perspectives",
    "Comparison with Previous Administration"
  ],
  "research_tree": {
    "primary": [
      {
        "question": "What are the key immigration policy changes under Trump in 2024-2025?",
        "queries": [
          "Trump immigration policy changes 2025",
          "Trump executive orders immigration 2024 2025",
          "Trump border policy latest news"
        ],
        "target_sources": ["news", "official"],
        "priority": 1.0
      },
      {
        "question": "What executive orders has Trump signed on immigration?",
        "queries": [
          "Trump immigration executive orders list 2025",
          "Trump border security executive order",
          "Trump deportation executive order text"
        ],
        "target_sources": ["official", "legal"],
        "priority": 0.95
      },
      {
        "question": "What legal challenges have been filed?",
        "queries": [
          "Trump immigration policy court challenges 2025",
          "Trump deportation policy lawsuit",
          "immigration policy legal analysis Trump"
        ],
        "target_sources": ["news", "legal"],
        "priority": 0.9
      }
    ],
    "secondary": [
      {
        "question": "What do immigration experts say?",
        "queries": [
          "immigration experts analysis Trump policy",
          "Trump immigration policy expert opinion"
        ],
        "target_sources": ["analysis", "academic"],
        "priority": 0.7
      },
      {
        "question": "What is the humanitarian impact?",
        "queries": [
          "Trump immigration policy humanitarian impact",
          "Trump deportation families affected"
        ],
        "target_sources": ["news", "advocacy"],
        "priority": 0.6
      }
    ]
  },
  "search_strategy": "Start with recent news and official announcements, then legal analysis and expert commentary."
}
```

### Processing
The planner converts the research_tree into a flat list of queries for the orchestrator:

```python
all_queries = [
    {"qid": "Q1", "query": "Trump immigration policy changes 2025", "section": "Key Policy Changes", "priority": 1.0},
    {"qid": "Q2", "query": "Trump executive orders immigration 2024 2025", "section": "Key Policy Changes", "priority": 1.0},
    {"qid": "Q3", "query": "Trump border policy latest news", "section": "Key Policy Changes", "priority": 1.0},
    {"qid": "Q4", "query": "Trump immigration executive orders list 2025", "section": "Executive Orders", "priority": 0.95},
    # ... more queries
]
```

### Output State Updates
```python
{
    "plan": {
        "topic": "Trump Administration Immigration Policies (2024-2025)",
        "outline": ["Executive Summary", "Key Policy Changes", ...],
        "queries": [...],  # Flat list with qid, query, section, priority
        "research_tree": {...}  # Hierarchical structure
    },
    "total_workers": 13,  # Number of queries
    "done_workers": 0,
    "research_iteration": 0,
    "max_iterations": 3,
    "overall_confidence": 0.0
}
```

---

## NODE 4: `orchestrator`

### Purpose
Coordinate parallel subagents by assigning research questions.

### Location
`src/nodes/orchestrator.py`

### What It Does

1. **Reads** the research plan (or refinement_queries if looping)
2. **Groups** queries by section/question
3. **Deduplicates** similar queries
4. **Creates** subagent assignments
5. **Limits** to max_subagents (default: 5)

### Input (First Iteration)
```python
{
    "plan": {
        "research_tree": {
            "primary": [...],
            "secondary": [...]
        }
    },
    "research_iteration": 0,
    "refinement_queries": []  # Empty on first iteration
}
```

### Processing
```python
def orchestrator_node(state):
    plan = state.get("plan")
    refinement_queries = state.get("refinement_queries") or []

    if refinement_queries:
        # Looping back - use refinement queries
        assignments = create_assignments_from_refinement(refinement_queries)
    else:
        # First iteration - use research tree
        assignments = create_assignments_from_tree(plan["research_tree"])

    # Deduplicate similar queries
    assignments = deduplicate_assignments(assignments)

    # Limit to max subagents
    assignments = assignments[:5]

    return {"subagent_assignments": assignments}
```

### Output: Subagent Assignments
```python
{
    "subagent_assignments": [
        {
            "subagent_id": "SA1",
            "question": "What are the key immigration policy changes under Trump in 2024-2025?",
            "queries": [
                "Trump immigration policy changes 2025",
                "Trump executive orders immigration 2024 2025",
                "Trump border policy latest news"
            ],
            "target_sources": ["news", "official"]
        },
        {
            "subagent_id": "SA2",
            "question": "What executive orders has Trump signed on immigration?",
            "queries": [
                "Trump immigration executive orders list 2025",
                "Trump border security executive order"
            ],
            "target_sources": ["official", "legal"]
        },
        {
            "subagent_id": "SA3",
            "question": "What legal challenges have been filed?",
            "queries": [
                "Trump immigration policy court challenges 2025",
                "Trump deportation policy lawsuit"
            ],
            "target_sources": ["news", "legal"]
        },
        {
            "subagent_id": "SA4",
            "question": "What do immigration experts say?",
            "queries": [
                "immigration experts analysis Trump policy"
            ],
            "target_sources": ["analysis"]
        }
    ],
    "orchestrator_state": {
        "phase": "primary_research",
        "questions_assigned": 4,
        "questions_completed": 0
    }
}
```

### Fan-Out (Parallel Execution)

After orchestrator, LangGraph uses `Send()` to spawn parallel subagents:

```python
def fanout_subagents(state):
    assignments = state.get("subagent_assignments") or []

    sends = []
    for assignment in assignments:
        # Each subagent gets the full state + its specific assignment
        sends.append(Send("subagent", {
            **state,
            "subagent_assignment": assignment
        }))

    return sends
```

This creates **4 parallel branches**, each running `subagent_node` independently.

---

## NODE 5: `subagent` (Parallel Workers)

### Purpose
Independent research worker that searches, fetches, and extracts evidence.

### Location
`src/nodes/orchestrator.py` (subagent_node function)

### What It Does (Per Subagent)

1. **Iterates** through assigned queries (max 2 iterations)
2. **Searches** using Tavily API
3. **Fetches** top 2 pages per query
4. **Extracts** evidence using LLM
5. **Assesses** confidence
6. **Refines** queries if confidence is low
7. **Compresses** findings into summary

### Input (SA1 Example)
```python
{
    "subagent_assignment": {
        "subagent_id": "SA1",
        "question": "What are the key immigration policy changes under Trump?",
        "queries": ["Trump immigration policy changes 2025", ...]
    },
    "use_cache": True,
    "cache_dir": ".cache_v9"
}
```

### Execution Flow

```
For each query:
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. SEARCH (Tavily API)                  │
│    Query: "Trump immigration policy     │
│            changes 2025"                │
│    Results: 6 URLs with titles/snippets │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 2. FETCH (Top 2 pages)                  │
│    URL: nytimes.com/trump-immigration   │
│    → HTML → Markdown → Chunks           │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 3. EXTRACT EVIDENCE (LLM)               │
│    "Given this content and question,    │
│     extract 2-4 factual evidence items" │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 4. ASSESS CONFIDENCE                    │
│    Based on evidence count + quality    │
│    Confidence: 0.75                     │
└─────────────────────────────────────────┘
```

### Evidence Extraction (LLM Call)

**Prompt**:
```
You extract concise, high-signal evidence from source content.

Question: What are the key immigration policy changes under Trump?

Source content:
[Content from nytimes.com/trump-immigration...]

Return JSON array of evidence items:
{"text": "1-3 sentences of factual evidence"}
```

**Response**:
```json
[
  {"text": "President Trump signed Executive Order 14159 on January 20, 2025, effectively ending asylum processing at the southern border and reinstating the 'Remain in Mexico' policy."},
  {"text": "The new policy requires all asylum seekers to apply from their home countries through U.S. embassies, eliminating the ability to claim asylum at ports of entry."},
  {"text": "Immigration experts estimate the policy affects approximately 2.5 million pending asylum cases."}
]
```

### Output (Per Subagent)
```python
{
    "subagent_findings": [{
        "subagent_id": "SA1",
        "question": "What are the key immigration policy changes?",
        "findings": "Trump signed EO 14159 ending asylum at border, reinstating 'Remain in Mexico'. Asylum seekers must apply from home countries. ~2.5M pending cases affected.",
        "evidence_ids": [],  # Assigned by reducer
        "confidence": 0.78,
        "iterations_used": 1,
        "dead_ends": []
    }],
    "raw_sources": [
        {"url": "https://nytimes.com/...", "title": "Trump Immigration Policy...", "snippet": "..."},
        {"url": "https://washingtonpost.com/...", "title": "Border Changes...", "snippet": "..."},
        # ... more sources
    ],
    "raw_evidence": [
        {"url": "https://nytimes.com/...", "title": "...", "text": "President Trump signed Executive Order 14159...", "section": "General"},
        {"url": "https://nytimes.com/...", "title": "...", "text": "The new policy requires all asylum seekers...", "section": "General"},
        # ... more evidence
    ],
    "done_workers": 1,
    "done_subagents": 1
}
```

### Parallel Aggregation

All 4 subagents run in parallel. LangGraph **accumulates** their outputs:

```python
# State after all subagents complete:
{
    "subagent_findings": [
        {"subagent_id": "SA1", "findings": "...", "confidence": 0.78},
        {"subagent_id": "SA2", "findings": "...", "confidence": 0.82},
        {"subagent_id": "SA3", "findings": "...", "confidence": 0.65},
        {"subagent_id": "SA4", "findings": "...", "confidence": 0.70}
    ],
    "raw_sources": [...],  # Combined from all subagents (~24 sources)
    "raw_evidence": [...], # Combined from all subagents (~20 items)
    "done_workers": 4,
    "done_subagents": 4
}
```

---

## NODE 6: `synthesizer`

### Purpose
Aggregate findings from all subagents and calculate overall confidence.

### Location
`src/nodes/orchestrator.py` (synthesizer_node function)

### What It Does

1. **Waits** for all workers to complete
2. **Aggregates** confidence scores
3. **Collects** dead ends from all subagents
4. **Updates** orchestrator state
5. **Adds** trajectory step for debugging

### Input
```python
{
    "total_workers": 4,
    "done_workers": 4,  # All complete
    "subagent_findings": [...],
    "orchestrator_state": {...}
}
```

### Processing
```python
def synthesizer_node(state):
    # Check if all workers done
    if state["done_workers"] < state["total_workers"]:
        return {}  # Not ready yet

    # Calculate average confidence
    findings = state["subagent_findings"]
    avg_confidence = sum(f["confidence"] for f in findings) / len(findings)

    # Collect dead ends
    all_dead_ends = []
    for f in findings:
        all_dead_ends.extend(f.get("dead_ends", []))

    return {
        "overall_confidence": avg_confidence,
        "dead_ends": all_dead_ends,
        "orchestrator_state": {
            "phase": "synthesis",
            "questions_completed": len(findings),
            "overall_confidence": avg_confidence
        }
    }
```

### Output
```python
{
    "overall_confidence": 0.74,  # Average of 0.78, 0.82, 0.65, 0.70
    "dead_ends": [...],
    "orchestrator_state": {
        "phase": "synthesis",
        "questions_completed": 4,
        "overall_confidence": 0.74
    },
    "research_trajectory": [
        {
            "iteration": 0,
            "action": "synthesize",
            "query": "Synthesized 4 subagent findings",
            "result_summary": "Avg confidence: 0.74",
            "timestamp": "2025-02-02T..."
        }
    ]
}
```

---

## NODE 7: `reducer`

### Purpose
Normalize raw sources/evidence into structured format with unique IDs.

### Location
`src/nodes/reducer.py`

### What It Does

1. **Deduplicates** sources by URL
2. **Assigns** source IDs (S1, S2, S3...)
3. **Assigns** evidence IDs (E1, E2, E3...)
4. **Links** evidence to sources via URL

### Input
```python
{
    "raw_sources": [
        {"url": "https://nytimes.com/...", "title": "...", "snippet": "..."},
        {"url": "https://nytimes.com/...", "title": "...", "snippet": "..."},  # Duplicate
        {"url": "https://washingtonpost.com/...", "title": "...", "snippet": "..."},
        # ... 24 raw sources (with duplicates)
    ],
    "raw_evidence": [
        {"url": "https://nytimes.com/...", "title": "...", "text": "...", "section": "General"},
        # ... 20 evidence items
    ]
}
```

### Processing
```python
def reducer_node(state):
    # Only run when all workers are done
    if state["done_workers"] < state["total_workers"]:
        return {}

    # Deduplicate sources by URL
    seen_urls = set()
    unique_sources = []
    for s in state["raw_sources"]:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique_sources.append(s)

    # Assign source IDs
    sources = []
    for i, s in enumerate(unique_sources):
        sources.append({
            "sid": f"S{i+1}",
            "url": s["url"],
            "title": s["title"],
            "snippet": s["snippet"]
        })

    # Build URL → SID mapping
    url_to_sid = {s["url"]: s["sid"] for s in sources}

    # Assign evidence IDs and link to sources
    evidence = []
    for i, e in enumerate(state["raw_evidence"]):
        sid = url_to_sid.get(e["url"], "S?")
        evidence.append({
            "eid": f"E{i+1}",
            "sid": sid,
            "url": e["url"],
            "title": e["title"],
            "section": e["section"],
            "text": e["text"]
        })

    return {"sources": sources, "evidence": evidence}
```

### Output
```python
{
    "sources": [
        {"sid": "S1", "url": "https://nytimes.com/...", "title": "Trump Immigration...", "snippet": "..."},
        {"sid": "S2", "url": "https://washingtonpost.com/...", "title": "Border Changes...", "snippet": "..."},
        {"sid": "S3", "url": "https://reuters.com/...", "title": "Executive Orders...", "snippet": "..."},
        # ... 12 unique sources (deduplicated from 24)
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "url": "...", "title": "...", "text": "President Trump signed Executive Order 14159...", "section": "General"},
        {"eid": "E2", "sid": "S1", "url": "...", "title": "...", "text": "The new policy requires all asylum seekers...", "section": "General"},
        {"eid": "E3", "sid": "S2", "url": "...", "title": "...", "text": "Immigration experts estimate 2.5 million...", "section": "General"},
        # ... 20 evidence items
    ]
}
```

---

## NODE 8: `gap_detector`

### Purpose
Analyze research coverage and decide whether to loop or continue.

### Location
`src/nodes/gap_detector.py`

### What It Does

1. **Groups** evidence by outline section
2. **Calls LLM** to analyze coverage
3. **Identifies** knowledge gaps
4. **Generates** refinement queries for gaps
5. **Decides**: loop back or proceed

### Input
```python
{
    "plan": {
        "topic": "Trump Administration Immigration Policies (2024-2025)",
        "outline": ["Executive Summary", "Key Policy Changes", "Executive Orders", "Legal Challenges", "Humanitarian Impact", ...]
    },
    "sources": [...],  # 12 sources
    "evidence": [...],  # 20 evidence items
    "research_iteration": 0,
    "max_iterations": 3,
    "overall_confidence": 0.74,
    "previous_confidence": 0.0
}
```

### LLM Prompt
```
You are analyzing research findings to identify knowledge gaps.

## Research Plan
Topic: Trump Administration Immigration Policies (2024-2025)
Outline: Executive Summary, Key Policy Changes, Executive Orders, Legal Challenges, Humanitarian Impact, ...

## Current Findings
12 sources, 20 evidence items

### Evidence by Section:
Key Policy Changes (8 items):
- President Trump signed Executive Order 14159...
- The new policy requires all asylum seekers...
- ...

Executive Orders (4 items):
- Trump signed EO 14159 on January 20, 2025...
- ...

Legal Challenges (3 items):
- ACLU filed lawsuit challenging...
- ...

Humanitarian Impact (2 items):
- Families report being separated...
- ...

Expert Analysis (1 item):
- ...

## Decision Criteria
CONTINUE if: confidence < 0.7 AND iteration < 2, OR any section < 0.5
STOP if: confidence >= 0.8, OR diminishing returns, OR max iterations

Analyze coverage and return JSON.
```

### LLM Response
```json
{
  "section_analysis": [
    {"section": "Key Policy Changes", "confidence": 0.85, "evidence_count": 8, "missing": "Nothing major"},
    {"section": "Executive Orders", "confidence": 0.75, "evidence_count": 4, "missing": "Full text of orders"},
    {"section": "Legal Challenges", "confidence": 0.60, "evidence_count": 3, "missing": "Court rulings, judge opinions"},
    {"section": "Humanitarian Impact", "confidence": 0.45, "evidence_count": 2, "missing": "Statistics, personal stories, NGO reports"},
    {"section": "Expert Analysis", "confidence": 0.35, "evidence_count": 1, "missing": "More expert opinions, academic analysis"}
  ],
  "overall_confidence": 0.65,
  "gaps": [
    {
      "section": "Humanitarian Impact",
      "severity": "high",
      "description": "Insufficient coverage of human impact",
      "suggested_queries": [
        "Trump immigration policy humanitarian crisis 2025",
        "families separated Trump deportation policy",
        "immigration detention conditions 2025"
      ]
    },
    {
      "section": "Expert Analysis",
      "severity": "medium",
      "description": "Need more expert perspectives",
      "suggested_queries": [
        "immigration law professors Trump policy analysis",
        "legal scholars Trump immigration executive orders"
      ]
    }
  ],
  "recommendation": "continue",
  "reasoning": "Humanitarian Impact (0.45) and Expert Analysis (0.35) are below threshold. More research needed."
}
```

### Processing
```python
def gap_detector_node(state):
    # ... LLM call ...

    # Build refinement queries
    refinement_queries = []
    for gap in gaps:
        for query in gap["suggested_queries"][:2]:
            refinement_queries.append({
                "query": query,
                "section": gap["section"],
                "priority": gap["priority"]
            })

    # Check stopping conditions
    should_stop = False
    if current_iteration >= max_iterations - 1:
        should_stop = True
    elif overall_confidence >= 0.85:
        should_stop = True
    elif confidence_delta < 0.05 and current_iteration > 0:
        should_stop = True  # Diminishing returns

    return {
        "knowledge_gaps": gaps,
        "refinement_queries": refinement_queries,
        "overall_confidence": 0.65,
        "proceed_to_synthesis": should_stop
    }
```

### Output (First Iteration)
```python
{
    "knowledge_gaps": [
        {"section": "Humanitarian Impact", "description": "Insufficient coverage", "priority": 1.0},
        {"section": "Expert Analysis", "description": "Need more perspectives", "priority": 0.7}
    ],
    "refinement_queries": [
        {"query": "Trump immigration policy humanitarian crisis 2025", "section": "Humanitarian Impact"},
        {"query": "families separated Trump deportation policy", "section": "Humanitarian Impact"},
        {"query": "immigration law professors Trump policy analysis", "section": "Expert Analysis"},
        {"query": "legal scholars Trump immigration executive orders", "section": "Expert Analysis"}
    ],
    "section_confidence": {
        "Key Policy Changes": 0.85,
        "Executive Orders": 0.75,
        "Legal Challenges": 0.60,
        "Humanitarian Impact": 0.45,
        "Expert Analysis": 0.35
    },
    "overall_confidence": 0.65,
    "previous_confidence": 0.74,
    "proceed_to_synthesis": False  # Continue researching
}
```

### Routing Decision
```python
def route_after_gap_detection(state):
    if state.get("proceed_to_synthesis"):
        return "trust_engine"  # Sufficient coverage
    if state.get("refinement_queries"):
        return "orchestrator"  # Loop back for more research
    return "trust_engine"  # Default
```

**Result**: `"orchestrator"` (loop back)

---

## NODE 9: `increment_iteration`

### Purpose
Simple node to increment the iteration counter when looping.

### Location
`src/core/graph_v9.py`

### What It Does
```python
def increment_iteration_node(state):
    current = state.get("research_iteration", 0)
    return {"research_iteration": current + 1}
```

### Output
```python
{
    "research_iteration": 1  # Was 0, now 1
}
```

This triggers another pass through orchestrator → subagent → synthesizer → reducer → gap_detector.

---

## THE LOOP (Iteration 2)

### Orchestrator (Second Pass)

Now uses `refinement_queries` instead of original plan:

```python
{
    "subagent_assignments": [
        {
            "subagent_id": "SA1",
            "question": "Find more information about: Humanitarian Impact",
            "queries": [
                "Trump immigration policy humanitarian crisis 2025",
                "families separated Trump deportation policy"
            ]
        },
        {
            "subagent_id": "SA2",
            "question": "Find more information about: Expert Analysis",
            "queries": [
                "immigration law professors Trump policy analysis"
            ]
        }
    ]
}
```

### After Second Iteration

Gap detector runs again:
```python
{
    "section_confidence": {
        "Key Policy Changes": 0.85,
        "Executive Orders": 0.75,
        "Legal Challenges": 0.60,
        "Humanitarian Impact": 0.72,  # Improved from 0.45
        "Expert Analysis": 0.68       # Improved from 0.35
    },
    "overall_confidence": 0.78,
    "proceed_to_synthesis": True  # Now sufficient
}
```

**Routing**: `"trust_engine"`

---

## NODE 10: `trust_engine`

### Purpose
Verify claims against evidence and calculate credibility scores.

### Location
`src/nodes/trust_engine_batched.py`

### What It Does (2 LLM Calls)

**Call 1: Credibility + Claims**
1. **Score** source credibility (domain trust, authority, content quality)
2. **Extract** claims from evidence
3. **Map** claims to supporting evidence

**Call 2: Verification**
1. **Verify** each claim against evidence spans
2. **Cross-validate** claims with multiple sources
3. **Calculate** confidence scores

### Input
```python
{
    "sources": [...],   # 18 sources (after second iteration)
    "evidence": [...],  # 32 evidence items
    "plan": {"outline": [...]}
}
```

### Call 1: Credibility + Claims

**Prompt**:
```
TASK 1: Assess source credibility (authority, content_quality)
TASK 2: Extract 8-15 factual claims organized by section

SOURCES:
[S1] Trump Immigration Policy Changes - NYTimes
URL: https://nytimes.com/...
Snippet: President Trump signed...

[S2] Border Policy Analysis - Reuters
...

EVIDENCE:
[E1] (from S1) President Trump signed Executive Order 14159...
[E2] (from S1) The new policy requires all asylum seekers...
...

SECTIONS: Executive Summary, Key Policy Changes, Executive Orders, ...
```

**Response**:
```json
{
  "source_assessments": [
    {"sid": "S1", "authority": 0.9, "content_quality": 0.85},
    {"sid": "S2", "authority": 0.85, "content_quality": 0.8},
    {"sid": "S3", "authority": 0.6, "content_quality": 0.7}
  ],
  "claims": [
    {
      "cid": "C1",
      "text": "Trump signed Executive Order 14159 on January 20, 2025, ending asylum processing at the southern border.",
      "section": "Executive Orders",
      "supporting_eids": ["E1", "E5", "E12"]
    },
    {
      "cid": "C2",
      "text": "The 'Remain in Mexico' policy has been reinstated for all asylum seekers.",
      "section": "Key Policy Changes",
      "supporting_eids": ["E1", "E8"]
    },
    {
      "cid": "C3",
      "text": "ACLU filed a federal lawsuit challenging the constitutionality of EO 14159.",
      "section": "Legal Challenges",
      "supporting_eids": ["E15"]
    },
    // ... 10 more claims
  ]
}
```

### Processing (Credibility)
```python
def calculate_credibility(source, llm_scores, domain_trust):
    return (
        domain_trust * 0.30 +      # nytimes.com = 0.85
        freshness * 0.15 +         # Recent = 0.8
        authority * 0.25 +         # LLM: 0.9
        content_quality * 0.30     # LLM: 0.85
    )
    # S1: 0.85*0.30 + 0.8*0.15 + 0.9*0.25 + 0.85*0.30 = 0.855
```

### Call 2: Verification

**Prompt**:
```
Verify each claim against evidence.

CLAIMS:
[
  {"cid": "C1", "text": "Trump signed Executive Order 14159...", "section": "Executive Orders"},
  ...
]

EVIDENCE:
[E1] (S1) President Trump signed Executive Order 14159 on January 20, 2025...
[E5] (S3) The executive order, numbered 14159, was signed...
...

For each claim:
1. Find exact evidence span supporting it
2. Check if multiple sources support it
3. Rate confidence (0-1)
```

**Response**:
```json
[
  {
    "cid": "C1",
    "verified": true,
    "evidence_span": "President Trump signed Executive Order 14159 on January 20, 2025",
    "source_eid": "E1",
    "match_confidence": 0.95,
    "cross_validated": true,
    "supporting_sids": ["S1", "S3", "S5"],
    "final_confidence": 0.92
  },
  {
    "cid": "C2",
    "verified": true,
    "evidence_span": "reinstating the 'Remain in Mexico' policy",
    "source_eid": "E1",
    "match_confidence": 0.88,
    "cross_validated": true,
    "supporting_sids": ["S1", "S2"],
    "final_confidence": 0.85
  },
  {
    "cid": "C3",
    "verified": true,
    "evidence_span": "ACLU filed a federal lawsuit",
    "source_eid": "E15",
    "match_confidence": 0.90,
    "cross_validated": false,
    "supporting_sids": ["S8"],
    "final_confidence": 0.72
  }
]
```

### Output
```python
{
    "source_credibility": {
        "S1": {"sid": "S1", "domain_trust": 0.85, "authority": 0.9, "overall": 0.855},
        "S2": {"sid": "S2", "domain_trust": 0.80, "authority": 0.85, "overall": 0.82},
        # ...
    },
    "claims": [
        {"cid": "C1", "text": "Trump signed EO 14159...", "section": "Executive Orders"},
        {"cid": "C2", "text": "Remain in Mexico reinstated...", "section": "Key Policy Changes"},
        # ... 13 claims
    ],
    "citations": [
        {"cid": "C1", "eids": ["E1", "E5", "E12"]},
        {"cid": "C2", "eids": ["E1", "E8"]},
        # ...
    ],
    "verified_citations": [
        {"cid": "C1", "eid": "E1", "evidence_span": "...", "match_score": 0.95, "verified": True, "cross_validated": True},
        # ...
    ],
    "claim_confidence": {
        "C1": 0.92,
        "C2": 0.85,
        "C3": 0.72,
        # ...
    },
    "cross_validated_claims": [...],  # Claims with 2+ sources
    "single_source_claims": [...],    # Claims with 1 source
    "overall_confidence": 0.82,
    "hallucination_score": 0.08       # 8% of claims unverified
}
```

---

## NODE 11: `writer`

### Purpose
Generate the final research report in markdown format.

### Location
`src/nodes/writer.py`

### What It Does

1. **Checks** for blocking issues
2. **Builds** claim packets with allowed citations
3. **Calls LLM** to write report
4. **Ensures** every paragraph has citations

### Input
```python
{
    "plan": {
        "topic": "Trump Administration Immigration Policies (2024-2025)",
        "outline": ["Executive Summary", "Key Policy Changes", ...]
    },
    "sources": [...],
    "evidence": [...],
    "claims": [
        {"cid": "C1", "text": "Trump signed EO 14159...", "section": "Executive Orders"},
        ...
    ],
    "citations": [
        {"cid": "C1", "eids": ["E1", "E5"]},
        ...
    ]
}
```

### Processing
```python
def writer_node(state):
    # Build claim packets with allowed citations
    claim_packets = []
    for claim in claims:
        sids = []
        for eid in citations[claim["cid"]]:
            sid = evidence[eid]["sid"]
            sids.append(sid)

        claim_packets.append({
            "cid": claim["cid"],
            "section": claim["section"],
            "text": claim["text"],
            "cite": [f"[{sid}]" for sid in sids]  # [S1], [S3], [S5]
        })
```

### LLM Prompt
```
You write a research report ONLY from the provided claims and citations.

Rules:
- Use ONLY the claims provided
- Every paragraph must include at least one citation like [S1]
- Citations must correspond to the sources provided

TITLE: Trump Administration Immigration Policies (2024-2025)

OUTLINE:
['Executive Summary', 'Key Policy Changes', 'Executive Orders', 'Legal Challenges', ...]

CLAIMS (each includes allowed citations):
[
  {"cid": "C1", "section": "Executive Orders", "text": "Trump signed EO 14159...", "cite": ["[S1]", "[S3]"]},
  {"cid": "C2", "section": "Key Policy Changes", "text": "Remain in Mexico reinstated...", "cite": ["[S1]", "[S2]"]},
  ...
]

SOURCES:
[S1] Trump Immigration Policy Changes — https://nytimes.com/...
[S2] Border Policy Analysis — https://reuters.com/...
...
```

### Output
```python
{
    "report": """# Trump Administration Immigration Policies (2024-2025)

## Executive Summary

The Trump administration has implemented sweeping changes to U.S. immigration policy since taking office in January 2025. Key changes include the reinstatement of the "Remain in Mexico" policy [S1][S2], the effective end of asylum processing at the southern border [S1][S3], and new deportation priorities targeting undocumented immigrants [S4].

## Key Policy Changes

President Trump signed Executive Order 14159 on January 20, 2025, which fundamentally restructured the asylum system [S1][S3][S5]. The order requires all asylum seekers to apply from their home countries through U.S. embassies, eliminating the ability to claim asylum at ports of entry [S1][S2].

The "Remain in Mexico" policy has been reinstated for all asylum seekers [S1][S2]. Under this policy, migrants seeking asylum must wait in Mexico while their cases are processed...

## Executive Orders

EO 14159 specifically targets:
1. Asylum processing at the border [S1]
2. Visa overstays [S6]
3. Sanctuary city funding [S7]

## Legal Challenges

The ACLU filed a federal lawsuit challenging the constitutionality of EO 14159 [S8]. The lawsuit argues that the order violates the Immigration and Nationality Act's provisions for asylum seekers...

## Humanitarian Impact

Immigration advocates report that families are being separated at unprecedented rates [S10][S11]. NGOs estimate that over 50,000 migrants are currently stranded in Mexican border towns [S12]...

## Expert Analysis

Immigration law professors have expressed concerns about the legal basis for these policies [S13]. Dr. Sarah Chen of Stanford Law School stated that "the executive order overreaches congressional authority" [S14]...

---

## Sources

[S1] Trump Immigration Policy Changes — https://nytimes.com/...
[S2] Border Policy Analysis — https://reuters.com/...
[S3] Executive Orders Overview — https://whitehouse.gov/...
...
""",
    "messages": [AIMessage(content="...")]  # Same report
}
```

---

## Complete Data Flow Example

Here's the complete flow for our Trump immigration query:

```
USER INPUT
│
│  "Deep research about Trump's new immigrant related policies"
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 1: query_analyzer                                          │
│                                                                 │
│ IN:  messages=[HumanMessage("Deep research...")]                │
│ LLM: Analyze intent, classify, check ambiguity                  │
│ OUT: query_class="current_events", confidence=0.95              │
│      needs_clarification=False                                  │
│      primary_anchor="Trump administration immigration policies" │
└─────────────────────────────────────────────────────────────────┘
│
│ route_after_analysis → "planner" (no clarification needed)
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 3: planner                                                 │
│                                                                 │
│ IN:  query_analysis, enriched_context=None                      │
│ LLM: Generate research plan with questions and queries          │
│ OUT: plan.topic="Trump Admin Immigration Policies (2024-2025)"  │
│      plan.outline=["Executive Summary", "Key Policy Changes",...│
│      plan.queries=[13 optimized search queries]                 │
│      research_iteration=0, max_iterations=3                     │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 4: orchestrator                                            │
│                                                                 │
│ IN:  plan.research_tree, research_iteration=0                   │
│ OUT: subagent_assignments=[4 assignments]                       │
│      SA1: "Key policy changes?" → 3 queries                     │
│      SA2: "Executive orders?" → 2 queries                       │
│      SA3: "Legal challenges?" → 2 queries                       │
│      SA4: "Expert opinions?" → 1 query                          │
└─────────────────────────────────────────────────────────────────┘
│
│ fanout_subagents → Send() to 4 parallel branches
│
├───────────────┬───────────────┬───────────────┐
▼               ▼               ▼               ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│subagent │ │subagent │ │subagent │ │subagent │
│   SA1   │ │   SA2   │ │   SA3   │ │   SA4   │
│         │ │         │ │         │ │         │
│Search,  │ │Search,  │ │Search,  │ │Search,  │
│Fetch,   │ │Fetch,   │ │Fetch,   │ │Fetch,   │
│Extract  │ │Extract  │ │Extract  │ │Extract  │
│         │ │         │ │         │ │         │
│conf=0.78│ │conf=0.82│ │conf=0.65│ │conf=0.70│
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                      │
                      │ Accumulate: raw_sources, raw_evidence
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 6: synthesizer                                             │
│                                                                 │
│ IN:  done_workers=4, subagent_findings=[4 findings]             │
│ OUT: overall_confidence=0.74 (avg of 0.78,0.82,0.65,0.70)       │
│      dead_ends=[...]                                            │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 7: reducer                                                 │
│                                                                 │
│ IN:  raw_sources=[24], raw_evidence=[20]                        │
│ OUT: sources=[12 unique, with SIDs]                             │
│      evidence=[20 items, with EIDs linked to SIDs]              │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 8: gap_detector (iteration 0)                              │
│                                                                 │
│ IN:  sources=[12], evidence=[20], overall_confidence=0.74       │
│ LLM: Analyze coverage by section                                │
│ OUT: knowledge_gaps=[2 gaps: Humanitarian, Expert Analysis]     │
│      refinement_queries=[4 new queries]                         │
│      overall_confidence=0.65                                    │
│      proceed_to_synthesis=False                                 │
└─────────────────────────────────────────────────────────────────┘
│
│ route_after_gap_detection → "orchestrator" (has gaps)
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 9: increment_iteration                                     │
│                                                                 │
│ IN:  research_iteration=0                                       │
│ OUT: research_iteration=1                                       │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 4: orchestrator (iteration 1)                              │
│                                                                 │
│ IN:  refinement_queries=[4 queries for gaps]                    │
│ OUT: subagent_assignments=[2 assignments for gaps]              │
└─────────────────────────────────────────────────────────────────┘
│
│ ... [subagent → synthesizer → reducer → gap_detector] ...
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 8: gap_detector (iteration 1)                              │
│                                                                 │
│ IN:  sources=[18], evidence=[32], previous_conf=0.65            │
│ LLM: Re-analyze coverage                                        │
│ OUT: Humanitarian Impact: 0.45→0.72 ✓                           │
│      Expert Analysis: 0.35→0.68 ✓                               │
│      overall_confidence=0.78                                    │
│      proceed_to_synthesis=True (sufficient)                     │
└─────────────────────────────────────────────────────────────────┘
│
│ route_after_gap_detection → "trust_engine" (sufficient)
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 10: trust_engine                                           │
│                                                                 │
│ IN:  sources=[18], evidence=[32]                                │
│ LLM1: Credibility scoring + claim extraction                    │
│ LLM2: Verification + cross-validation                           │
│ OUT: source_credibility={18 scores}                             │
│      claims=[13 verified claims]                                │
│      verified_citations=[13 links]                              │
│      cross_validated_claims=[8 multi-source]                    │
│      overall_confidence=0.82                                    │
│      hallucination_score=0.08                                   │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE 11: writer                                                 │
│                                                                 │
│ IN:  plan, sources, claims, citations                           │
│ LLM: Write report using ONLY provided claims+citations          │
│ OUT: report="# Trump Administration Immigration Policies...     │
│              ## Executive Summary                               │
│              The Trump administration has implemented...        │
│              [S1][S2]...                                        │
│              ## Sources                                         │
│              [S1] NYTimes — https://..."                        │
└─────────────────────────────────────────────────────────────────┘
│
▼
[END] → Return final state with report
```

---

## Key Design Decisions

### 1. LLM-Driven Classification (No Regex)
**Why**: The original bug (Trump query → "concept") was caused by hardcoded regex patterns. LLM understands semantic intent.

**How**: `query_analyzer_node` uses a detailed prompt with examples and critical rules.

### 2. Human-in-the-Loop with Graph Interrupt
**Why**: Ambiguous queries like "John Smith professor" need user input before proceeding.

**How**: `interrupt_before=["clarify"]` pauses the graph, application collects input, then resumes.

### 3. Parallel Subagents with Send()
**Why**: Research questions are independent and can run concurrently.

**How**: `fanout_subagents` creates multiple `Send("subagent", ...)` calls that LangGraph executes in parallel.

### 4. Iterative Gap Detection
**Why**: First pass may miss important aspects. Looping allows filling gaps.

**How**: `gap_detector` analyzes coverage, generates refinement queries, routes back to orchestrator.

### 5. Batched Trust Engine (2 LLM calls)
**Why**: Original trust engine had 5 sequential LLM calls. Batching saves ~60% cost.

**How**: Combine credibility+claims in one call, verification+confidence in another.

### 6. Source-Level Deduplication
**Why**: Multiple subagents may find the same URL.

**How**: `reducer_node` deduplicates by URL before assigning SIDs.

### 7. Citation Grounding
**Why**: Every claim must trace back to evidence with exact source attribution.

**How**: Trust engine extracts evidence spans, writer enforces `[S1]` citations in every paragraph.

---

## Summary Statistics

For the Trump immigration query example:

| Metric | Value |
|--------|-------|
| Total Nodes | 12 |
| Research Iterations | 2 |
| Parallel Subagents | 4 per iteration |
| LLM Calls | ~15 (analyzer: 1, planner: 1, subagents: 8, gap: 2, trust: 2, writer: 1) |
| Total Sources | 18 |
| Total Evidence | 32 items |
| Verified Claims | 13 |
| Cross-Validated | 8 (62%) |
| Final Confidence | 0.82 |
| Hallucination Score | 0.08 (8%) |

---

*Architecture designed based on research of OpenAI Deep Research, Anthropic Claude Orchestrator, Google Gemini Deep Research, and Perplexity's RAG pipeline.*
