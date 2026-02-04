# V8 Optimized Pipeline - Complete Technical Guide

This document provides an exhaustive explanation of every step in the Research Studio v8 optimized pipeline, including methods used, prompts sent, edge cases handled, and inputs/outputs at each point.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Phase 1: Discovery](#phase-1-discovery)
   - [Step 1.1: Analyzer Node](#step-11-analyzer-node)
   - [Step 1.2: Discovery Node](#step-12-discovery-node)
   - [Step 1.3: Complexity Router Node](#step-13-complexity-router-node)
   - [Step 1.4: Clarify Node (Conditional)](#step-14-clarify-node-conditional)
3. [Phase 2: Planning](#phase-2-planning)
   - [Step 2.1: Planner Node V8](#step-21-planner-node-v8)
4. [Phase 3: Research](#phase-3-research)
   - [Step 3.1: Orchestrator Node](#step-31-orchestrator-node)
   - [Step 3.2: Subagent Node (Parallel)](#step-32-subagent-node-parallel)
   - [Step 3.3: Synthesizer Node](#step-33-synthesizer-node)
   - [Step 3.4: Early Termination Check](#step-34-early-termination-check)
   - [Step 3.5: Gap Detector Node](#step-35-gap-detector-node)
5. [Phase 4: Synthesis](#phase-4-synthesis)
   - [Step 4.1: Reducer Node](#step-41-reducer-node)
6. [Phase 5: Trust Engine (Batched)](#phase-5-trust-engine-batched)
   - [Step 5.1: Batched Credibility + Claims Node](#step-51-batched-credibility--claims-node)
   - [Step 5.2: Ranker Node](#step-52-ranker-node)
   - [Step 5.3: Batched Verification Node](#step-53-batched-verification-node)
7. [Phase 6: Report Generation](#phase-6-report-generation)
   - [Step 6.1: Writer Node V8](#step-61-writer-node-v8)
8. [Complete State Schema](#complete-state-schema)
9. [Optimization Summary](#optimization-summary)

---

## Pipeline Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DISCOVERY                                                  │
│ ┌─────────┐   ┌───────────┐   ┌──────────────────┐   ┌──────────┐  │
│ │Analyzer │ → │ Discovery │ → │Complexity Router │ → │ Clarify? │  │
│ └─────────┘   └───────────┘   └──────────────────┘   └──────────┘  │
│     │             │                   │                    │        │
│  Pattern +     Tavily           Route by              Human         │
│  ERA-CoT      Search +         complexity            Input          │
│  Analysis     Entity ID                                             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: PLANNING                                                   │
│ ┌─────────────┐                                                     │
│ │ Planner V8  │ → Research Tree (Primary/Secondary/Tertiary)        │
│ └─────────────┘                                                     │
│       │                                                             │
│   Template-based for person/concept/technical                       │
│   LLM-based for complex/custom queries                              │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: RESEARCH (Multi-Agent)                                     │
│ ┌──────────────┐                                                    │
│ │ Orchestrator │ → Assigns questions to subagents                   │
│ └──────────────┘                                                    │
│       │                                                             │
│       ▼ (fan-out)                                                   │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐                             │
│ │Subagent 1│ │Subagent 2│ │Subagent 3│  (parallel execution)        │
│ └──────────┘ └──────────┘ └──────────┘                             │
│       │           │           │                                     │
│       └───────────┼───────────┘                                     │
│                   ▼                                                 │
│ ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐          │
│ │ Synthesizer │ → │Early Termination │ → │ Gap Detector │          │
│ └─────────────┘   └──────────────────┘   └──────────────┘          │
│                                                │                    │
│                        ┌───────────────────────┼───────────────┐    │
│                        ▼                       ▼               ▼    │
│                 [Continue]              [Backtrack]       [Proceed] │
│                 Orchestrator           (pivot strategy)    Reducer  │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: SYNTHESIS                                                  │
│ ┌─────────┐                                                         │
│ │ Reducer │ → Dedup sources, assign IDs, merge evidence             │
│ └─────────┘                                                         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: TRUST ENGINE (Batched - 2 LLM calls instead of 5)         │
│ ┌────────────────────────┐   ┌────────┐   ┌───────────────────────┐│
│ │Batched Cred + Claims   │ → │ Ranker │ → │ Batched Verification  ││
│ │(credibility + extract) │   │        │   │(span + cross + conf)  ││
│ └────────────────────────┘   └────────┘   └───────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6: REPORT                                                     │
│ ┌───────────┐                                                       │
│ │ Writer V8 │ → Report with confidence indicators [✓✓] [✓] [⚠]      │
│ └───────────┘                                                       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Final Report + Metadata
```

---

## Phase 1: Discovery

### Step 1.1: Analyzer Node

**File:** `src/nodes/discovery.py` → `analyzer_node()`

**Purpose:** Analyze the user's query to determine type, extract entities, and identify anchor terms for search.

**Methods Used:**

1. **Pattern-Based Preprocessing** (`pattern_preprocess()`)
   - Regex patterns match common query structures
   - Fast, no LLM call needed

2. **ERA-CoT Analysis** (Entity Relationship Analysis Chain-of-Thought)
   - LLM-based structured reasoning
   - Optional self-consistency with multiple passes

**Input:**
```python
state = {
    "messages": [HumanMessage(content="Research Pranjal Chalise at Amherst College")]
}
```

**LLM Prompt (ERA-CoT):**
```
SYSTEM: You are an expert query analyzer using structured reasoning (ERA-CoT).

Analyze the research query step by step:

**Step 1 - Parse Query Structure:**
Identify the query pattern and extract explicit entities.
- What type of query is this? (asking about a person, concept, technology, event, etc.)
- What entities are mentioned? (names, organizations, technologies, etc.)
- What relationships are implied? ("at" implies affiliation, "by" implies authorship, etc.)

**Step 2 - Entity Classification:**
For each entity found:
- What type is it? (person, organization, concept, technology, product, event)
- How ambiguous is it? (low = unique/specific, medium = few possibilities, high = many possibilities)
- Why might it be ambiguous? (common name, multiple meanings, incomplete info)

**Step 3 - Relationship Mapping:**
What relationships connect the entities?
- affiliation: person AT organization
- creation: person CREATED thing
- part_of: concept PART_OF field
- temporal: event IN time_period
- location: entity IN place

**Step 4 - Anchor Term Strategy:**
Design search anchors to find the RIGHT entity:
- core: Most specific, unique identifying terms (always include these)
- disambiguation: Terms that distinguish from similar entities
- expansion: Related terms for broader coverage (optional)

**Step 5 - Confidence Assessment:**
- How certain is the interpretation? (0.0 to 1.0)
- What could cause confusion?
- Would clarification help?

Return ONLY valid JSON:
{
  "query_type": "person|concept|technical|event|organization|comparison|general",
  "entities": [...],
  "relationships": [...],
  "anchor_strategy": {
    "core": ["most specific identifying terms"],
    "disambiguation": ["terms that help distinguish"],
    "expansion": ["related broader terms"]
  },
  "confidence": 0.0-1.0,
  "clarification_needed": true/false,
  "suggested_clarification": "question to ask if clarification needed",
  "reasoning": "brief explanation of your analysis"
}

USER: Research query: Research Pranjal Chalise at Amherst College
```

**Pattern Matching (Before LLM):**
```python
QUERY_PATTERNS = [
    # "X at Y" → person at organization
    (r"(?i)^(?:research|find|who is)?\s*(.+?)\s+at\s+(.+)$", "person", ["person", "organization"]),
    # ... more patterns
]

# For "Research Pranjal Chalise at Amherst College":
# Matches pattern → query_type="person", extracted={"person": "Pranjal Chalise", "organization": "Amherst College"}
```

**Output:**
```python
{
    "original_query": "Research Pranjal Chalise at Amherst College",
    "discovery": {
        "query_type": "person",
        "entity_candidates": [],  # Populated by discovery_node
        "confidence": 0.0,
        "needs_clarification": False,
        "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    },
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
}
```

**Edge Cases Handled:**
- Empty query → Returns confidence 0.0, needs_clarification=True
- No pattern match → Uses LLM analysis only
- Low LLM confidence → Optional self-consistency (3 passes, majority vote)
- Concept/technical queries → Automatically boost confidence (less ambiguous)

---

### Step 1.2: Discovery Node

**File:** `src/nodes/discovery.py` → `discovery_node()`

**Purpose:** Perform initial Tavily search and identify entity candidates.

**Methods Used:**
1. Tavily Search API (cached)
2. LLM entity analysis

**Input:**
```python
state = {
    "original_query": "Research Pranjal Chalise at Amherst College",
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    "discovery": {"query_type": "person", ...}
}
```

**Tavily API Call:**
```python
cached_search(
    query="Pranjal Chalise Amherst College",  # Anchor terms joined
    max_results=10,
    lane="general",
    use_cache=True,
    cache_dir=".cache_v8/search"
)
```

**LLM Prompt:**
```
SYSTEM: You analyze search results to identify distinct entities and assess confidence.

Given search results for a query, determine:
1. Are these results about ONE clear entity, or MULTIPLE different entities?
2. What are the distinct entity candidates found?
3. How confident are you that we've identified the correct entity?

For PERSON queries, cluster results by:
- Institution/affiliation (same university, company, etc.)
- Role/title (professor, engineer, researcher, etc.)
- Field/domain (AI, biology, economics, etc.)
- Time period (current vs historical)
- Unique identifiers (personal website, LinkedIn, publications)

Return ONLY valid JSON:
{
  "confidence": 0.0-1.0,
  "entity_candidates": [
    {
      "name": "Full name or title",
      "description": "Brief description with key identifiers",
      "identifiers": ["unique identifier 1", "unique identifier 2"],
      "confidence": 0.0-1.0,
      "evidence_urls": ["urls that mention this entity"]
    }
  ],
  "reasoning": "Why you assessed confidence this way",
  "needs_clarification": true/false,
  "clarification_question": "Specific question to disambiguate (if needed)"
}

USER: Query type: person
Original query: Research Pranjal Chalise at Amherst College
Anchor terms: ['Pranjal Chalise', 'Amherst College']

Search results:
[1] Pranjal Chalise - Amherst College
URL: https://amherst.edu/people/pchalise
Snippet: Pranjal Chalise is a student at Amherst College...

[2] ...
```

**Output:**
```python
{
    "discovery": {
        "query_type": "person",
        "entity_candidates": [
            {
                "name": "Pranjal Chalise",
                "description": "Student at Amherst College",
                "identifiers": ["Amherst College", "student"],
                "confidence": 0.85
            }
        ],
        "confidence": 0.85,
        "needs_clarification": False,
        "anchor_terms": ["Amherst College", "student"]
    },
    "selected_entity": {...},
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College", "student"],
    "clarification_request": None
}
```

**Edge Cases Handled:**
- No search results → Low confidence, needs_clarification=True
- Multiple candidates with similar confidence → needs_clarification=True
- Concept/technical queries → Auto-boost confidence (unambiguous)
- Single high-confidence candidate → Auto-select, skip clarification

---

### Step 1.3: Complexity Router Node

**File:** `src/nodes/iterative.py` → `complexity_router_node()`

**Purpose:** Determine query complexity to route to appropriate path (fast/standard/deep).

**Methods Used:**
1. Keyword pattern matching
2. Word count heuristics
3. Config-based routing

**Input:**
```python
state = {
    "original_query": "Research Pranjal Chalise at Amherst College",
    "discovery": {...}
}
```

**Logic (No LLM Call - Rule-Based):**
```python
def get_query_complexity(query: str) -> str:
    query_lower = query.lower()

    # Check for complex indicators
    complex_keywords = ("compare", "analyze", "evaluate", "research",
                        "investigate", "comprehensive", "in-depth")
    for keyword in complex_keywords:
        if keyword in query_lower:
            return "complex"

    # Check for simple indicators
    simple_keywords = ("what is", "who is", "define", "when was", "where is")
    for keyword in simple_keywords:
        if query_lower.startswith(keyword):
            return "simple"

    # Word count heuristic
    word_count = len(query.split())
    if word_count <= 6:
        return "simple"
    elif word_count >= 15:
        return "complex"

    return "medium"
```

**Output (Simple Query):**
```python
{
    "query_complexity": "simple",
    "max_subagents": 1,
    "max_research_iterations": 1,
    "enable_cross_validation": False,
    "batch_trust_engine": True,
    "_fast_path": True
}
```

**Output (Complex Query):**
```python
{
    "query_complexity": "complex",
    "max_subagents": 3,
    "max_research_iterations": 2,
    "enable_cross_validation": True,
    "batch_trust_engine": True,
    "_fast_path": False
}
```

**Routing:**
- `_fast_path=True` → Skips multi-agent, goes to single worker
- `_fast_path=False` → Full orchestrator-subagent flow

---

### Step 1.4: Clarify Node (Conditional)

**File:** `src/nodes/discovery.py` → `clarify_node()`

**Purpose:** Process human clarification input when disambiguation is needed.

**Trigger:** Only runs when `needs_clarification=True` (interrupt happens BEFORE this node)

**Input:**
```python
state = {
    "discovery": {
        "entity_candidates": [
            {"name": "John Smith (MIT)", ...},
            {"name": "John Smith (Stanford)", ...}
        ]
    },
    "human_clarification": "John Smith at MIT professor"  # From user
}
```

**LLM Prompt:**
```
SYSTEM: Extract the PRIMARY entity and CONTEXT qualifiers from the user's clarification.

IMPORTANT: Distinguish between:
- PRIMARY ENTITY: The main subject being researched (person's full name)
- CONTEXT QUALIFIERS: Additional terms that help identify (organization, role)

Return ONLY a JSON object:
{
  "primary_entity": "The main subject's name",
  "entity_type": "person|organization|concept|technology|other",
  "context_qualifiers": ["qualifier1", "qualifier2"],
  "identifiers": ["unique identifier 1", "unique identifier 2"]
}

USER: Query type: person
User clarification: John Smith at MIT professor
```

**Output:**
```python
{
    "selected_entity": {
        "name": "John Smith",
        "description": "professor at MIT",
        "identifiers": ["MIT", "professor"],
        "confidence": 0.9
    },
    "primary_anchor": "John Smith",
    "anchor_terms": ["MIT", "professor"],
    "discovery": {..., "needs_clarification": False}
}
```

**Edge Cases:**
- User enters number (1, 2, 3) → Selects from candidates list
- User enters "yes"/"no" → Confirms/rejects first candidate
- User enters empty → Uses best guess (first candidate)
- User provides free text → LLM extracts entities

---

## Phase 2: Planning

### Step 2.1: Planner Node V8

**File:** `src/nodes/planner_v8.py` → `planner_node_v8()`

**Purpose:** Generate a hierarchical research tree with prioritized questions.

**Methods Used:**
1. **Template-based generation** (for known query types)
2. **LLM-based generation** (for complex/custom queries)

**Input:**
```python
state = {
    "original_query": "Research Pranjal Chalise at Amherst College",
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "discovery": {"query_type": "person"}
}
```

**Template-Based Generation (Person Query):**
```python
def _generate_person_research_tree(name: str, context: List[str]) -> ResearchTree:
    quoted_name = f'"{name}"'  # "Pranjal Chalise"
    ctx = context[0] if context else ""  # "Amherst College"

    primary = [
        {
            "question": f"Who is {name}? What is their background?",
            "queries": [
                f'{quoted_name} {ctx}',  # "Pranjal Chalise" Amherst College
                f'{quoted_name} profile background',
            ],
            "target_sources": ["official", "news"],
            "confidence_weight": 0.25,
        },
        {
            "question": f"What is {name}'s educational background?",
            "queries": [
                f'{quoted_name} education university',
                f'{quoted_name} degree school college',
            ],
            "target_sources": ["academic", "official"],
            "confidence_weight": 0.2,
        },
        # ... more questions
    ]

    secondary = [...]
    tertiary = [...]

    return {"primary": primary, "secondary": secondary, "tertiary": tertiary}
```

**LLM-Based Generation (Complex Queries):**
```
SYSTEM: You are a research planning expert.

Create a RESEARCH TREE that decomposes the research question into structured questions.

Guidelines:
1. PRIMARY questions (3-4): Must answer to fulfill the research goal
2. SECONDARY questions (2-3): Should answer for comprehensive coverage
3. TERTIARY questions (1-2): Nice to have, deeper exploration

For each question, provide:
- 2-3 optimized search queries (3-6 words, quote full names)
- Target source types (academic, news, official, community)
- Confidence weight (how much this contributes to overall understanding)

Query optimization rules:
- Use quotes around full names: "John Smith" MIT
- Keep queries 3-6 words (optimal for search APIs)
- Include context terms for disambiguation

Return ONLY valid JSON:
{
  "topic": "Main topic",
  "outline": ["Section1", "Section2", ...],
  "research_tree": {
    "primary": [...],
    "secondary": [...],
    "tertiary": [...]
  }
}
```

**Output:**
```python
{
    "plan": {
        "topic": "Pranjal Chalise",
        "outline": ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"],
        "queries": [
            {"qid": "Q1", "query": '"Pranjal Chalise" Amherst College', "section": "Who is...", "priority": 0.25},
            {"qid": "Q2", "query": '"Pranjal Chalise" profile background', "section": "Who is...", "priority": 0.25},
            # ... 8-12 total queries
        ],
        "research_tree": {
            "primary": [...],
            "secondary": [...],
            "tertiary": [...]
        }
    },
    "total_workers": 10,
    "done_workers": 0,
    "raw_sources": [],
    "raw_evidence": [],
    "research_iteration": 0
}
```

**Primary Anchor Enforcement:**
```python
# Ensure all queries contain the primary entity name
def _ensure_primary_anchor_in_queries(queries, primary_anchor):
    quoted = f'"{primary_anchor}"'
    for q in queries:
        if primary_anchor.lower() not in q["query"].lower():
            q["query"] = f'{quoted} {q["query"]}'
```

---

## Phase 3: Research

### Step 3.1: Orchestrator Node

**File:** `src/nodes/orchestrator.py` → `orchestrator_node()`

**Purpose:** Assign research questions to parallel subagents with query deduplication.

**Methods Used:**
1. Research tree parsing
2. Query deduplication (OPTIMIZATION)
3. Assignment creation

**Input:**
```python
state = {
    "plan": {
        "research_tree": {...},
        "queries": [...]
    },
    "primary_anchor": "Pranjal Chalise",
    "max_subagents": 3
}
```

**Query Deduplication (Optimization):**
```python
from src.utils.optimization import deduplicate_queries

# Before: 12 queries with overlaps
queries = [
    '"Pranjal Chalise" education university',
    '"Pranjal Chalise" university education',  # ~90% similar
    '"Pranjal Chalise" degree school',
    # ...
]

# After deduplication (threshold=0.85)
deduped, removed = deduplicate_queries(queries, threshold=0.85)
# deduped: 8 queries
# removed: 4 duplicates
```

**Assignment Creation:**
```python
assignments = [
    {
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise? What is their background?",
        "queries": ['"Pranjal Chalise" Amherst College', '"Pranjal Chalise" profile'],
        "target_sources": ["official", "news"]
    },
    {
        "subagent_id": "SA2",
        "question": "What is Pranjal Chalise's educational background?",
        "queries": ['"Pranjal Chalise" education university'],
        "target_sources": ["academic"]
    },
    {
        "subagent_id": "SA3",
        "question": "What are Pranjal Chalise's achievements?",
        "queries": ['"Pranjal Chalise" achievements awards'],
        "target_sources": ["news"]
    }
]
```

**Output:**
```python
{
    "subagent_assignments": assignments,
    "orchestrator_state": {
        "phase": "primary_research",
        "questions_assigned": 3,
        "questions_completed": 0,
        "overall_confidence": 0.0
    },
    "total_workers": 3,
    "done_workers": 0,
    "dedup_stats": {
        "queries_before": 12,
        "queries_after": 8,
        "queries_removed": 4
    }
}
```

**Fan-Out:**
```python
def fanout_subagents(state):
    assignments = state.get("subagent_assignments") or []
    return [
        Send("subagent", {"subagent_assignment": assignment})
        for assignment in assignments
    ]
# Creates 3 parallel subagent executions
```

---

### Step 3.2: Subagent Node (Parallel)

**File:** `src/nodes/orchestrator.py` → `subagent_node()`

**Purpose:** Independently research assigned question with iterative refinement.

**Methods Used:**
1. Tavily search (cached)
2. Full page fetching
3. LLM evidence extraction
4. Confidence assessment
5. Query refinement (if needed)

**Input (Per Subagent):**
```python
state = {
    "subagent_assignment": {
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise?",
        "queries": ['"Pranjal Chalise" Amherst College']
    },
    "primary_anchor": "Pranjal Chalise"
}
```

**Iteration Loop:**
```python
for iteration in range(max_iterations):  # Usually 2
    for query in queries:
        # 1. Search
        results = cached_search(query, max_results=6, ...)

        # 2. Fetch top 2 pages
        for r in results[:2]:
            chunks = fetch_and_chunk(r["url"], ...)

            # 3. Extract evidence
            evidence = _extract_evidence(llm, url, title, chunks, question)
            all_evidence.extend(evidence)

    # 4. Assess confidence
    confidence = _assess_subagent_confidence(question, all_evidence)

    if confidence >= 0.8:
        break  # Sufficient

    # 5. Refine queries for next iteration
    if confidence < 0.7:
        queries = _refine_subagent_queries(question, all_evidence, primary_anchor)
```

**Evidence Extraction Prompt:**
```
SYSTEM: You extract concise, high-signal evidence from source content.

Given the content from a web page and the research question, extract 2-4 evidence items.

Return ONLY a JSON array of:
{"text": "1-3 sentences of factual evidence directly relevant to the question"}

Rules:
- Evidence text must be short, specific, and factual (no fluff)
- Only include information that directly answers or relates to the query
- Prefer specific numbers, dates, findings over vague statements
- Do not invent or hallucinate information

USER: Question: Who is Pranjal Chalise?

Source content:
[Content from fetched page...]

Return JSON only.
```

**Findings Compression Prompt:**
```
SYSTEM: Compress research findings into a concise summary.

Given evidence items collected for a research question, create a compressed summary
that captures the key facts and findings.

Rules:
1. Maximum 3-4 sentences
2. Include only verified facts from the evidence
3. Prioritize unique, specific information
4. Note any gaps or uncertainties

Return a plain text summary (not JSON).

USER: Question: Who is Pranjal Chalise?

Evidence collected:
- Pranjal Chalise is a student at Amherst College...
- He is involved in computer science research...
```

**Output:**
```python
{
    "subagent_findings": [{
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise?",
        "findings": "Pranjal Chalise is a student at Amherst College...",
        "evidence_ids": [],
        "confidence": 0.82,
        "iterations_used": 1,
        "dead_ends": []
    }],
    "raw_sources": [{...}, {...}],
    "raw_evidence": [{...}, {...}],
    "done_workers": 1,
    "done_subagents": 1
}
```

---

### Step 3.3: Synthesizer Node

**File:** `src/nodes/orchestrator.py` → `synthesizer_node()`

**Purpose:** Combine findings from all subagents and calculate overall confidence.

**Input:**
```python
state = {
    "subagent_findings": [
        {"subagent_id": "SA1", "confidence": 0.82, ...},
        {"subagent_id": "SA2", "confidence": 0.75, ...},
        {"subagent_id": "SA3", "confidence": 0.68, ...}
    ],
    "total_workers": 3,
    "done_workers": 3
}
```

**Logic (No LLM Call - Rule-Based):**
```python
def synthesizer_node(state):
    # Wait for all subagents
    if done_workers < total_workers:
        return {}

    # Calculate average confidence
    confidences = [f["confidence"] for f in subagent_findings]
    avg_confidence = sum(confidences) / len(confidences)

    # Collect dead ends
    all_dead_ends = []
    for finding in subagent_findings:
        all_dead_ends.extend(finding.get("dead_ends", []))

    return {
        "orchestrator_state": {..., "overall_confidence": avg_confidence},
        "overall_confidence": avg_confidence,
        "dead_ends": all_dead_ends,
        "research_trajectory": trajectory + [step]
    }
```

**Output:**
```python
{
    "orchestrator_state": {
        "phase": "primary_research",
        "questions_completed": 3,
        "overall_confidence": 0.75
    },
    "overall_confidence": 0.75,
    "dead_ends": [],
    "research_trajectory": [{"iteration": 0, "action": "synthesize", ...}]
}
```

---

### Step 3.4: Early Termination Check

**File:** `src/nodes/iterative.py` → `early_termination_check_node()`

**Purpose:** Check if research should stop early based on diminishing returns.

**Methods Used:**
1. Confidence delta calculation
2. New sources counting
3. Cost budget tracking

**Input:**
```python
state = {
    "overall_confidence": 0.75,
    "previous_confidence": 0.0,  # First iteration
    "research_iteration": 0,
    "raw_sources": [...]  # 15 sources found
}
```

**Logic (No LLM Call - Rule-Based):**
```python
def should_terminate_early(
    current_confidence,
    previous_confidence,
    new_sources_found,
    iteration,
    max_iterations,
    min_confidence_delta=0.05,
    min_new_sources=2,
    cost_tracker=None
):
    # Hard limit
    if iteration >= max_iterations:
        return True, "max_iterations_reached"

    # Cost budget exceeded
    if cost_tracker and cost_tracker.is_over_budget():
        return True, "cost_budget_exceeded"

    # Diminishing returns
    confidence_delta = current_confidence - previous_confidence
    if confidence_delta < min_confidence_delta and iteration > 0:
        return True, f"diminishing_returns (delta={confidence_delta:.2%})"

    # Not finding new sources
    if new_sources_found < min_new_sources and iteration > 0:
        return True, f"insufficient_new_sources ({new_sources_found})"

    # High confidence achieved
    if current_confidence >= 0.9:
        return True, "high_confidence_achieved"

    return False, ""
```

**Output (Continue):**
```python
{
    "should_terminate": False,
    "termination_reason": "",
    "previous_confidence": 0.75,
    "previous_source_count": 15
}
```

**Output (Terminate):**
```python
{
    "should_terminate": True,
    "termination_reason": "high_confidence_achieved",
    ...
}
```

---

### Step 3.5: Gap Detector Node

**File:** `src/nodes/iterative.py` → `gap_detector_node()`

**Purpose:** Identify knowledge gaps and decide whether to continue research.

**Methods Used:**
1. Section-based evidence counting
2. LLM gap analysis
3. Refinement query generation

**Input:**
```python
state = {
    "evidence": [{...}, {...}, ...],  # 20 evidence items
    "plan": {"outline": ["Overview", "Education", "Career", ...]},
    "research_iteration": 0,
    "max_research_iterations": 2
}
```

**LLM Prompt:**
```
SYSTEM: Analyze research findings and identify knowledge gaps.

Given:
- Original question
- Research outline (target sections)
- Current findings by section
- Evidence collected

Identify:
1. Sections with low coverage (insufficient evidence)
2. Unanswered aspects of the question
3. Conflicting information needing resolution
4. Missing perspectives

Return JSON:
{
  "overall_confidence": 0.0-1.0,
  "section_confidence": {"Section Name": 0.0-1.0, ...},
  "gaps": [
    {
      "section": "Section Name",
      "description": "What information is missing",
      "priority": 0.0-1.0,
      "suggested_queries": ["query1", "query2"]
    }
  ],
  "conflicts": [...],
  "recommendation": "continue" | "sufficient",
  "reasoning": "explanation"
}

USER: Original question: Research Pranjal Chalise at Amherst College

Target sections: ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]

Current evidence by section:
**Overview** (5 items):
- Pranjal Chalise is a student at Amherst College...
- ...

**Education** (3 items):
- ...

**Career** (0 items):
No evidence found

Total sources: 15
Research iteration: 1 of 2

Analyze gaps and recommend whether to continue researching or proceed to synthesis.
```

**Decision Logic:**
```python
should_continue = (
    recommendation == "continue" and
    current_iteration < max_iterations and
    overall_confidence < 0.8 and
    len(gaps) > 0
)

if should_continue:
    # Generate refinement queries from gaps
    refinement_queries = []
    for gap in sorted(gaps, key=lambda g: g["priority"], reverse=True)[:3]:
        for query in gap["suggested_queries"][:2]:
            if primary_anchor not in query.lower():
                query = f'"{primary_anchor}" {query}'
            refinement_queries.append({"query": query, "section": gap["section"]})
```

**Output (Continue):**
```python
{
    "knowledge_gaps": [
        {"section": "Career", "description": "No career info", "priority": 0.8, ...}
    ],
    "overall_confidence": 0.68,
    "section_confidence": {"Overview": 0.9, "Education": 0.7, "Career": 0.0},
    "proceed_to_synthesis": False,
    "refinement_queries": [
        {"query": '"Pranjal Chalise" career work projects', "section": "Career"}
    ],
    "research_iteration": 1
}
# Routes back to orchestrator
```

**Output (Sufficient):**
```python
{
    "knowledge_gaps": [],
    "overall_confidence": 0.85,
    "proceed_to_synthesis": True,
    "research_iteration": 1
}
# Routes to reducer
```

---

## Phase 4: Synthesis

### Step 4.1: Reducer Node

**File:** `src/nodes/reducer.py` → `reducer_node()`

**Purpose:** Deduplicate sources and assign unique IDs.

**Methods Used:**
1. URL-based deduplication
2. Sequential ID assignment

**Input:**
```python
state = {
    "total_workers": 3,
    "done_workers": 3,
    "raw_sources": [
        {"url": "https://...", "title": "Page 1", "snippet": "..."},
        {"url": "https://...", "title": "Page 1", "snippet": "..."},  # Duplicate
        {"url": "https://...", "title": "Page 2", "snippet": "..."},
        # ... 20 total
    ],
    "raw_evidence": [
        {"url": "https://...", "title": "...", "section": "...", "text": "..."},
        # ... 30 total
    ]
}
```

**Logic (No LLM Call - Rule-Based):**
```python
def reducer_node(state):
    if done_workers < total_workers:
        return {}  # Not ready

    # Deduplicate by URL
    deduped = dedup_sources(raw_sources)  # 20 → 12 unique

    # Assign source IDs: S1, S2, S3, ...
    sources = assign_source_ids(deduped)

    # Build URL → SID mapping
    url_to_sid = {s["url"]: s["sid"] for s in sources}

    # Assign evidence IDs: E1, E2, E3, ...
    # Link evidence to source via SID
    evidence = assign_evidence_ids(raw_evidence, url_to_sid)

    return {"sources": sources, "evidence": evidence}
```

**Output:**
```python
{
    "sources": [
        {"sid": "S1", "url": "https://...", "title": "Page 1", "snippet": "..."},
        {"sid": "S2", "url": "https://...", "title": "Page 2", "snippet": "..."},
        # ... 12 unique sources
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "url": "...", "text": "Evidence text..."},
        {"eid": "E2", "sid": "S1", "url": "...", "text": "More evidence..."},
        # ... 25 evidence items
    ]
}
```

---

## Phase 5: Trust Engine (Batched)

### Step 5.1: Batched Credibility + Claims Node

**File:** `src/nodes/trust_engine_batched.py` → `batched_credibility_claims_node()`

**Purpose:** Score source credibility AND extract claims in single LLM call.

**This is an OPTIMIZATION - replaces 2 separate nodes.**

**Methods Used:**
1. Rule-based domain trust scoring
2. Single LLM call for authority + content quality + claims

**Input:**
```python
state = {
    "sources": [
        {"sid": "S1", "url": "https://amherst.edu/...", "title": "...", "snippet": "..."},
        # ... 12 sources
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "text": "Pranjal Chalise is..."},
        # ... 25 evidence items
    ],
    "plan": {"outline": ["Overview", "Education", "Career", ...]}
}
```

**Domain Trust Scoring (No LLM - Rule-Based):**
```python
DOMAIN_TRUST_HIGH = {
    ".edu": 0.9,  # amherst.edu → 0.9
    ".gov": 0.9,
    "arxiv.org": 0.9,
    "wikipedia.org": 0.75,
    "linkedin.com": 0.7,
}

def _calculate_domain_trust(url):
    host = urlparse(url).hostname.lower()
    for pattern, score in DOMAIN_TRUST_HIGH.items():
        if pattern.startswith("."):
            if host.endswith(pattern):
                return score
        else:
            if pattern in host:
                return score
    return 0.5  # Default
```

**LLM Prompt (BATCHED - does 2 tasks at once):**
```
SYSTEM: You are an expert at analyzing sources and extracting factual claims.

TASK 1: Assess source credibility
For each source, evaluate:
- AUTHORITY (0-1): Author credentials, citations, official status
- CONTENT_QUALITY (0-1): Detailed, specific, well-structured content

TASK 2: Extract factual claims
From the evidence, extract 8-15 distinct factual claims organized by section.
Each claim should be:
- A single, verifiable statement
- Directly supported by the evidence
- Labeled with which section it belongs to

Return JSON with this structure:
{
  "source_assessments": [
    {"sid": "S1", "authority": 0.7, "content_quality": 0.8}
  ],
  "claims": [
    {"cid": "C1", "text": "Factual claim here", "section": "Background", "supporting_eids": ["E1", "E3"]}
  ]
}

USER: SOURCES:
[S1] Pranjal Chalise - Amherst College
URL: https://amherst.edu/people/pchalise
Snippet: Pranjal Chalise is a student at Amherst College...

[S2] ...

EVIDENCE:
[E1] (from S1)
Pranjal Chalise is a computer science student at Amherst College with interests in...

[E2] ...

SECTIONS to organize claims: Overview, Background, Education, Career, Achievements, Recent Activity

Perform both tasks and return the combined JSON.
```

**Credibility Calculation:**
```python
# Combine scores with weights
overall = (
    domain_trust * 0.30 +     # 0.9 * 0.3 = 0.27 (for .edu)
    freshness * 0.15 +        # 0.7 * 0.15 = 0.105
    authority * 0.25 +        # 0.8 * 0.25 = 0.2
    content_quality * 0.30    # 0.8 * 0.3 = 0.24
)
# Total: 0.815
```

**Output:**
```python
{
    "sources": [  # Filtered by min_credibility (0.35)
        {"sid": "S1", "url": "...", "credibility": 0.815, ...},
        # ... 10 sources (2 filtered out)
    ],
    "evidence": [  # Only from valid sources
        {"eid": "E1", "sid": "S1", ...},
        # ... 22 evidence items
    ],
    "source_credibility": {
        "S1": {
            "sid": "S1",
            "domain_trust": 0.9,
            "authority": 0.8,
            "content_quality": 0.8,
            "overall": 0.815
        },
        # ...
    },
    "claims": [
        {"cid": "C1", "text": "Pranjal Chalise is a student at Amherst College", "section": "Overview", "eids": ["E1"]},
        {"cid": "C2", "text": "He studies computer science", "section": "Education", "eids": ["E1", "E3"]},
        # ... 12 claims
    ],
    "citations": [
        {"cid": "C1", "eids": ["E1"]},
        {"cid": "C2", "eids": ["E1", "E3"]},
        # ...
    ]
}
```

---

### Step 5.2: Ranker Node

**File:** `src/nodes/ranker.py` → `ranker_node()`

**Purpose:** Secondary relevance filter and quality-based sorting.

**Methods Used:**
1. LLM-based entity relevance filter (for person queries)
2. Quality scoring
3. Top-N selection

**Input:**
```python
state = {
    "sources": [...],  # 10 sources
    "evidence": [...],
    "discovery": {"query_type": "person"},
    "primary_anchor": "Pranjal Chalise"
}
```

**LLM Prompt (Entity Relevance Filter):**
```
SYSTEM: You filter sources to only include those about the EXACT target entity.

Given:
- TARGET: The specific person/entity being researched
- CONTEXT: Identifying information
- SOURCES: List of source titles and snippets

Return ONLY sources that are clearly about the EXACT target entity.
Exclude sources about:
- Different people with similar names
- Places/organizations with similar names
- People at different institutions than the target

Return JSON:
{
  "relevant_urls": ["url1", "url2"],
  "excluded": [{"url": "...", "reason": "different person - X at Y instead of target"}]
}

Be strict - when uncertain, exclude.

USER: TARGET: Pranjal Chalise
CONTEXT: Amherst College

SOURCES:
- URL: https://amherst.edu/people/pchalise
  Title: Pranjal Chalise - Amherst College
  Snippet: Pranjal Chalise is a student at Amherst College...

- URL: https://othersite.com/pranjal
  Title: Pranjal's Blog
  Snippet: Pranjal shares thoughts on...  (different person?)
```

**Quality Scoring (Rule-Based):**
```python
def quality_score(url, title, snippet, lane, date):
    score = 0.5  # Base

    # Domain bonuses
    if ".edu" in url: score += 0.2
    if ".gov" in url: score += 0.2
    if "wikipedia" in url: score += 0.1

    # Content quality
    if len(snippet) > 200: score += 0.1

    # Lane bonus
    if lane == "academic": score += 0.1

    return min(1.0, score)
```

**Output:**
```python
{
    "sources": [  # Top 15, sorted by quality
        {"sid": "S1", "url": "...", "score": 0.85, ...},
        {"sid": "S3", "url": "...", "score": 0.78, ...},
        # ...
    ],
    "evidence": [  # Filtered to only valid source SIDs
        {"eid": "E1", "sid": "S1", ...},
        # ...
    ]
}
```

---

### Step 5.3: Batched Verification Node

**File:** `src/nodes/trust_engine_batched.py` → `batched_verification_node()`

**Purpose:** Verify claims with span matching, cross-validation, AND confidence scoring in single call.

**This is an OPTIMIZATION - replaces 3 separate nodes.**

**Input:**
```python
state = {
    "claims": [
        {"cid": "C1", "text": "Pranjal Chalise is a student at Amherst College", "section": "Overview"},
        {"cid": "C2", "text": "He studies computer science", "section": "Education"},
        # ... 12 claims
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "text": "Pranjal Chalise is a computer science student at Amherst College..."},
        # ... 22 evidence items
    ],
    "citations": [
        {"cid": "C1", "eids": ["E1"]},
        # ...
    ],
    "source_credibility": {"S1": {"overall": 0.815}, ...}
}
```

**LLM Prompt (BATCHED - does 3 tasks at once):**
```
SYSTEM: You are an expert fact-checker. Perform comprehensive verification of claims.

For each claim:
1. SPAN VERIFICATION: Find the exact text in evidence that supports it
2. CROSS-VALIDATION: Check if multiple sources support it
3. CONFIDENCE SCORING: Rate overall confidence (0-1)

Return JSON array:
[
  {
    "cid": "C1",
    "verified": true,
    "evidence_span": "Exact quote from evidence supporting this",
    "source_eid": "E3",
    "match_confidence": 0.9,
    "cross_validated": true,
    "supporting_sids": ["S1", "S3", "S5"],
    "final_confidence": 0.85
  },
  {
    "cid": "C2",
    "verified": false,
    "reason": "No evidence supports this claim",
    "final_confidence": 0.0
  }
]

Guidelines:
- verified=true: Claim is directly stated or strongly implied in evidence
- match_confidence: 0.9+ for exact matches, 0.7-0.9 for paraphrases, 0.5-0.7 for inferences
- cross_validated=true: At least 2 independent sources support the claim
- final_confidence: Overall reliability considering verification, cross-validation, source quality

USER: CLAIMS TO VERIFY:
[
  {"cid": "C1", "text": "Pranjal Chalise is a student at Amherst College", "section": "Overview"},
  {"cid": "C2", "text": "He studies computer science", "section": "Education"}
]

EVIDENCE:
[E1] (Source: S1)
Pranjal Chalise is a computer science student at Amherst College with interests in AI and machine learning...

[E2] (Source: S2)
Pranjal, a student at Amherst, has worked on several research projects...

SOURCE CREDIBILITY SCORES:
S1: credibility=0.82
S2: credibility=0.75

Verify each claim, checking for span matches and cross-validation.
```

**Output:**
```python
{
    "verified_citations": [
        {
            "cid": "C1",
            "eid": "E1",
            "claim_text": "Pranjal Chalise is a student at Amherst College",
            "evidence_span": "Pranjal Chalise is a computer science student at Amherst College",
            "match_score": 0.95,
            "verified": True,
            "cross_validated": True,
            "supporting_sids": ["S1", "S2"]
        },
        # ...
    ],
    "unverified_claims": ["C5", "C8"],  # Claims that couldn't be verified
    "cross_validated_claims": [...],
    "single_source_claims": [...],
    "claim_confidence": {
        "C1": 0.88,
        "C2": 0.82,
        # ...
    },
    "section_confidence": {
        "Overview": 0.85,
        "Education": 0.78,
        "Career": 0.45
    },
    "overall_confidence": 0.76,
    "hallucination_score": 0.15  # 15% claims unverified
}
```

---

## Phase 6: Report Generation

### Step 6.1: Writer Node V8

**File:** `src/nodes/writer_v8.py` → `writer_node_v8()`

**Purpose:** Generate final report with confidence indicators and quality metrics.

**Input:**
```python
state = {
    "plan": {"topic": "Pranjal Chalise", "outline": [...]},
    "sources": [...],
    "evidence": [...],
    "claims": [...],
    "verified_citations": [...],
    "cross_validated_claims": [...],
    "claim_confidence": {"C1": 0.88, "C2": 0.82, ...},
    "overall_confidence": 0.76,
    "knowledge_gaps": [{"section": "Career", "description": "Limited info"}],
    "source_credibility": {...}
}
```

**Claim Packet Building:**
```python
for claim in verified_claims:
    cid = claim["cid"]
    conf = claim_confidence.get(cid, 0.5)

    # Determine indicator
    if cid in cross_validated and conf >= 0.8:
        indicator = "✓✓"  # High confidence, multiple sources
    elif conf >= 0.6:
        indicator = "✓"   # Verified, single source
    else:
        indicator = "⚠"   # Lower confidence

    claim_packets.append({
        "cid": cid,
        "text": claim["text"],
        "indicator": indicator,
        "cite": ["[S1]", "[S3]"],
        "confidence": conf
    })
```

**LLM Prompt:**
```
SYSTEM: You write a research report with confidence indicators.

Rules:
1. Use ONLY the verified claims provided - do NOT add any information not in the claims
2. Include confidence indicators after each statement:
   - [✓✓] = High confidence (cross-validated by multiple sources)
   - [✓] = Verified (single source, but verified)
   - [⚠] = Lower confidence (use "according to [source]" phrasing)
3. Every paragraph must include at least one source citation like [S1]
4. If information is limited, acknowledge this honestly
5. Include a "Research Quality" section at the end

Structure:
- Title
- Executive Summary (2-3 sentences)
- Main sections following the outline
- Research Quality section
- Sources section with [S#] Title — URL

IMPORTANT: Only write what the evidence supports.

USER: TITLE: Pranjal Chalise

OUTLINE: ['Overview', 'Background', 'Education', 'Career', 'Achievements', 'Recent Activity']

CLAIMS (with confidence indicators):
[✓✓] (Overview) Pranjal Chalise is a student at Amherst College [S1] [S2]
[✓] (Education) He studies computer science [S1]
[⚠] (Career) He has worked on research projects [S3]

SOURCES:
[S1] Pranjal Chalise - Amherst College — https://amherst.edu/people/pchalise
[S2] LinkedIn - Pranjal Chalise — https://linkedin.com/in/...

RESEARCH QUALITY:
- Overall Confidence: 76%
- Verified Claims: 10/12
- Cross-Validated Claims: 5
- Sources Used: 8

KNOWLEDGE GAPS:
- **Career**: Limited information available about work experience

Write a comprehensive research report with confidence indicators.
```

**Output:**
```python
{
    "report": """# Pranjal Chalise

## Executive Summary

Pranjal Chalise is a student at Amherst College [✓✓] studying computer science [✓].
This report compiles verified information from 8 sources with 76% overall confidence.

## Overview

Pranjal Chalise is a student at Amherst College [✓✓] [S1][S2]. He is involved in
computer science and has interests in AI and machine learning [✓] [S1].

## Education

He is pursuing studies in computer science at Amherst College [✓] [S1]. According
to his profile, he has taken courses in machine learning and data structures [⚠] [S3].

## Career

Limited verified information is available about work experience. According to one
source, he has worked on research projects [⚠] [S3].

---

## Research Quality

| Metric | Value |
|--------|-------|
| Overall Confidence | 76% |
| Verified Claims | 10/12 |
| Cross-Validated | 5 |
| Sources Used | 8 |

### Knowledge Gaps
- **Career**: Limited information available

---

## Sources

[S1] Pranjal Chalise - Amherst College — https://amherst.edu/people/pchalise
[S2] LinkedIn - Pranjal Chalise — https://linkedin.com/in/pranjalchalise
...
""",
    "messages": [AIMessage(content=report)],
    "research_metadata": {
        "overall_confidence": 0.76,
        "verified_claims": 10,
        "total_claims": 12,
        "knowledge_gaps": 1,
        "sources_used": 8,
        "research_iterations": 2,
        "total_searches": 12
    }
}
```

**Edge Cases:**
- No sources/evidence → Generates "No Verified Information Found" report
- Claims exist but none verified → Generates "Insufficient Verified Information" report
- Blocking issues → Returns error message

---

## Complete State Schema

```python
AgentState = TypedDict("AgentState", {
    # === Messages ===
    "messages": Annotated[List[BaseMessage], operator.add],

    # === Query Analysis ===
    "original_query": str,
    "primary_anchor": str,           # Main entity name
    "anchor_terms": List[str],       # Context qualifiers
    "query_complexity": str,         # "simple" | "medium" | "complex"

    # === Discovery ===
    "discovery": DiscoveryResult,
    "selected_entity": EntityCandidate,
    "clarification_request": str,
    "human_clarification": str,

    # === Planning ===
    "plan": Plan,
    "total_workers": int,
    "done_workers": int,

    # === Research ===
    "raw_sources": Annotated[List[RawSource], operator.add],
    "raw_evidence": Annotated[List[RawEvidence], operator.add],
    "subagent_assignments": List[SubagentAssignment],
    "subagent_findings": Annotated[List[SubagentFindings], operator.add],
    "orchestrator_state": OrchestratorState,

    # === Iteration ===
    "research_iteration": int,
    "knowledge_gaps": List[KnowledgeGap],
    "refinement_queries": List[Dict],
    "dead_ends": List[DeadEnd],
    "research_trajectory": List[TrajectoryStep],
    "overall_confidence": float,
    "previous_confidence": float,

    # === Synthesis ===
    "sources": List[Source],          # With SIDs
    "evidence": List[Evidence],       # With EIDs

    # === Trust Engine ===
    "source_credibility": Dict[str, SourceCredibility],
    "claims": List[Claim],
    "citations": List[Citation],
    "verified_citations": List[VerifiedCitation],
    "unverified_claims": List[str],
    "cross_validated_claims": List[VerifiedCitation],
    "single_source_claims": List[VerifiedCitation],
    "claim_confidence": Dict[str, float],
    "section_confidence": Dict[str, float],
    "hallucination_score": float,

    # === Output ===
    "report": str,
    "research_metadata": ResearchMetadata,
    "issues": List[Issue],
})
```

---

## Optimization Summary

| Optimization | Original | Optimized | Savings |
|-------------|----------|-----------|---------|
| Trust Engine LLM Calls | 5 | 2 | 60% |
| Query Deduplication | 0 | Yes | ~30% Tavily |
| Complexity Routing | No | Yes | ~50% for simple |
| Early Termination | No | Yes | Variable |
| Model Tiering | gpt-4o everywhere | gpt-4o-mini where possible | ~40% cost |

**Total Resource Usage (Optimized):**
- LLM Calls: ~20 (down from ~32)
- Tavily Searches: ~12 (down from ~18)
- Estimated Cost: ~$0.08-0.12 per query (down from ~$0.15-0.20)
