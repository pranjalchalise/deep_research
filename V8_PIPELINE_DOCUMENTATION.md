# Research Studio v8 - Complete Pipeline Documentation

## Overview

The v8 research pipeline is a sophisticated multi-agent system that conducts deep research with iterative refinement, source verification, and confidence scoring. This document explains every step in detail.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: DISCOVERY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   START                                                                      │
│     │                                                                        │
│     ▼                                                                        │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────────────┐          │
│  │   ANALYZER   │ ───▶ │  DISCOVERY   │ ───▶ │ CONFIDENCE CHECK  │          │
│  └──────────────┘      └──────────────┘      └───────────────────┘          │
│                                                       │                      │
│                              ┌────────────────────────┼───────────────┐      │
│                              ▼                        ▼               ▼      │
│                        ┌─────────┐            ┌─────────────┐   ┌─────────┐ │
│                        │ CLARIFY │            │ AUTO REFINE │   │ PLANNER │ │
│                        └────┬────┘            └──────┬──────┘   └────┬────┘ │
│                             │                        │               │      │
│                             └────────────────────────┴───────────────┘      │
│                                                      │                      │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: MULTI-AGENT RESEARCH                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌───────────────┐                                 │
│                           │  ORCHESTRATOR │◀─────────────────────┐          │
│                           └───────┬───────┘                      │          │
│                                   │                              │          │
│            ┌──────────────────────┼──────────────────────┐       │          │
│            ▼                      ▼                      ▼       │          │
│     ┌────────────┐         ┌────────────┐         ┌────────────┐ │          │
│     │ SUBAGENT 1 │         │ SUBAGENT 2 │   ...   │ SUBAGENT N │ │          │
│     └─────┬──────┘         └─────┬──────┘         └─────┬──────┘ │          │
│           │                      │                      │        │          │
│           └──────────────────────┴──────────────────────┘        │          │
│                                  │                               │          │
│                                  ▼                               │          │
│                         ┌─────────────────┐                      │          │
│                         │   SYNTHESIZER   │                      │          │
│                         └────────┬────────┘                      │          │
│                                  │                               │          │
│                                  ▼                               │          │
│                         ┌─────────────────┐                      │          │
│                         │  GAP DETECTOR   │──────────────────────┘          │
│                         └────────┬────────┘       (loop if gaps)            │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              ▼                   │                   ▼                      │
│       ┌───────────┐              │            ┌───────────┐                 │
│       │ BACKTRACK │──────────────┘            │  REDUCE   │                 │
│       └───────────┘                           └─────┬─────┘                 │
│                                                     │                       │
└─────────────────────────────────────────────────────┼───────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 3: TRUST ENGINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌────────┐   ┌────────┐   ┌──────┐   ┌─────────────┐     │
│  │ CREDIBILITY │──▶│ RANKER │──▶│ CLAIMS │──▶│ CITE │──▶│ SPAN VERIFY │     │
│  └─────────────┘   └────────┘   └────────┘   └──────┘   └──────┬──────┘     │
│                                                                 │           │
│                                                                 ▼           │
│  ┌────────┐   ┌──────────────────┐   ┌─────────────────────────────────┐    │
│  │ WRITER │◀──│ CONFIDENCE SCORE │◀──│     CROSS VALIDATE              │    │
│  └───┬────┘   └──────────────────┘   └─────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│    END                                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: DISCOVERY

### Step 1.1: Analyzer Node

**File:** `src/nodes/discovery.py` → `analyzer_node()`

**Purpose:** Parse and classify the user's research query to understand what type of research is needed.

**Input:**
```python
{
    "messages": [HumanMessage(content="Research Pranjal Chalise at Amherst College")]
}
```

**Process:**
1. **Pattern Matching** - Tries regex patterns to quickly classify:
   - `"X at Y"` → person at organization
   - `"What is X"` → concept query
   - `"X vs Y"` → comparison
   - `"How does X work"` → technical

2. **ERA-CoT LLM Analysis** - Uses GPT-4o-mini with structured prompting:
   - Step 1: Parse query structure
   - Step 2: Classify entities (person, org, concept)
   - Step 3: Map relationships
   - Step 4: Design anchor term strategy
   - Step 5: Assess confidence

3. **Self-Consistency (optional)** - For ambiguous queries, runs 3 LLM passes and takes majority vote

**Output:**
```python
{
    "original_query": "Research Pranjal Chalise at Amherst College",
    "discovery": {
        "query_type": "person",
        "entity_candidates": [],
        "confidence": 0.0,  # Updated by discovery_node
        "needs_clarification": False,
        "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    },
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
}
```

---

### Step 1.2: Discovery Node

**File:** `src/nodes/discovery.py` → `discovery_node()`

**Purpose:** Search the web to find and identify entity candidates, assess disambiguation needs.

**Input:** State from analyzer with `original_query`, `discovery.query_type`, `anchor_terms`

**Process:**
1. **Web Search** - Uses Tavily API to search with anchor terms:
   ```python
   results = cached_search(
       query="Pranjal Chalise Amherst College",
       max_results=10,
       lane="general"
   )
   ```

2. **Entity Clustering** - LLM analyzes search results to:
   - Identify distinct entities (are results about 1 person or multiple?)
   - Extract unique identifiers (institution, role, field)
   - Calculate confidence per candidate

3. **Auto-Selection** - If confidence ≥ 0.8 and single clear entity, auto-select

**Output:**
```python
{
    "discovery": {
        "query_type": "person",
        "entity_candidates": [
            {
                "name": "Pranjal Chalise",
                "description": "Student at Amherst College studying CS",
                "identifiers": ["Amherst College", "Computer Science"],
                "confidence": 0.85
            }
        ],
        "confidence": 0.85,
        "needs_clarification": False,
    },
    "selected_entity": {...},  # If auto-selected
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College", "Computer Science"],
    "clarification_request": None,  # or question if ambiguous
}
```

---

### Step 1.3: Confidence Check Node (v8 NEW)

**File:** `src/nodes/iterative.py` → `confidence_check_node()`

**Purpose:** Decide the next routing based on discovery confidence.

**Input:** State with `discovery.confidence`, `discovery.entity_candidates`

**Process:**
1. Check query type - concept/technical queries skip to planner (rarely ambiguous)
2. Check confidence threshold:
   - ≥ 0.85 → proceed to planner
   - Multiple similar candidates → need clarification
   - < 0.7 with single candidate → auto-refine
   - Explicit needs_clarification → clarify

**Output:**
```python
{
    "_route": "planner" | "clarify" | "auto_refine",
    "refinement_queries": [...]  # If auto_refine
}
```

**Routing:**
- `"planner"` → Skip to planning (high confidence)
- `"clarify"` → Interrupt for human input
- `"auto_refine"` → Try additional searches to increase confidence

---

### Step 1.3a: Clarify Node (if needed)

**File:** `src/nodes/discovery.py` → `clarify_node()`

**Purpose:** Process human clarification response.

**Input:** State with `human_clarification` (user's response)

**Process:**
1. Parse as number (selecting from options)
2. Parse yes/no confirmation
3. Extract new entity + context terms from free-text response

**Output:**
```python
{
    "selected_entity": {...},
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "discovery": {..., "needs_clarification": False}
}
```

---

### Step 1.3b: Auto Refine Node (v8 NEW)

**File:** `src/nodes/iterative.py` → `auto_refine_node()`

**Purpose:** Automatically refine discovery without human input by doing targeted searches.

**Input:** State with `refinement_queries`, `discovery.entity_candidates`

**Process:**
1. Execute refinement searches (LinkedIn, GitHub, official profiles)
2. LLM analyzes new results:
   - Do they confirm the candidate?
   - What new identifiers found?
   - Updated confidence?
3. Update candidate with new info

**Output:**
```python
{
    "_route": "planner" | "clarify",  # Based on new confidence
    "discovery": {...updated...},
    "selected_entity": {...updated...},
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College", "CS", "LinkedIn profile"]
}
```

---

### Step 1.4: Planner Node (v8)

**File:** `src/nodes/planner_v8.py` → `planner_node_v8()`

**Purpose:** Create a hierarchical research plan with a "research tree."

**Input:** State with `original_query`, `primary_anchor`, `anchor_terms`, `discovery.query_type`

**Process:**
1. **Template Selection** - Based on query type:
   - `person` → Use `_generate_person_research_tree()`
   - `concept` → Use `_generate_concept_research_tree()`
   - `technical` → Use `_generate_technical_research_tree()`
   - `comparison` → Use `_generate_comparison_research_tree()`
   - Other → LLM-generated tree

2. **Research Tree Structure:**
   ```python
   {
       "primary": [  # MUST answer (3-4 questions)
           {
               "question": "Who is Pranjal Chalise?",
               "queries": ['"Pranjal Chalise" Amherst', '"Pranjal Chalise" profile'],
               "target_sources": ["official", "news"],
               "confidence_weight": 0.25
           },
           ...
       ],
       "secondary": [...],  # SHOULD answer (2-3 questions)
       "tertiary": [...]    # NICE TO HAVE (1-2 questions)
   }
   ```

3. **Ensure Anchor in Queries** - Every query includes primary anchor:
   ```python
   '"Pranjal Chalise" education university'
   ```

**Output:**
```python
{
    "plan": {
        "topic": "Pranjal Chalise",
        "outline": ["Overview", "Background", "Education", "Career", "Achievements"],
        "queries": [
            {"qid": "Q1", "query": '"Pranjal Chalise" biography', "section": "Overview", "priority": 0.25},
            {"qid": "Q2", "query": '"Pranjal Chalise" education', "section": "Education", "priority": 0.2},
            ...
        ],
        "research_tree": {...}
    },
    "total_workers": 10,
    "done_workers": 0,
    "raw_sources": [],
    "raw_evidence": [],
    "research_iteration": 0,
    "knowledge_gaps": [],
    ...
}
```

---

## PHASE 2: MULTI-AGENT RESEARCH

### Step 2.1: Orchestrator Node (v8 NEW)

**File:** `src/nodes/orchestrator.py` → `orchestrator_node()`

**Purpose:** Lead agent that assigns research questions to parallel subagents.

**Input:** State with `plan.research_tree`, `primary_anchor`, `refinement_queries` (if iterating)

**Process:**
1. **First Iteration** - Create assignments from research tree:
   - Take primary questions (up to 4)
   - Add 1 secondary question if space
   - Limit to `max_subagents` (default: 5)

2. **Subsequent Iterations** - Use `refinement_queries` from gap detector

3. **Build Assignments:**
   ```python
   [
       {
           "subagent_id": "SA1",
           "question": "Who is Pranjal Chalise?",
           "queries": ['"Pranjal Chalise" biography', '"Pranjal Chalise" profile'],
           "target_sources": ["official", "news"]
       },
       ...
   ]
   ```

**Output:**
```python
{
    "subagent_assignments": [...],
    "orchestrator_state": {
        "phase": "primary_research" | "refinement",
        "questions_assigned": 5,
        "questions_completed": 0,
        "overall_confidence": 0.0
    },
    "total_workers": 5,
    "done_workers": 0
}
```

---

### Step 2.2: Subagent Node (v8 NEW) - PARALLEL EXECUTION

**File:** `src/nodes/orchestrator.py` → `subagent_node()`

**Purpose:** Independent worker that researches an assigned question with internal iteration.

**Note:** Multiple subagents run in PARALLEL via LangGraph's `Send` mechanism.

**Input:** Each subagent receives:
```python
{
    "subagent_assignment": {
        "subagent_id": "SA1",
        "question": "What is Pranjal's educational background?",
        "queries": ['"Pranjal Chalise" education', '"Pranjal Chalise" university']
    }
}
```

**Process:**
1. **Iterative Research Loop** (up to `subagent_max_iterations`):
   ```python
   for iteration in range(max_iterations):
       for query in queries:
           # Search
           results = cached_search(query, max_results=6)

           # Add to sources
           all_sources.extend(results)

           # Fetch top 2 pages
           for r in results[:2]:
               chunks = fetch_and_chunk(r["url"])
               evidence = _extract_evidence(chunks, question)
               all_evidence.extend(evidence)

       # Assess confidence
       confidence = _assess_subagent_confidence(question, all_evidence)

       if confidence >= 0.8:
           break  # Sufficient

       # Refine queries for next iteration
       if confidence < 0.7:
           queries = _refine_subagent_queries(question, evidence)
   ```

2. **Evidence Extraction:**
   ```
   LLM Prompt: "Given this page content and question, extract 2-4 evidence items"

   Returns: [{"text": "Pranjal Chalise is a CS major at Amherst College..."}]
   ```

3. **Compress Findings:**
   ```
   LLM Prompt: "Given evidence items, create 3-4 sentence summary"

   Returns: "Pranjal Chalise is a student at Amherst College majoring in..."
   ```

**Output:**
```python
{
    "subagent_findings": [{  # Accumulated via operator.add
        "subagent_id": "SA1",
        "question": "What is educational background?",
        "findings": "Pranjal Chalise is a student at Amherst College...",
        "evidence_ids": [],
        "confidence": 0.75,
        "iterations_used": 2,
        "dead_ends": [{"query": "...", "reason": "no_results"}]
    }],
    "raw_sources": [...],      # Accumulated
    "raw_evidence": [...],     # Accumulated
    "done_workers": 1,
    "done_subagents": 1
}
```

---

### Step 2.3: Synthesizer Node (v8 NEW)

**File:** `src/nodes/orchestrator.py` → `synthesizer_node()`

**Purpose:** Wait for all subagents, combine their findings, update orchestrator state.

**Input:** State with `subagent_findings`, `total_workers`, `done_workers`

**Process:**
1. Wait until `done_workers >= total_workers`
2. Calculate average confidence from all subagents
3. Collect all dead ends
4. Add trajectory step for logging

**Output:**
```python
{
    "orchestrator_state": {
        "questions_completed": 5,
        "overall_confidence": 0.72
    },
    "overall_confidence": 0.72,
    "dead_ends": [...collected from all subagents...],
    "research_trajectory": [..., {
        "iteration": 0,
        "action": "synthesize",
        "result_summary": "Avg confidence: 0.72"
    }]
}
```

---

### Step 2.4: Gap Detector Node (v8 NEW)

**File:** `src/nodes/iterative.py` → `gap_detector_node()`

**Purpose:** Analyze coverage, identify gaps, decide whether to continue researching.

**Input:** State with `evidence`, `sources`, `plan.outline`, `research_iteration`

**Process:**
1. **Section Coverage Analysis:**
   ```python
   for section in outline:
       evidence_count = count(evidence where section == section)
       section_confidence[section] = min(1.0, evidence_count / 3.0)
   ```

2. **Quick Check:**
   - If all sections ≥ 0.8 avg and ≥ 0.5 min → proceed to synthesis

3. **LLM Gap Analysis:**
   ```
   Prompt: "Given evidence by section, identify:
   1. Sections with low coverage
   2. Unanswered aspects
   3. Conflicts needing resolution

   Return: {
       'overall_confidence': 0.65,
       'gaps': [
           {'section': 'Career', 'description': 'No work history found', 'priority': 0.8, 'suggested_queries': [...]}
       ],
       'recommendation': 'continue' | 'sufficient'
   }"
   ```

4. **Decision Logic:**
   ```python
   should_continue = (
       recommendation == "continue" and
       current_iteration < max_iterations and
       overall_confidence < 0.8 and
       len(gaps) > 0
   )
   ```

**Output (if continuing):**
```python
{
    "knowledge_gaps": [{"section": "Career", "description": "...", "priority": 0.8}],
    "overall_confidence": 0.65,
    "proceed_to_synthesis": False,
    "refinement_queries": [
        {"query": '"Pranjal Chalise" career work', "section": "Career", "priority": 0.8}
    ],
    "research_iteration": 1  # Incremented
}
```
→ Routes back to **Orchestrator** for another iteration

**Output (if sufficient):**
```python
{
    "knowledge_gaps": [],
    "overall_confidence": 0.82,
    "proceed_to_synthesis": True,
    "research_iteration": 0
}
```
→ Routes to **Reduce** node

---

### Step 2.5: Backtrack Handler Node (v8 NEW)

**File:** `src/nodes/iterative.py` → `backtrack_handler_node()`

**Purpose:** When searches fail, generate alternative approaches.

**Input:** State with `dead_ends` containing unhandled failures

**Process:**
1. **Failure Type Analysis:**
   - `no_results` → Broaden query (remove quotes, add synonyms)
   - `irrelevant` → Add more context terms
   - `paywall` → Try alternative domains (wikipedia, github)
   - `low_credibility` → Target .edu, .gov sites

2. **Generate Alternatives:**
   ```python
   if reason == "no_results":
       alternatives = [
           failed_query.replace('"', ''),  # Remove quotes
           f"{failed_query} OR background OR profile",
       ]
   elif reason == "paywall":
       alternatives = [
           f'{failed_query} site:wikipedia.org',
           f'{failed_query} -site:wsj.com -site:nytimes.com'
       ]
   ```

3. Mark dead ends as handled

**Output:**
```python
{
    "dead_ends": [...updated with alternative_tried=True...],
    "refinement_queries": [
        {"query": "Pranjal Chalise profile", "section": "General", "priority": 0.7}
    ],
    "research_trajectory": [..., {"action": "backtrack", ...}]
}
```
→ Routes to **Orchestrator** to try alternatives

---

### Step 2.6: Reduce Node

**File:** `src/nodes/reducer.py` → `reducer_node()`

**Purpose:** Merge and normalize all raw sources/evidence after workers complete.

**Input:** State with `raw_sources` (accumulated), `raw_evidence` (accumulated)

**Process:**
1. Wait until `done_workers >= total_workers`
2. Deduplicate sources by URL
3. Assign unique IDs:
   - Sources: S1, S2, S3...
   - Evidence: E1, E2, E3...
4. Link evidence to sources via `sid`

**Output:**
```python
{
    "sources": [
        {"sid": "S1", "url": "https://...", "title": "...", "snippet": "..."},
        {"sid": "S2", ...}
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "url": "...", "text": "..."},
        {"eid": "E2", "sid": "S1", ...},
        {"eid": "E3", "sid": "S2", ...}
    ]
}
```

---

## PHASE 3: TRUST ENGINE

### Step 3.1: Credibility Scorer Node (v8 NEW)

**File:** `src/nodes/trust_engine.py` → `credibility_scorer_node()`

**Purpose:** Score each source's credibility using E-E-A-T principles.

**Input:** State with `sources`, `evidence`

**Process:**
1. **Domain Trust (rule-based):**
   ```python
   DOMAIN_TRUST_HIGH = {
       ".edu": 0.9, ".gov": 0.9, "arxiv.org": 0.9,
       "nature.com": 0.9, "wikipedia.org": 0.75
   }
   DOMAIN_TRUST_LOW = {
       "medium.com": 0.45, "quora.com": 0.4, "pinterest.com": 0.3
   }
   ```

2. **LLM Assessment:**
   ```
   Prompt: "For each source, evaluate:
   - AUTHORITY (0-1): Author credentials? Citations? Official status?
   - CONTENT_QUALITY (0-1): Detailed? Specific? Well-structured?"
   ```

3. **Weighted Score:**
   ```python
   overall = (
       domain_trust * 0.30 +
       freshness * 0.15 +
       authority * 0.25 +
       content_quality * 0.30
   )
   ```

4. **Filter** - Remove sources below `min_source_credibility` (default: 0.35)

**Output:**
```python
{
    "sources": [...filtered...],
    "evidence": [...filtered to match remaining sources...],
    "source_credibility": {
        "S1": {
            "sid": "S1",
            "domain_trust": 0.75,
            "freshness": 0.7,
            "authority": 0.8,
            "content_quality": 0.7,
            "overall": 0.74
        },
        ...
    }
}
```

---

### Step 3.2: Ranker Node

**File:** `src/nodes/ranker.py` → `ranker_node()`

**Purpose:** Rank sources by quality, filter irrelevant ones for person queries.

**Input:** State with `sources`, `evidence`, `primary_anchor`, `discovery.query_type`

**Process:**
1. **Entity Relevance Filter** (for person queries):
   ```
   LLM: "Are these sources about the EXACT target entity or different people with similar names?"

   Returns: {"relevant_urls": ["url1", "url2"], "excluded": [...]}
   ```

2. **Quality Scoring:**
   ```python
   score = quality_score(url, title, snippet, lane, published_date)
   # Factors: domain authority, content type, recency
   ```

3. **Sort & Limit** to `rerank_top_n` (default: 15)

**Output:**
```python
{
    "sources": [...top 15 by quality...],
    "evidence": [...filtered to match remaining sources...]
}
```

---

### Step 3.3: Claims Node

**File:** `src/nodes/claims.py` → `claims_node()`

**Purpose:** Generate atomic claims grounded in evidence.

**Input:** State with `plan.outline`, `evidence`

**Process:**
```
LLM Prompt: "Generate 8-24 atomic claims from this evidence.
Each claim must be one sentence and directly supported by evidence.

Return: [{"cid": "C1", "section": "Education", "text": "Pranjal studied CS at Amherst."}]"
```

**Output:**
```python
{
    "claims": [
        {"cid": "C1", "section": "Education", "text": "Pranjal Chalise studies Computer Science at Amherst College."},
        {"cid": "C2", "section": "Projects", "text": "He built a research automation tool."},
        ...
    ]
}
```

---

### Step 3.4: Cite Node

**File:** `src/nodes/cite.py` → `cite_node()`

**Purpose:** Attach evidence IDs to claims.

**Input:** State with `claims`, `evidence`

**Process:**
```
LLM Prompt: "For each claim, identify which evidence items support it.

Return: [{"cid": "C1", "eids": ["E2", "E5"]}]"
```

**Fallback:** If LLM fails to cite a claim, attach best evidence by section.

**Output:**
```python
{
    "citations": [
        {"cid": "C1", "eids": ["E2", "E5"]},
        {"cid": "C2", "eids": ["E3"]},
        ...
    ]
}
```

---

### Step 3.5: Span Verify Node (v8 NEW)

**File:** `src/nodes/trust_engine.py` → `span_verify_node()`

**Purpose:** Verify each claim against EXACT text spans in evidence.

**Input:** State with `claims`, `evidence`, `citations`

**Process:**
```
LLM Prompt: "For each claim, find the EXACT TEXT in evidence that supports it.

Return: [
    {"cid": "C1", "verified": true, "evidence_span": "Pranjal is a CS major...", "match_confidence": 0.95},
    {"cid": "C3", "verified": false, "reason": "No evidence directly supports this"}
]"
```

**Output:**
```python
{
    "verified_citations": [
        {
            "cid": "C1",
            "eid": "E2",
            "claim_text": "Pranjal studies CS at Amherst",
            "evidence_span": "Pranjal Chalise is a Computer Science major at Amherst College",
            "match_score": 0.95,
            "verified": True,
            "cross_validated": False
        },
        ...
    ],
    "unverified_claims": ["C3", "C7"],  # CIDs without verification
    "hallucination_score": 0.15  # 15% of claims unsupported
}
```

---

### Step 3.6: Cross Validate Node (v8 NEW)

**File:** `src/nodes/trust_engine.py` → `cross_validate_node()`

**Purpose:** Check if claims are supported by MULTIPLE independent sources.

**Input:** State with `verified_citations`, `evidence`

**Process:**
```
LLM Prompt: "For each verified claim, check if OTHER evidence (from different sources) also supports it.

Return: [
    {"cid": "C1", "cross_validated": true, "supporting_sids": ["S1", "S3", "S5"]},
    {"cid": "C2", "cross_validated": false, "reason": "Only one source mentions this"}
]"
```

**Output:**
```python
{
    "cross_validated_claims": [
        {..., "cid": "C1", "cross_validated": True, "supporting_sids": ["S1", "S3"]}
    ],
    "single_source_claims": [
        {..., "cid": "C2", "cross_validated": False}
    ]
}
```

---

### Step 3.7: Claim Confidence Scorer Node (v8 NEW)

**File:** `src/nodes/trust_engine.py` → `claim_confidence_scorer_node()`

**Purpose:** Calculate per-claim and overall confidence scores.

**Input:** State with `verified_citations`, `cross_validated_claims`, `source_credibility`, `claims`

**Process:**
```python
for each verified_citation:
    # Base: span match score (0-1)
    base_score = vc["match_score"]

    # Bonus: cross-validation (+0.15)
    cross_bonus = 0.15 if cid in cross_validated_cids else 0

    # Weight by source credibility
    avg_source_cred = average(source_credibility[sid] for sid in supporting_sids)

    # Final
    confidence = base_score * 0.5 + avg_source_cred * 0.35 + cross_bonus
```

**Output:**
```python
{
    "claim_confidence": {
        "C1": 0.92,  # High - cross-validated, high source cred
        "C2": 0.68,  # Medium - single source
        "C5": 0.55   # Lower - weak match
    },
    "section_confidence": {
        "Education": 0.85,
        "Career": 0.45
    },
    "overall_confidence": 0.72
}
```

---

### Step 3.8: Writer Node (v8)

**File:** `src/nodes/writer_v8.py` → `writer_node_v8()`

**Purpose:** Generate final report with confidence indicators and research quality metrics.

**Input:** State with everything from previous steps

**Process:**
1. **Filter Verified Claims Only:**
   ```python
   verified_cids = {vc["cid"] for vc in verified_citations}
   verified_claims = [c for c in claims if c["cid"] in verified_cids]
   ```

2. **Build Claim Packets with Indicators:**
   ```python
   for claim in verified_claims:
       confidence = claim_confidence[cid]

       if cid in cross_validated and confidence >= 0.8:
           indicator = "✓✓"  # High confidence
       elif confidence >= 0.6:
           indicator = "✓"   # Verified
       else:
           indicator = "⚠"   # Lower confidence
   ```

3. **Generate Report:**
   ```
   LLM Prompt: "Write a research report with these claims and indicators.

   CLAIMS:
   [✓✓] (Education) Pranjal studies CS at Amherst. [S1][S3]
   [✓] (Projects) He built a research tool. [S2]
   [⚠] (Awards) He may have received a scholarship. [S4]

   Include a Research Quality section at the end."
   ```

**Output:**
```python
{
    "report": """# Research Report: Pranjal Chalise

## Executive Summary
Pranjal Chalise is a Computer Science student at Amherst College... [✓✓]

## Education
Pranjal is pursuing a degree in Computer Science at Amherst College. [✓✓] [S1][S3]

## Projects
According to his GitHub, he built a research automation tool. [✓] [S2]

...

## Research Quality

| Metric | Value |
|--------|-------|
| Overall Confidence | 72% |
| Verified Claims | 8/10 |
| Cross-Validated | 3 |
| Sources Used | 5 |

## Sources
[S1] LinkedIn Profile — https://linkedin.com/in/...
[S2] GitHub — https://github.com/...
""",
    "messages": [AIMessage(content=report)],
    "research_metadata": {
        "overall_confidence": 0.72,
        "verified_claims": 8,
        "total_claims": 10,
        "knowledge_gaps": 2,
        "sources_used": 5,
        "research_iterations": 2,
        "total_searches": 15
    }
}
```

---

## State Schema Summary

```python
class AgentState(MessagesState):
    # === Discovery ===
    original_query: str
    discovery: DiscoveryResult  # query_type, candidates, confidence
    selected_entity: EntityCandidate
    primary_anchor: str  # Main subject (in EVERY query)
    anchor_terms: List[str]  # Context qualifiers

    # === Planning ===
    plan: Plan  # topic, outline, queries, research_tree

    # === Iterative Research (v8) ===
    research_iteration: int  # 0, 1, 2...
    knowledge_gaps: List[KnowledgeGap]
    research_trajectory: List[TrajectoryStep]
    dead_ends: List[DeadEnd]
    refinement_queries: List[Dict]

    # === Multi-Agent (v8) ===
    subagent_assignments: List[SubagentAssignment]
    subagent_findings: List[SubagentFindings]  # Accumulated
    orchestrator_state: OrchestratorState

    # === Worker Accumulation ===
    total_workers: int
    done_workers: int  # Accumulated
    raw_sources: List[RawSource]  # Accumulated
    raw_evidence: List[RawEvidence]  # Accumulated

    # === Normalized ===
    sources: List[Source]  # With SIDs
    evidence: List[Evidence]  # With EIDs

    # === Trust Engine (v8) ===
    source_credibility: Dict[str, SourceCredibility]
    claims: List[Claim]
    citations: List[ClaimCitations]
    verified_citations: List[VerifiedCitation]
    unverified_claims: List[str]
    cross_validated_claims: List[VerifiedCitation]
    claim_confidence: Dict[str, float]  # CID -> 0-1
    overall_confidence: float

    # === Output ===
    report: str
    research_metadata: ResearchMetadata
```

---

## Configuration (V8Config)

```python
@dataclass
class V8Config:
    # Iterative Research
    max_research_iterations: int = 3
    min_confidence_to_proceed: float = 0.7
    enable_backtracking: bool = True

    # Multi-Agent
    use_multi_agent: bool = True
    max_subagents: int = 5
    subagent_max_iterations: int = 2

    # Source Credibility
    enable_credibility_scoring: bool = True
    min_source_credibility: float = 0.35

    # Citation Verification
    enable_span_verification: bool = True
    enable_cross_validation: bool = True
    span_match_threshold: float = 0.6

    # Confidence Thresholds
    high_confidence_threshold: float = 0.8  # ✓✓
    medium_confidence_threshold: float = 0.6  # ✓
```

---

## Usage

```bash
# Run with multi-agent (default)
python -m src.research_v8 "Who is Satya Nadella?"

# Skip clarification prompt
python -m src.research_v8 --simple "What is quantum computing?"

# Single-agent mode (like v7)
python -m src.research_v8 --single-agent "Compare Python vs JavaScript"
```

---

## Key Improvements Over v7

| Feature | v7 | v8 |
|---------|----|----|
| Research Pattern | Single-pass | Iterative with gap detection |
| Agent Architecture | Single agent | Orchestrator + parallel subagents |
| Dead End Handling | None | Backtracking with alternatives |
| Source Scoring | Basic quality | E-E-A-T credibility |
| Citation | LLM-based | Span verification |
| Multi-source | None | Cross-validation |
| Confidence | Per-report | Per-claim with indicators |
| Output | Plain report | Report + research metadata |
