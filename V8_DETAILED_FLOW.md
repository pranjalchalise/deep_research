# V8 Research Pipeline - Detailed Step-by-Step Flow

## Example Query: "Research Pranjal Chalise at Amherst College"

This document traces the EXACT flow through every node with concrete inputs and outputs.

---

# PHASE 1: DISCOVERY

---

## Step 1: ANALYZER NODE

**File:** `src/nodes/discovery.py:305` → `analyzer_node()`

### What Happens:
```
1. Extract query from messages
2. Try pattern matching (regex)
3. Run LLM analysis (ERA-CoT)
4. Merge results and determine query type
```

### Input State:
```python
{
    "messages": [HumanMessage(content="Research Pranjal Chalise at Amherst College")]
}
```

### Operations:

**Operation 1.1: Extract Query**
```python
original_query = "Research Pranjal Chalise at Amherst College"
```

**Operation 1.2: Pattern Matching**
```python
# Tries regex patterns like:
# r"(?i)^(?:research|find|who is)?\s*(.+?)\s+at\s+(.+)$" → "person"

pattern_result = {
    "pattern_matched": True,
    "query_type": "person",
    "extracted_entities": {
        "person": "Pranjal Chalise",
        "organization": "Amherst College"
    },
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    "pattern_confidence": 0.75
}
```

**Operation 1.3: LLM Analysis (ERA-CoT)**
```python
# Sends to GPT-4o-mini with ANALYZER_SYSTEM_V2 prompt

llm_response = {
    "query_type": "person",
    "entities": [
        {"text": "Pranjal Chalise", "type": "person", "ambiguity": "medium"},
        {"text": "Amherst College", "type": "organization", "ambiguity": "low"}
    ],
    "relationships": [
        {"from": "Pranjal Chalise", "to": "Amherst College", "type": "affiliation"}
    ],
    "anchor_strategy": {
        "core": ["Pranjal Chalise", "Amherst College"],
        "disambiguation": ["student", "computer science"],
        "expansion": ["research", "projects"]
    },
    "confidence": 0.7,
    "clarification_needed": False,
    "reasoning": "Person query with clear institutional affiliation"
}
```

**Operation 1.4: Merge Results**
```python
# Combine pattern + LLM results
final_query_type = "person"
anchor_terms = ["Pranjal Chalise", "Amherst College"]
confidence = 0.7 + 0.1  # Boost because pattern matched same type
```

### Output State Update:
```python
{
    "original_query": "Research Pranjal Chalise at Amherst College",
    "discovery": {
        "query_type": "person",
        "entity_candidates": [],  # Filled by discovery_node
        "confidence": 0.0,        # Filled by discovery_node
        "needs_clarification": False,
        "anchor_terms": ["Pranjal Chalise", "Amherst College"],
        "_analysis": {
            "pattern_result": {...},
            "llm_analysis": {...},
            "entities": [...],
            "relationships": [...]
        }
    },
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    "_suggested_clarification": ""
}
```

### Next Node: `discovery`

---

## Step 2: DISCOVERY NODE

**File:** `src/nodes/discovery.py:481` → `discovery_node()`

### What Happens:
```
1. Build search query from anchor terms
2. Execute Tavily search
3. LLM analyzes results to find entity candidates
4. Calculate confidence and decide if clarification needed
5. Auto-select entity if high confidence
```

### Input State:
```python
{
    "original_query": "Research Pranjal Chalise at Amherst College",
    "discovery": {"query_type": "person", ...},
    "anchor_terms": ["Pranjal Chalise", "Amherst College"],
    "use_cache": True,
    "cache_dir": ".cache_v8"
}
```

### Operations:

**Operation 2.1: Build Search Query**
```python
# For person queries, use anchor terms
search_query = "Pranjal Chalise Amherst College"
```

**Operation 2.2: Tavily Search**
```python
results = cached_search(
    query="Pranjal Chalise Amherst College",
    max_results=10,
    lane="general",
    use_cache=True,
    cache_dir=".cache_v8/search"
)

# Returns:
results = [
    {
        "url": "https://linkedin.com/in/pranjal-chalise",
        "title": "Pranjal Chalise - Amherst College | LinkedIn",
        "snippet": "Pranjal Chalise is a Computer Science student at Amherst College..."
    },
    {
        "url": "https://github.com/pranjalchalise",
        "title": "pranjalchalise (Pranjal Chalise) · GitHub",
        "snippet": "Pranjal Chalise has 15 repositories..."
    },
    {
        "url": "https://amherst.edu/people/students/pranjal",
        "title": "Pranjal Chalise '25 - Amherst College",
        "snippet": "Pranjal is a CS major interested in AI research..."
    },
    # ... more results
]
```

**Operation 2.3: LLM Entity Analysis**
```python
# Sends results to GPT-4o-mini with DISCOVERY_SYSTEM prompt
# Analyzes: Are these about ONE entity or MULTIPLE?

analysis = {
    "confidence": 0.85,
    "entity_candidates": [
        {
            "name": "Pranjal Chalise",
            "description": "Computer Science student at Amherst College, Class of 2025",
            "identifiers": ["Amherst College", "Computer Science", "Class of 2025"],
            "confidence": 0.85,
            "evidence_urls": ["linkedin.com/...", "amherst.edu/..."]
        }
    ],
    "reasoning": "All results consistently refer to same person at Amherst",
    "needs_clarification": False,
    "clarification_question": ""
}
```

**Operation 2.4: Auto-Selection Logic**
```python
# confidence >= 0.8 AND single candidate → auto-select
if confidence >= 0.8 and len(entity_candidates) == 1:
    selected_entity = entity_candidates[0]
    primary_anchor = "Pranjal Chalise"
    needs_clarification = False
```

### Output State Update:
```python
{
    "discovery": {
        "query_type": "person",
        "entity_candidates": [
            {
                "name": "Pranjal Chalise",
                "description": "Computer Science student at Amherst College",
                "identifiers": ["Amherst College", "Computer Science"],
                "confidence": 0.85
            }
        ],
        "confidence": 0.85,
        "needs_clarification": False,
        "anchor_terms": ["Amherst College", "Computer Science"]
    },
    "selected_entity": {
        "name": "Pranjal Chalise",
        "description": "Computer Science student at Amherst College",
        "identifiers": ["Amherst College", "Computer Science"],
        "confidence": 0.85
    },
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College", "Computer Science"],
    "clarification_request": None
}
```

### Next Node: `confidence_check`

---

## Step 3: CONFIDENCE CHECK NODE (v8 NEW)

**File:** `src/nodes/iterative.py:34` → `confidence_check_node()`

### What Happens:
```
1. Check query type (concept/technical skip to planner)
2. Check confidence level
3. Check for multiple similar candidates
4. Route to appropriate next node
```

### Input State:
```python
{
    "discovery": {
        "query_type": "person",
        "confidence": 0.85,
        "entity_candidates": [...],
        "needs_clarification": False
    }
}
```

### Operations:

**Operation 3.1: Query Type Check**
```python
query_type = "person"  # NOT concept/technical, so continue checks
```

**Operation 3.2: Confidence Check**
```python
confidence = 0.85
if confidence >= 0.85:
    route = "planner"  # High confidence → proceed
```

**Operation 3.3: Multiple Candidates Check**
```python
# Only 1 candidate, so no ambiguity
candidates = [{"name": "Pranjal Chalise", "confidence": 0.85}]
# No close competitors
```

### Output State Update:
```python
{
    "_route": "planner"  # Internal routing flag
}
```

### Routing Decision:
```python
def route_after_confidence(state):
    return state.get("_route", "planner")  # Returns "planner"
```

### Next Node: `planner` (skipping clarify and auto_refine)

---

## Step 4: PLANNER NODE V8

**File:** `src/nodes/planner_v8.py:72` → `planner_node_v8()`

### What Happens:
```
1. Determine query type
2. Select appropriate template (person/concept/technical/comparison)
3. Generate research tree with hierarchical questions
4. Convert to flat queries for workers
5. Ensure primary anchor in all queries
```

### Input State:
```python
{
    "original_query": "Research Pranjal Chalise at Amherst College",
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College", "Computer Science"],
    "discovery": {"query_type": "person", ...},
    "selected_entity": {...}
}
```

### Operations:

**Operation 4.1: Select Template**
```python
query_type = "person"
primary_anchor = "Pranjal Chalise"
# Uses _generate_person_research_tree()
```

**Operation 4.2: Generate Research Tree**
```python
research_tree = _generate_person_research_tree("Pranjal Chalise", ["Amherst College"])

# Returns:
research_tree = {
    "primary": [
        {
            "question": "Who is Pranjal Chalise? What is their background?",
            "queries": [
                '"Pranjal Chalise" Amherst College',
                '"Pranjal Chalise" profile background'
            ],
            "target_sources": ["official", "news"],
            "confidence_weight": 0.25
        },
        {
            "question": "What is Pranjal Chalise's educational background?",
            "queries": [
                '"Pranjal Chalise" education university',
                '"Pranjal Chalise" degree school college'
            ],
            "target_sources": ["academic", "official"],
            "confidence_weight": 0.2
        },
        {
            "question": "What is Pranjal Chalise's professional work?",
            "queries": [
                '"Pranjal Chalise" career work experience',
                '"Pranjal Chalise" projects achievements'
            ],
            "target_sources": ["official", "news"],
            "confidence_weight": 0.25
        },
        {
            "question": "What are Pranjal Chalise's notable achievements?",
            "queries": [
                '"Pranjal Chalise" achievements awards',
                '"Pranjal Chalise" contributions notable'
            ],
            "target_sources": ["news", "academic"],
            "confidence_weight": 0.2
        }
    ],
    "secondary": [
        {
            "question": "What is Pranjal Chalise's online presence?",
            "queries": [
                '"Pranjal Chalise" LinkedIn',
                '"Pranjal Chalise" GitHub Twitter'
            ],
            "target_sources": ["official"],
            "confidence_weight": 0.1
        }
    ],
    "tertiary": [
        {
            "question": "What do others say about Pranjal Chalise?",
            "queries": ['"Pranjal Chalise" mentioned recommended'],
            "target_sources": ["community", "news"],
            "confidence_weight": 0.05
        }
    ]
}
```

**Operation 4.3: Flatten to Queries**
```python
queries = _flatten_tree_to_queries(research_tree)

# Returns:
queries = [
    {"qid": "Q1", "query": '"Pranjal Chalise" Amherst College', "section": "Who is Pranjal Chalise?...", "lane": "general", "priority": 0.25},
    {"qid": "Q2", "query": '"Pranjal Chalise" profile background', "section": "Who is Pranjal Chalise?...", "lane": "general", "priority": 0.25},
    {"qid": "Q3", "query": '"Pranjal Chalise" education university', "section": "What is educational...", "lane": "general", "priority": 0.2},
    # ... 11 total queries
]
```

**Operation 4.4: Build Plan**
```python
plan = {
    "topic": "Pranjal Chalise",
    "outline": ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"],
    "queries": queries,  # 11 queries
    "research_tree": research_tree
}
```

### Output State Update:
```python
{
    "plan": {
        "topic": "Pranjal Chalise",
        "outline": ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"],
        "queries": [
            {"qid": "Q1", "query": '"Pranjal Chalise" Amherst College', ...},
            {"qid": "Q2", "query": '"Pranjal Chalise" profile background', ...},
            # ... 11 queries total
        ],
        "research_tree": {
            "primary": [...],   # 4 questions
            "secondary": [...], # 1 question
            "tertiary": [...]   # 1 question
        }
    },
    "total_workers": 11,
    "done_workers": 0,
    "done_subagents": 0,
    "raw_sources": [],
    "raw_evidence": [],
    "sources": None,
    "evidence": None,
    "claims": None,
    "citations": None,
    "issues": [],
    "research_iteration": 0,
    "knowledge_gaps": [],
    "research_trajectory": [],
    "dead_ends": [],
    "subagent_findings": []
}
```

### Next Node: `orchestrator`

---

# PHASE 2: MULTI-AGENT RESEARCH

---

## Step 5: ORCHESTRATOR NODE (v8 NEW)

**File:** `src/nodes/orchestrator.py:63` → `orchestrator_node()`

### What Happens:
```
1. Get research tree from plan
2. Create subagent assignments from questions
3. Limit to max_subagents (5)
4. Initialize orchestrator state
```

### Input State:
```python
{
    "plan": {
        "research_tree": {
            "primary": [4 questions],
            "secondary": [1 question],
            "tertiary": [1 question]
        }
    },
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "research_iteration": 0,
    "max_subagents": 5
}
```

### Operations:

**Operation 5.1: Create Assignments from Tree**
```python
assignments = _create_assignments_from_tree(research_tree, primary_anchor, anchor_terms)

# Takes primary questions (up to 4) + 1 secondary
assignments = [
    {
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise? What is their background?",
        "queries": [
            '"Pranjal Chalise" Amherst College',
            '"Pranjal Chalise" profile background'
        ],
        "target_sources": ["official", "news"]
    },
    {
        "subagent_id": "SA2",
        "question": "What is Pranjal Chalise's educational background?",
        "queries": [
            '"Pranjal Chalise" education university',
            '"Pranjal Chalise" degree school college'
        ],
        "target_sources": ["academic", "official"]
    },
    {
        "subagent_id": "SA3",
        "question": "What is Pranjal Chalise's professional work?",
        "queries": [
            '"Pranjal Chalise" career work experience',
            '"Pranjal Chalise" projects achievements'
        ],
        "target_sources": ["official", "news"]
    },
    {
        "subagent_id": "SA4",
        "question": "What are Pranjal Chalise's notable achievements?",
        "queries": [
            '"Pranjal Chalise" achievements awards',
            '"Pranjal Chalise" contributions notable'
        ],
        "target_sources": ["news", "academic"]
    },
    {
        "subagent_id": "SA5",
        "question": "What is Pranjal Chalise's online presence?",
        "queries": [
            '"Pranjal Chalise" LinkedIn',
            '"Pranjal Chalise" GitHub Twitter'
        ],
        "target_sources": ["official"]
    }
]
```

**Operation 5.2: Initialize Orchestrator State**
```python
orchestrator_state = {
    "phase": "primary_research",  # First iteration
    "questions_assigned": 5,
    "questions_completed": 0,
    "overall_confidence": 0.0
}
```

### Output State Update:
```python
{
    "subagent_assignments": [
        {"subagent_id": "SA1", "question": "Who is...", "queries": [...], ...},
        {"subagent_id": "SA2", "question": "What is educational...", ...},
        {"subagent_id": "SA3", "question": "What is professional...", ...},
        {"subagent_id": "SA4", "question": "What are achievements...", ...},
        {"subagent_id": "SA5", "question": "What is online presence...", ...}
    ],
    "orchestrator_state": {
        "phase": "primary_research",
        "questions_assigned": 5,
        "questions_completed": 0,
        "overall_confidence": 0.0
    },
    "total_workers": 5,
    "done_workers": 0,
    "done_subagents": 0
}
```

### Fanout to Subagents:
```python
def fanout_subagents(state):
    assignments = state["subagent_assignments"]
    return [
        Send("subagent", {"subagent_assignment": assignments[0]}),
        Send("subagent", {"subagent_assignment": assignments[1]}),
        Send("subagent", {"subagent_assignment": assignments[2]}),
        Send("subagent", {"subagent_assignment": assignments[3]}),
        Send("subagent", {"subagent_assignment": assignments[4]})
    ]
    # All 5 execute IN PARALLEL
```

### Next Nodes: 5x `subagent` (PARALLEL)

---

## Step 6: SUBAGENT NODE (x5 PARALLEL) (v8 NEW)

**File:** `src/nodes/orchestrator.py:259` → `subagent_node()`

### What Happens (for EACH subagent):
```
1. Get assigned question and queries
2. FOR EACH iteration (up to 2):
   a. FOR EACH query:
      - Execute Tavily search
      - Fetch top 2 page contents
      - Extract evidence via LLM
   b. Assess confidence
   c. If confidence >= 0.8, stop
   d. Else, refine queries
3. Compress findings into summary
4. Return findings + raw evidence
```

### Input State (for SA1):
```python
{
    "subagent_assignment": {
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise? What is their background?",
        "queries": [
            '"Pranjal Chalise" Amherst College',
            '"Pranjal Chalise" profile background'
        ],
        "target_sources": ["official", "news"]
    },
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "use_cache": True,
    "cache_dir": ".cache_v8",
    "subagent_max_iterations": 2
}
```

### Operations (SA1 Example):

**Operation 6.1: First Iteration, Query 1**
```python
# Search
results = cached_search(
    query='"Pranjal Chalise" Amherst College',
    max_results=6,
    lane="general"
)
# Returns 6 results

# Add to sources
all_sources.extend([
    {"url": "https://linkedin.com/in/pranjal-chalise", "title": "Pranjal Chalise - LinkedIn", "snippet": "..."},
    {"url": "https://github.com/pranjalchalise", "title": "pranjalchalise - GitHub", "snippet": "..."},
    # ... 4 more
])

# Fetch top 2 pages
for url in ["linkedin.com/...", "github.com/..."]:
    chunks = fetch_and_chunk(url, chunk_chars=3500, max_chunks=4)

    if chunks:
        # Extract evidence via LLM
        evidence = _extract_evidence(
            llm=llm,
            url=url,
            title="...",
            chunks=chunks,
            question="Who is Pranjal Chalise?",
            max_evidence=3
        )
        all_evidence.extend(evidence)
```

**Operation 6.2: LLM Evidence Extraction**
```python
# Prompt to GPT-4o-mini:
"""
Question: Who is Pranjal Chalise? What is their background?

Source content:
[LinkedIn page content...]

Return JSON only: [{"text": "1-3 sentences of factual evidence"}]
"""

# Response:
evidence = [
    {
        "url": "linkedin.com/...",
        "title": "Pranjal Chalise - LinkedIn",
        "section": "General",
        "text": "Pranjal Chalise is a Computer Science student at Amherst College, Class of 2025."
    },
    {
        "url": "linkedin.com/...",
        "title": "Pranjal Chalise - LinkedIn",
        "section": "General",
        "text": "He has experience in machine learning and software development."
    }
]
```

**Operation 6.3: First Iteration, Query 2**
```python
# Similar process for '"Pranjal Chalise" profile background'
# Adds more sources and evidence
```

**Operation 6.4: Assess Confidence**
```python
def _assess_subagent_confidence(question, evidence):
    evidence_count = len(evidence)  # 6 items
    avg_length = 120  # Average text length

    count_score = min(1.0, 6 / 4) = 1.0
    length_score = min(1.0, 120 / 200) = 0.6

    return count_score * 0.6 + length_score * 0.4 = 0.84

confidence = 0.84  # >= 0.8, so STOP iterating
```

**Operation 6.5: Compress Findings**
```python
# Prompt to GPT-4o-mini:
"""
Question: Who is Pranjal Chalise? What is their background?

Evidence collected:
- Pranjal Chalise is a Computer Science student at Amherst College...
- He has experience in machine learning...
- His GitHub shows 15 repositories...

Create a compressed summary.
"""

# Response:
compressed_findings = "Pranjal Chalise is a Computer Science student at Amherst College, Class of 2025. He has experience in machine learning and software development, with 15 public repositories on GitHub showcasing his work."
```

### Output State Update (from SA1):
```python
{
    "subagent_findings": [{  # Accumulated via operator.add
        "subagent_id": "SA1",
        "question": "Who is Pranjal Chalise? What is their background?",
        "findings": "Pranjal Chalise is a Computer Science student at Amherst College...",
        "evidence_ids": [],
        "confidence": 0.84,
        "iterations_used": 1,
        "dead_ends": []
    }],
    "raw_sources": [
        {"url": "linkedin.com/...", "title": "...", "snippet": "..."},
        {"url": "github.com/...", "title": "...", "snippet": "..."},
        # ... more sources
    ],
    "raw_evidence": [
        {"url": "linkedin.com/...", "title": "...", "section": "General", "text": "Pranjal is a CS student..."},
        {"url": "github.com/...", "title": "...", "section": "General", "text": "He has 15 repos..."},
        # ... more evidence
    ],
    "done_workers": 1,
    "done_subagents": 1
}
```

### All 5 Subagents Complete:
```python
# After all 5 subagents finish (parallel), state has:
{
    "subagent_findings": [
        {"subagent_id": "SA1", "confidence": 0.84, ...},
        {"subagent_id": "SA2", "confidence": 0.72, ...},
        {"subagent_id": "SA3", "confidence": 0.65, ...},
        {"subagent_id": "SA4", "confidence": 0.55, ...},
        {"subagent_id": "SA5", "confidence": 0.90, ...}
    ],
    "raw_sources": [... 25 sources total ...],
    "raw_evidence": [... 30 evidence items total ...],
    "done_workers": 5,
    "done_subagents": 5
}
```

### Next Node: `synthesizer`

---

## Step 7: SYNTHESIZER NODE (v8 NEW)

**File:** `src/nodes/orchestrator.py:480` → `synthesizer_node()`

### What Happens:
```
1. Wait for all subagents to complete
2. Calculate average confidence
3. Collect all dead ends
4. Update orchestrator state
5. Add trajectory step
```

### Input State:
```python
{
    "total_workers": 5,
    "done_workers": 5,
    "subagent_findings": [
        {"subagent_id": "SA1", "confidence": 0.84, "dead_ends": []},
        {"subagent_id": "SA2", "confidence": 0.72, "dead_ends": []},
        {"subagent_id": "SA3", "confidence": 0.65, "dead_ends": [{"query": "...", "reason": "no_results"}]},
        {"subagent_id": "SA4", "confidence": 0.55, "dead_ends": []},
        {"subagent_id": "SA5", "confidence": 0.90, "dead_ends": []}
    ],
    "orchestrator_state": {...}
}
```

### Operations:

**Operation 7.1: Check Completion**
```python
if done_workers < total_workers:
    return {}  # Not ready yet
# All done, proceed
```

**Operation 7.2: Calculate Average Confidence**
```python
confidences = [0.84, 0.72, 0.65, 0.55, 0.90]
avg_confidence = sum(confidences) / 5 = 0.732
```

**Operation 7.3: Collect Dead Ends**
```python
all_dead_ends = []
for finding in subagent_findings:
    all_dead_ends.extend(finding.get("dead_ends", []))

# Result: [{"query": "...", "reason": "no_results"}]
```

**Operation 7.4: Add Trajectory Step**
```python
trajectory_step = {
    "iteration": 0,
    "action": "synthesize",
    "query": "Synthesized 5 subagent findings",
    "result_summary": "Avg confidence: 0.73",
    "confidence_delta": 0.73 - 0.0,  # From 0 to 0.73
    "timestamp": "2024-01-15T10:30:45"
}
```

### Output State Update:
```python
{
    "orchestrator_state": {
        "phase": "primary_research",
        "questions_assigned": 5,
        "questions_completed": 5,
        "overall_confidence": 0.732
    },
    "overall_confidence": 0.732,
    "dead_ends": [{"query": "...", "reason": "no_results", "alternative_tried": False}],
    "research_trajectory": [{
        "iteration": 0,
        "action": "synthesize",
        "result_summary": "Avg confidence: 0.73",
        ...
    }]
}
```

### Routing:
```python
def route_after_synthesis(state):
    return "gap_detector"  # Always goes to gap detector
```

### Next Node: `gap_detector`

---

## Step 8: GAP DETECTOR NODE (v8 NEW)

**File:** `src/nodes/iterative.py:311` → `gap_detector_node()`

### What Happens:
```
1. Group evidence by section
2. Calculate section confidence
3. Quick check: if all sections covered, proceed
4. LLM gap analysis
5. Decision: continue research OR proceed to synthesis
```

### Input State:
```python
{
    "evidence": [... 30 items ...],  # From subagents via reduce
    "sources": [... 25 items ...],
    "plan": {"outline": ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]},
    "research_iteration": 0,
    "max_research_iterations": 3
}
```

### Operations:

**Operation 8.1: Group Evidence by Section**
```python
section_evidence = {
    "Overview": [5 evidence items],
    "Background": [8 evidence items],
    "Education": [6 evidence items],
    "Career": [3 evidence items],      # Low!
    "Achievements": [2 evidence items], # Low!
    "Recent Activity": [6 evidence items]
}
```

**Operation 8.2: Calculate Section Confidence**
```python
# Heuristic: 3+ items = 1.0 confidence
section_confidence = {
    "Overview": 1.0,       # 5/3 = 1.0
    "Background": 1.0,     # 8/3 = 1.0
    "Education": 1.0,      # 6/3 = 1.0
    "Career": 1.0,         # 3/3 = 1.0
    "Achievements": 0.67,  # 2/3 = 0.67
    "Recent Activity": 1.0 # 6/3 = 1.0
}

min_conf = 0.67
avg_conf = 0.95
```

**Operation 8.3: Quick Check**
```python
# avg >= 0.8 AND min >= 0.5? Yes, but let's do LLM analysis anyway
```

**Operation 8.4: LLM Gap Analysis**
```python
# Prompt to GPT-4o-mini:
"""
Original question: Research Pranjal Chalise at Amherst College

Target sections: ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]

Current evidence by section:
**Overview** (5 items):
- Pranjal is a CS student at Amherst...
- He has ML experience...

**Career** (3 items):
- Limited work history found...

Total sources: 25
Research iteration: 1 of 3

Analyze gaps and recommend whether to continue.
"""

# Response:
analysis = {
    "overall_confidence": 0.78,
    "section_confidence": {
        "Overview": 0.9,
        "Background": 0.85,
        "Education": 0.9,
        "Career": 0.5,        # Gap identified!
        "Achievements": 0.6,   # Gap identified!
        "Recent Activity": 0.85
    },
    "gaps": [
        {
            "section": "Career",
            "description": "Limited information about internships or work experience",
            "priority": 0.8,
            "suggested_queries": [
                '"Pranjal Chalise" internship',
                '"Pranjal Chalise" work experience'
            ]
        },
        {
            "section": "Achievements",
            "description": "Few specific awards or recognitions found",
            "priority": 0.6,
            "suggested_queries": [
                '"Pranjal Chalise" award',
                '"Pranjal Chalise" hackathon'
            ]
        }
    ],
    "conflicts": [],
    "recommendation": "continue",  # More research needed
    "reasoning": "Career and Achievements sections have low coverage"
}
```

**Operation 8.5: Decision Logic**
```python
should_continue = (
    recommendation == "continue" and    # True
    current_iteration < max_iterations and  # 0 < 3
    overall_confidence < 0.8 and        # 0.78 < 0.8
    len(gaps) > 0                       # 2 > 0
)
# should_continue = True

# Generate refinement queries
refinement_queries = [
    {"query": '"Pranjal Chalise" internship', "section": "Career", "priority": 0.8},
    {"query": '"Pranjal Chalise" work experience', "section": "Career", "priority": 0.8},
    {"query": '"Pranjal Chalise" award', "section": "Achievements", "priority": 0.6},
    {"query": '"Pranjal Chalise" hackathon', "section": "Achievements", "priority": 0.6}
]
```

### Output State Update:
```python
{
    "knowledge_gaps": [
        {"section": "Career", "description": "Limited work info", "priority": 0.8, ...},
        {"section": "Achievements", "description": "Few awards found", "priority": 0.6, ...}
    ],
    "overall_confidence": 0.78,
    "section_confidence": {"Overview": 0.9, "Career": 0.5, ...},
    "proceed_to_synthesis": False,  # MORE RESEARCH NEEDED
    "refinement_queries": [
        {"query": '"Pranjal Chalise" internship', "section": "Career", ...},
        {"query": '"Pranjal Chalise" award', "section": "Achievements", ...}
    ],
    "research_iteration": 1  # Incremented
}
```

### Routing:
```python
def route_after_gaps(state):
    proceed = state.get("proceed_to_synthesis", False)  # False
    dead_ends = state.get("dead_ends") or []
    unhandled = [d for d in dead_ends if not d.get("alternative_tried")]  # 1 unhandled

    if unhandled and enable_backtracking:
        return "backtrack"  # Handle dead ends first
    elif not proceed:
        return "orchestrator"  # More research
    else:
        return "reduce"  # Proceed to synthesis
```

### Next Node: `backtrack` (because there's an unhandled dead end)

---

## Step 9: BACKTRACK HANDLER NODE (v8 NEW)

**File:** `src/nodes/iterative.py:460` → `backtrack_handler_node()`

### What Happens:
```
1. Find unhandled dead ends
2. For each dead end, generate alternative queries
3. Mark dead ends as handled
4. Add to refinement queries
```

### Input State:
```python
{
    "dead_ends": [
        {"query": '"Pranjal Chalise" career work", "reason": "no_results", "alternative_tried": False}
    ],
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "research_iteration": 1
}
```

### Operations:

**Operation 9.1: Find Unhandled Dead Ends**
```python
unhandled = [d for d in dead_ends if not d.get("alternative_tried")]
# Result: [{"query": '"Pranjal Chalise" career work", "reason": "no_results"}]
```

**Operation 9.2: Generate Alternatives**
```python
for dead_end in unhandled[:3]:
    reason = "no_results"
    failed_query = '"Pranjal Chalise" career work'

    if reason == "no_results":
        alternatives = [
            'Pranjal Chalise career work',  # Remove quotes
            '"Pranjal Chalise" career work OR background OR profile',
            'Pranjal Chalise Amherst College'  # primary + context
        ]
```

**Operation 9.3: Mark as Handled**
```python
updated_dead_ends = [
    {"query": '"Pranjal Chalise" career work", "reason": "no_results", "alternative_tried": True}
]
```

### Output State Update:
```python
{
    "dead_ends": [
        {"query": "...", "reason": "no_results", "alternative_tried": True}  # Marked handled
    ],
    "research_trajectory": [..., {
        "iteration": 1,
        "action": "backtrack",
        "query": "Handling 1 dead ends",
        "result_summary": "Generated 3 alternative queries"
    }],
    "refinement_queries": [
        # Original gaps + alternatives
        {"query": '"Pranjal Chalise" internship', "section": "Career", "priority": 0.8},
        {"query": 'Pranjal Chalise career work', "section": "General", "priority": 0.7},
        {"query": 'Pranjal Chalise career work OR background', "section": "General", "priority": 0.7}
    ]
}
```

### Next Node: `orchestrator` (loop back for iteration 2)

---

## Step 10: ORCHESTRATOR (Iteration 2)

Same as Step 5, but uses `refinement_queries` instead of `research_tree`:

```python
# Creates new assignments from refinement queries
assignments = _create_assignments_from_refinement(refinement_queries, primary_anchor)

assignments = [
    {"subagent_id": "SA1", "question": "Find more about: Career", "queries": ['"Pranjal Chalise" internship', ...]}
]
```

→ Subagents → Synthesizer → Gap Detector → (if sufficient) → Reduce

---

## Step 11: REDUCE NODE

**File:** `src/nodes/reducer.py:8` → `reducer_node()`

### What Happens:
```
1. Wait for all workers to complete
2. Deduplicate sources by URL
3. Assign unique IDs (S1, S2, ...)
4. Assign evidence IDs (E1, E2, ...)
5. Link evidence to sources
```

### Input State:
```python
{
    "total_workers": 5,
    "done_workers": 5,
    "raw_sources": [
        {"url": "linkedin.com/...", "title": "...", "snippet": "..."},
        {"url": "github.com/...", "title": "...", "snippet": "..."},
        {"url": "linkedin.com/...", "title": "...", "snippet": "..."},  # Duplicate!
        # ... 40 total with duplicates
    ],
    "raw_evidence": [
        {"url": "linkedin.com/...", "title": "...", "text": "Pranjal is..."},
        {"url": "github.com/...", "title": "...", "text": "He has 15 repos..."},
        # ... 45 total
    ]
}
```

### Operations:

**Operation 11.1: Deduplicate Sources**
```python
deduped = dedup_sources(raw_sources)
# 40 → 28 unique sources
```

**Operation 11.2: Assign Source IDs**
```python
sources = assign_source_ids(deduped)

sources = [
    {"sid": "S1", "url": "linkedin.com/...", "title": "Pranjal Chalise - LinkedIn", "snippet": "..."},
    {"sid": "S2", "url": "github.com/...", "title": "pranjalchalise - GitHub", "snippet": "..."},
    {"sid": "S3", "url": "amherst.edu/...", "title": "Pranjal Chalise '25", "snippet": "..."},
    # ... S28
]
```

**Operation 11.3: Create URL→SID Mapping**
```python
url_to_sid = {
    "linkedin.com/...": "S1",
    "github.com/...": "S2",
    "amherst.edu/...": "S3",
    # ...
}
```

**Operation 11.4: Assign Evidence IDs**
```python
evidence = assign_evidence_ids(raw_evidence, url_to_sid)

evidence = [
    {"eid": "E1", "sid": "S1", "url": "linkedin.com/...", "text": "Pranjal is a CS student..."},
    {"eid": "E2", "sid": "S1", "url": "linkedin.com/...", "text": "He has ML experience..."},
    {"eid": "E3", "sid": "S2", "url": "github.com/...", "text": "15 public repositories..."},
    # ... E45
]
```

### Output State Update:
```python
{
    "sources": [
        {"sid": "S1", "url": "linkedin.com/...", "title": "...", "snippet": "..."},
        {"sid": "S2", "url": "github.com/...", "title": "...", "snippet": "..."},
        # ... 28 sources
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "url": "...", "title": "...", "section": "...", "text": "..."},
        {"eid": "E2", "sid": "S1", "url": "...", "title": "...", "section": "...", "text": "..."},
        # ... 45 evidence items
    ]
}
```

### Next Node: `credibility`

---

# PHASE 3: TRUST ENGINE

---

## Step 12: CREDIBILITY SCORER NODE (v8 NEW)

**File:** `src/nodes/trust_engine.py:91` → `credibility_scorer_node()`

### What Happens:
```
1. Calculate domain trust for each source (rule-based)
2. LLM assesses authority and content quality
3. Combine into overall credibility score
4. Filter out low-credibility sources
```

### Input State:
```python
{
    "sources": [
        {"sid": "S1", "url": "https://linkedin.com/in/pranjal-chalise", "title": "..."},
        {"sid": "S2", "url": "https://github.com/pranjalchalise", "title": "..."},
        {"sid": "S3", "url": "https://amherst.edu/people/pranjal", "title": "..."},
        {"sid": "S4", "url": "https://medium.com/@random/article", "title": "..."},
        # ... 28 sources
    ],
    "evidence": [... 45 items ...],
    "min_source_credibility": 0.35
}
```

### Operations:

**Operation 12.1: Domain Trust (Rule-Based)**
```python
DOMAIN_TRUST_HIGH = {".edu": 0.9, "linkedin.com": 0.7, "github.com": 0.75}
DOMAIN_TRUST_LOW = {"medium.com": 0.45, "quora.com": 0.4}

domain_scores = {
    "S1": 0.7,   # linkedin.com → MEDIUM
    "S2": 0.75,  # github.com → MEDIUM
    "S3": 0.9,   # .edu → HIGH
    "S4": 0.45,  # medium.com → LOW
    # ...
}
```

**Operation 12.2: LLM Authority Assessment**
```python
# Prompt to GPT-4o-mini:
"""
Assess these sources:

[S1] Pranjal Chalise - LinkedIn
URL: linkedin.com/...
Snippet: Pranjal Chalise is a CS student...

[S2] pranjalchalise - GitHub
URL: github.com/...
Snippet: 15 repositories...

Return: [{"sid": "S1", "authority": 0.8, "content_quality": 0.7}]
"""

# Response:
llm_scores = {
    "S1": {"authority": 0.8, "content_quality": 0.75},
    "S2": {"authority": 0.7, "content_quality": 0.8},
    "S3": {"authority": 0.9, "content_quality": 0.85},
    "S4": {"authority": 0.4, "content_quality": 0.5},
}
```

**Operation 12.3: Calculate Overall Credibility**
```python
for source in sources:
    sid = source["sid"]

    overall = (
        domain_scores[sid] * 0.30 +     # 30% domain
        0.7 * 0.15 +                     # 15% freshness (default)
        llm_scores[sid]["authority"] * 0.25 +
        llm_scores[sid]["content_quality"] * 0.30
    )

    source["credibility"] = overall

# Results:
# S1 (LinkedIn): 0.7*0.3 + 0.7*0.15 + 0.8*0.25 + 0.75*0.3 = 0.74
# S2 (GitHub):   0.75*0.3 + 0.7*0.15 + 0.7*0.25 + 0.8*0.3 = 0.75
# S3 (.edu):     0.9*0.3 + 0.7*0.15 + 0.9*0.25 + 0.85*0.3 = 0.86
# S4 (Medium):   0.45*0.3 + 0.7*0.15 + 0.4*0.25 + 0.5*0.3 = 0.49
```

**Operation 12.4: Filter Low-Credibility**
```python
min_credibility = 0.35

filtered_sources = [s for s in sources if s["credibility"] >= 0.35]
# All 28 pass (even S4 at 0.49 is above threshold)

# Filter evidence to match
valid_sids = {s["sid"] for s in filtered_sources}
filtered_evidence = [e for e in evidence if e["sid"] in valid_sids]
```

### Output State Update:
```python
{
    "sources": [... 28 sources with credibility scores ...],
    "evidence": [... 45 evidence items ...],
    "source_credibility": {
        "S1": {
            "sid": "S1",
            "url": "linkedin.com/...",
            "domain_trust": 0.7,
            "freshness": 0.7,
            "authority": 0.8,
            "content_quality": 0.75,
            "overall": 0.74
        },
        "S2": {..., "overall": 0.75},
        "S3": {..., "overall": 0.86},
        # ... all 28
    }
}
```

### Next Node: `ranker`

---

## Step 13: RANKER NODE

**File:** `src/nodes/ranker.py:86` → `ranker_node()`

### What Happens:
```
1. For person queries, filter sources about wrong entities
2. Calculate quality scores
3. Sort by quality
4. Limit to top N (default: 15)
```

### Input State:
```python
{
    "sources": [... 28 sources ...],
    "evidence": [... 45 items ...],
    "discovery": {"query_type": "person"},
    "primary_anchor": "Pranjal Chalise",
    "anchor_terms": ["Amherst College"],
    "rerank_top_n": 15
}
```

### Operations:

**Operation 13.1: Entity Relevance Filter**
```python
# LLM checks: Are these sources about the EXACT target?

# Prompt:
"""
TARGET: Pranjal Chalise
CONTEXT: Amherst College

SOURCES:
- URL: linkedin.com/in/pranjal-chalise
  Title: Pranjal Chalise - LinkedIn
  Snippet: CS student at Amherst...

- URL: linkedin.com/in/pranjal-other
  Title: Pranjal Smith - LinkedIn
  Snippet: Engineer at Google...

Return relevant URLs only.
"""

# Response:
{"relevant_urls": ["linkedin.com/in/pranjal-chalise", ...], "excluded": ["linkedin.com/in/pranjal-other"]}

# 28 → 26 sources (2 filtered as wrong person)
```

**Operation 13.2: Quality Scoring**
```python
for source in sources:
    source["score"] = quality_score(
        url=source["url"],
        title=source["title"],
        snippet=source["snippet"],
        lane="general",
        published_date=""
    )
    # Considers: domain authority, content type, recency
```

**Operation 13.3: Sort and Limit**
```python
sources_sorted = sorted(sources, key=lambda s: s["score"], reverse=True)
sources_final = sources_sorted[:15]  # Top 15

# Filter evidence to match
valid_sids = {s["sid"] for s in sources_final}
evidence_final = [e for e in evidence if e["sid"] in valid_sids]
# 45 → 32 evidence items
```

### Output State Update:
```python
{
    "sources": [... top 15 sources ...],
    "evidence": [... 32 evidence items from those sources ...]
}
```

### Next Node: `claims`

---

## Step 14: CLAIMS NODE

**File:** `src/nodes/claims.py:23` → `claims_node()`

### What Happens:
```
1. Get outline from plan
2. Format evidence for LLM
3. Generate 8-24 atomic claims
```

### Input State:
```python
{
    "plan": {"outline": ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]},
    "evidence": [
        {"eid": "E1", "text": "Pranjal Chalise is a CS student at Amherst College, Class of 2025."},
        {"eid": "E2", "text": "He has experience in machine learning and software development."},
        {"eid": "E3", "text": "His GitHub shows 15 public repositories."},
        # ... 32 items
    ]
}
```

### Operations:

**Operation 14.1: Format Evidence**
```python
ev_block = """
[E1] Pranjal Chalise is a CS student at Amherst College, Class of 2025.
[E2] He has experience in machine learning and software development.
[E3] His GitHub shows 15 public repositories.
...
"""
```

**Operation 14.2: LLM Claim Generation**
```python
# Prompt to GPT-4o-mini:
"""
Outline:
["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]

Evidence:
[E1] Pranjal Chalise is a CS student...
[E2] He has ML experience...
...

Generate 8-24 atomic claims. Each must be one sentence supported by evidence.
Return JSON: [{"cid": "C1", "section": "...", "text": "..."}]
"""

# Response:
claims = [
    {"cid": "C1", "section": "Overview", "text": "Pranjal Chalise is a Computer Science student at Amherst College."},
    {"cid": "C2", "section": "Overview", "text": "He is part of the Class of 2025."},
    {"cid": "C3", "section": "Background", "text": "Pranjal has experience in machine learning."},
    {"cid": "C4", "section": "Background", "text": "He has software development skills."},
    {"cid": "C5", "section": "Education", "text": "He is pursuing a degree at Amherst College."},
    {"cid": "C6", "section": "Career", "text": "Pranjal has worked on multiple open source projects."},
    {"cid": "C7", "section": "Career", "text": "He maintains 15 public repositories on GitHub."},
    {"cid": "C8", "section": "Achievements", "text": "His projects have received community recognition."},
    {"cid": "C9", "section": "Recent Activity", "text": "He is actively contributing to open source."},
    {"cid": "C10", "section": "Recent Activity", "text": "His GitHub profile shows recent commits."}
]
```

### Output State Update:
```python
{
    "claims": [
        {"cid": "C1", "section": "Overview", "text": "Pranjal Chalise is a CS student at Amherst College."},
        {"cid": "C2", "section": "Overview", "text": "He is part of the Class of 2025."},
        # ... 10 claims total
    ]
}
```

### Next Node: `cite`

---

## Step 15: CITE NODE

**File:** `src/nodes/cite.py:75` → `cite_node()`

### What Happens:
```
1. Get all valid evidence IDs
2. LLM attaches evidence to each claim
3. Fallback heuristics for uncited claims
```

### Input State:
```python
{
    "claims": [
        {"cid": "C1", "text": "Pranjal Chalise is a CS student at Amherst College."},
        {"cid": "C2", "text": "He is part of the Class of 2025."},
        # ... 10 claims
    ],
    "evidence": [
        {"eid": "E1", "sid": "S1", "text": "Pranjal Chalise is a CS student...Class of 2025"},
        {"eid": "E2", "sid": "S1", "text": "ML and software development..."},
        {"eid": "E3", "sid": "S2", "text": "15 public repositories..."},
        # ... 32 items
    ]
}
```

### Operations:

**Operation 15.1: Build Valid EIDs Set**
```python
valid_eids = {"E1", "E2", "E3", ..., "E32"}
```

**Operation 15.2: LLM Citation Matching**
```python
# Prompt to GPT-4o-mini:
"""
CLAIMS:
[{"cid": "C1", "text": "Pranjal Chalise is a CS student at Amherst College."},
 {"cid": "C2", "text": "He is part of the Class of 2025."},
 ...]

EVIDENCE:
[E1] Pranjal Chalise is a CS student at Amherst College, Class of 2025.
[E2] He has experience in ML...
...

Attach evidence to claims. Return: [{"cid": "C1", "eids": ["E1"]}]
"""

# Response:
citations = [
    {"cid": "C1", "eids": ["E1"]},
    {"cid": "C2", "eids": ["E1"]},
    {"cid": "C3", "eids": ["E2"]},
    {"cid": "C4", "eids": ["E2"]},
    {"cid": "C5", "eids": ["E1", "E5"]},
    {"cid": "C6", "eids": ["E3", "E7"]},
    {"cid": "C7", "eids": ["E3"]},
    {"cid": "C8", "eids": ["E12"]},
    {"cid": "C9", "eids": ["E3", "E15"]},
    {"cid": "C10", "eids": ["E15"]}
]
```

### Output State Update:
```python
{
    "citations": [
        {"cid": "C1", "eids": ["E1"]},
        {"cid": "C2", "eids": ["E1"]},
        {"cid": "C3", "eids": ["E2"]},
        # ... 10 citations
    ]
}
```

### Next Node: `span_verify`

---

## Step 16: SPAN VERIFY NODE (v8 NEW)

**File:** `src/nodes/trust_engine.py:262` → `span_verify_node()`

### What Happens:
```
1. For each claim, find EXACT text span in evidence
2. Calculate match confidence
3. Mark as verified or unverified
4. Calculate hallucination score
```

### Input State:
```python
{
    "claims": [... 10 claims ...],
    "evidence": [... 32 items ...],
    "citations": [... 10 citation mappings ...]
}
```

### Operations:

**Operation 16.1: Batch Processing**
```python
# Process claims in batches of 5
batch1 = claims[0:5]  # C1-C5
batch2 = claims[5:10] # C6-C10
```

**Operation 16.2: LLM Span Verification**
```python
# Prompt to GPT-4o-mini:
"""
CLAIMS:
[{"cid": "C1", "text": "Pranjal Chalise is a CS student at Amherst College."}]

EVIDENCE:
[E1] (from S1) Pranjal Chalise is a Computer Science student at Amherst College, Class of 2025.
[E2] (from S1) He has experience in machine learning...

Verify each claim. Find EXACT TEXT that supports it.

Return: [
  {"cid": "C1", "verified": true, "evidence_span": "Pranjal Chalise is a Computer Science student at Amherst College", "source_eid": "E1", "match_confidence": 0.95}
]
"""

# Response:
verifications = [
    {
        "cid": "C1",
        "verified": True,
        "evidence_span": "Pranjal Chalise is a Computer Science student at Amherst College",
        "source_eid": "E1",
        "match_confidence": 0.95
    },
    {
        "cid": "C2",
        "verified": True,
        "evidence_span": "Class of 2025",
        "source_eid": "E1",
        "match_confidence": 0.9
    },
    {
        "cid": "C3",
        "verified": True,
        "evidence_span": "experience in machine learning",
        "source_eid": "E2",
        "match_confidence": 0.85
    },
    # ...
    {
        "cid": "C8",
        "verified": False,
        "reason": "No evidence directly mentions 'community recognition'"
    }
]
```

**Operation 16.3: Build Verified Citations**
```python
verified_citations = []
unverified_claims = []

for v in verifications:
    if v["verified"]:
        verified_citations.append({
            "cid": v["cid"],
            "eid": v["source_eid"],
            "claim_text": claims[v["cid"]]["text"],
            "evidence_span": v["evidence_span"],
            "match_score": v["match_confidence"],
            "verified": True,
            "cross_validated": False,  # Set by next node
            "supporting_sids": []
        })
    else:
        unverified_claims.append(v["cid"])
```

**Operation 16.4: Calculate Hallucination Score**
```python
total_claims = 10
verified_count = 8  # C1-C7, C9-C10 verified; C8 unverified
hallucination_score = 1.0 - (8 / 10) = 0.2  # 20% hallucination
```

### Output State Update:
```python
{
    "verified_citations": [
        {
            "cid": "C1",
            "eid": "E1",
            "claim_text": "Pranjal Chalise is a CS student at Amherst College.",
            "evidence_span": "Pranjal Chalise is a Computer Science student at Amherst College",
            "match_score": 0.95,
            "verified": True,
            "cross_validated": False,
            "supporting_sids": []
        },
        # ... 8 verified citations
    ],
    "unverified_claims": ["C8"],  # 1 claim couldn't be verified
    "hallucination_score": 0.2
}
```

### Next Node: `cross_validate`

---

## Step 17: CROSS VALIDATE NODE (v8 NEW)

**File:** `src/nodes/trust_engine.py:397` → `cross_validate_node()`

### What Happens:
```
1. For each verified claim, check if OTHER sources also support it
2. Mark as cross-validated if 2+ sources agree
```

### Input State:
```python
{
    "verified_citations": [... 8 verified claims ...],
    "evidence": [... 32 items ...]
}
```

### Operations:

**Operation 17.1: Build EID→SID Mapping**
```python
eid_to_sid = {
    "E1": "S1",  # LinkedIn
    "E2": "S1",  # LinkedIn
    "E3": "S2",  # GitHub
    "E5": "S3",  # Amherst.edu
    # ...
}
```

**Operation 17.2: LLM Cross-Validation**
```python
# Prompt to GPT-4o-mini:
"""
VERIFIED CLAIMS:
[{"cid": "C1", "claim": "Pranjal is a CS student at Amherst", "primary_eid": "E1"}]

ALL EVIDENCE:
[E1] (Source: S1) Pranjal Chalise is a CS student at Amherst...
[E5] (Source: S3) Pranjal Chalise '25 - Computer Science major...
[E8] (Source: S4) Amherst student Pranjal Chalise presented...

Identify which claims are supported by MULTIPLE sources.
"""

# Response:
cross_validation = [
    {
        "cid": "C1",
        "cross_validated": True,
        "supporting_eids": ["E1", "E5", "E8"],
        "supporting_sids": ["S1", "S3", "S4"]  # 3 sources!
    },
    {
        "cid": "C2",
        "cross_validated": True,
        "supporting_eids": ["E1", "E5"],
        "supporting_sids": ["S1", "S3"]  # 2 sources
    },
    {
        "cid": "C7",
        "cross_validated": False,
        "reason": "Only GitHub (S2) mentions repository count"
    }
]
```

**Operation 17.3: Split into Categories**
```python
cross_validated_claims = [
    {..., "cid": "C1", "cross_validated": True, "supporting_sids": ["S1", "S3", "S4"]},
    {..., "cid": "C2", "cross_validated": True, "supporting_sids": ["S1", "S3"]},
    {..., "cid": "C5", "cross_validated": True, "supporting_sids": ["S1", "S3"]},
]

single_source_claims = [
    {..., "cid": "C3", "cross_validated": False, "supporting_sids": ["S1"]},
    {..., "cid": "C7", "cross_validated": False, "supporting_sids": ["S2"]},
    # ...
]
```

### Output State Update:
```python
{
    "cross_validated_claims": [
        {"cid": "C1", ..., "cross_validated": True, "supporting_sids": ["S1", "S3", "S4"]},
        {"cid": "C2", ..., "cross_validated": True, "supporting_sids": ["S1", "S3"]},
        {"cid": "C5", ..., "cross_validated": True, "supporting_sids": ["S1", "S3"]}
    ],
    "single_source_claims": [
        {"cid": "C3", ..., "cross_validated": False},
        {"cid": "C4", ..., "cross_validated": False},
        {"cid": "C6", ..., "cross_validated": False},
        {"cid": "C7", ..., "cross_validated": False},
        {"cid": "C9", ..., "cross_validated": False},
        {"cid": "C10", ..., "cross_validated": False}
    ]
}
```

### Next Node: `confidence_score`

---

## Step 18: CLAIM CONFIDENCE SCORER NODE (v8 NEW)

**File:** `src/nodes/trust_engine.py:479` → `claim_confidence_scorer_node()`

### What Happens:
```
1. For each verified claim, calculate confidence
2. Factors: match score + source credibility + cross-validation bonus
3. Calculate section-level and overall confidence
```

### Input State:
```python
{
    "verified_citations": [... 8 verified ...],
    "cross_validated_claims": [... 3 cross-validated ...],
    "single_source_claims": [... 5 single-source ...],
    "source_credibility": {
        "S1": {"overall": 0.74},
        "S2": {"overall": 0.75},
        "S3": {"overall": 0.86},
        "S4": {"overall": 0.49}
    },
    "claims": [... 10 original claims ...]
}
```

### Operations:

**Operation 18.1: Calculate Per-Claim Confidence**
```python
cv_cids = {"C1", "C2", "C5"}  # Cross-validated

for vc in verified_citations:
    cid = vc["cid"]

    # Base: span match score
    base_score = vc["match_score"]  # e.g., 0.95

    # Cross-validation bonus
    cross_bonus = 0.15 if cid in cv_cids else 0

    # Source credibility
    supporting_sids = vc["supporting_sids"]
    avg_source_cred = average([source_credibility[sid]["overall"] for sid in supporting_sids])

    # Final confidence
    confidence = min(1.0, base_score * 0.5 + avg_source_cred * 0.35 + cross_bonus)

# Results:
# C1: 0.95*0.5 + 0.70*0.35 + 0.15 = 0.475 + 0.245 + 0.15 = 0.87
# C2: 0.90*0.5 + 0.80*0.35 + 0.15 = 0.45 + 0.28 + 0.15 = 0.88
# C3: 0.85*0.5 + 0.74*0.35 + 0.0 = 0.425 + 0.259 = 0.68
# C7: 0.80*0.5 + 0.75*0.35 + 0.0 = 0.40 + 0.26 = 0.66
```

**Operation 18.2: Calculate Section Confidence**
```python
section_scores = {
    "Overview": [0.87, 0.88],     # C1, C2
    "Background": [0.68, 0.65],   # C3, C4
    "Education": [0.82],          # C5
    "Career": [0.70, 0.66],       # C6, C7
    "Recent Activity": [0.72, 0.68]  # C9, C10
}

section_confidence = {
    "Overview": 0.875,
    "Background": 0.665,
    "Education": 0.82,
    "Career": 0.68,
    "Recent Activity": 0.70
}
```

**Operation 18.3: Calculate Overall Confidence**
```python
all_confidences = [0.87, 0.88, 0.68, 0.65, 0.82, 0.70, 0.66, 0.72, 0.68]
overall_confidence = sum(all_confidences) / len(all_confidences) = 0.74
```

### Output State Update:
```python
{
    "claim_confidence": {
        "C1": 0.87,
        "C2": 0.88,
        "C3": 0.68,
        "C4": 0.65,
        "C5": 0.82,
        "C6": 0.70,
        "C7": 0.66,
        "C9": 0.72,
        "C10": 0.68
    },
    "section_confidence": {
        "Overview": 0.875,
        "Background": 0.665,
        "Education": 0.82,
        "Career": 0.68,
        "Recent Activity": 0.70
    },
    "overall_confidence": 0.74
}
```

### Next Node: `write`

---

## Step 19: WRITER NODE V8

**File:** `src/nodes/writer_v8.py:67` → `writer_node_v8()`

### What Happens:
```
1. Filter to verified claims only
2. Assign confidence indicators (✓✓, ✓, ⚠)
3. Build research quality metrics
4. Generate report with LLM
5. Return report + metadata
```

### Input State:
```python
{
    "plan": {"topic": "Pranjal Chalise", "outline": [...]},
    "sources": [... 15 sources ...],
    "evidence": [... 32 items ...],
    "claims": [... 10 claims ...],
    "citations": [... 10 mappings ...],
    "verified_citations": [... 8 verified ...],
    "unverified_claims": ["C8"],
    "cross_validated_claims": [... 3 ...],
    "claim_confidence": {"C1": 0.87, "C2": 0.88, ...},
    "overall_confidence": 0.74,
    "knowledge_gaps": [{"section": "Career", ...}],
    "source_credibility": {...}
}
```

### Operations:

**Operation 19.1: Filter Verified Claims**
```python
verified_cids = {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10"}
verified_claims = [c for c in claims if c["cid"] in verified_cids]
# 10 → 9 claims (C8 dropped)
```

**Operation 19.2: Assign Confidence Indicators**
```python
cv_cids = {"C1", "C2", "C5"}  # Cross-validated
high_threshold = 0.8
medium_threshold = 0.6

claim_packets = []
for claim in verified_claims:
    cid = claim["cid"]
    conf = claim_confidence[cid]

    if cid in cv_cids and conf >= 0.8:
        indicator = "✓✓"  # High confidence, cross-validated
    elif conf >= 0.6:
        indicator = "✓"   # Verified
    else:
        indicator = "⚠"   # Lower confidence

    claim_packets.append({
        "cid": cid,
        "section": claim["section"],
        "text": claim["text"],
        "cite": ["[S1]", "[S3]"],  # From citations
        "confidence": conf,
        "indicator": indicator,
        "cross_validated": cid in cv_cids
    })

# Results:
# C1: ✓✓ (0.87, cross-validated)
# C2: ✓✓ (0.88, cross-validated)
# C3: ✓  (0.68, verified)
# C4: ✓  (0.65, verified)
# C5: ✓✓ (0.82, cross-validated)
# C6: ✓  (0.70, verified)
# C7: ✓  (0.66, verified)
# C9: ✓  (0.72, verified)
# C10: ✓ (0.68, verified)
```

**Operation 19.3: Build Quality Metrics**
```python
quality_metrics = {
    "overall_confidence": "74%",
    "verified_claims": "9/10",
    "cross_validated": "3",
    "high_confidence": "3",  # Above 0.8
    "sources_used": 15
}
```

**Operation 19.4: Format Knowledge Gaps**
```python
gaps_summary = """
- **Career**: Limited information about internships or work experience
"""
```

**Operation 19.5: LLM Report Generation**
```python
# Prompt to GPT-4o-mini:
"""
TITLE: Pranjal Chalise

OUTLINE: ["Overview", "Background", "Education", "Career", "Achievements", "Recent Activity"]

CLAIMS (with confidence indicators):
[✓✓] (Overview) Pranjal Chalise is a Computer Science student at Amherst College. [S1][S3]
[✓✓] (Overview) He is part of the Class of 2025. [S1][S3]
[✓] (Background) Pranjal has experience in machine learning. [S1]
[✓] (Background) He has software development skills. [S1]
[✓✓] (Education) He is pursuing a degree at Amherst College. [S1][S3]
[✓] (Career) Pranjal has worked on multiple open source projects. [S2]
[✓] (Career) He maintains 15 public repositories on GitHub. [S2]
[✓] (Recent Activity) He is actively contributing to open source. [S2]
[✓] (Recent Activity) His GitHub profile shows recent commits. [S2]

SOURCES:
[S1] Pranjal Chalise - LinkedIn — https://linkedin.com/in/pranjal-chalise
[S2] pranjalchalise - GitHub — https://github.com/pranjalchalise
[S3] Pranjal Chalise '25 - Amherst College — https://amherst.edu/people/pranjal

RESEARCH QUALITY:
- Overall Confidence: 74%
- Verified Claims: 9/10
- Cross-Validated: 3
- Sources Used: 15

KNOWLEDGE GAPS:
- **Career**: Limited work experience information

Write a comprehensive research report with confidence indicators.
"""

# Response:
report = """
# Research Report: Pranjal Chalise

## Executive Summary

Pranjal Chalise is a Computer Science student at Amherst College, Class of 2025. [✓✓] He has demonstrated skills in machine learning and software development through various open source projects. [✓]

## Background

Pranjal Chalise is currently pursuing a degree at Amherst College. [✓✓] He has developed expertise in machine learning [✓] and software development [✓], as evidenced by his professional profiles and project work.

## Education

He is part of the Class of 2025 at Amherst College, studying Computer Science. [✓✓] [S1][S3]

## Career & Projects

Pranjal has worked on multiple open source projects. [✓] He maintains 15 public repositories on GitHub [✓], showcasing his active development work. [S2]

*Note: Detailed internship or work experience information was limited in available sources.*

## Recent Activity

He is actively contributing to open source projects [✓], with his GitHub profile showing recent commits. [✓] [S2]

---

## Research Quality

| Metric | Value |
|--------|-------|
| Overall Confidence | 74% |
| Verified Claims | 9/10 |
| Cross-Validated Claims | 3 |
| High Confidence Claims | 3 |
| Sources Used | 15 |

### Knowledge Gaps
- **Career**: Limited information about internships or work experience

---

## Sources

[S1] Pranjal Chalise - LinkedIn — https://linkedin.com/in/pranjal-chalise
[S2] pranjalchalise - GitHub — https://github.com/pranjalchalise
[S3] Pranjal Chalise '25 - Amherst College — https://amherst.edu/people/pranjal
...
"""
```

**Operation 19.6: Build Research Metadata**
```python
research_metadata = {
    "overall_confidence": 0.74,
    "verified_claims": 9,
    "total_claims": 10,
    "knowledge_gaps": 1,
    "sources_used": 15,
    "research_iterations": 2,
    "total_searches": 18,
    "time_elapsed_seconds": 0  # Would be tracked externally
}
```

### Output State Update:
```python
{
    "report": "# Research Report: Pranjal Chalise\n\n## Executive Summary\n...",
    "messages": [AIMessage(content=report)],
    "research_metadata": {
        "overall_confidence": 0.74,
        "verified_claims": 9,
        "total_claims": 10,
        "knowledge_gaps": 1,
        "sources_used": 15,
        "research_iterations": 2,
        "total_searches": 18
    }
}
```

### Next Node: `END`

---

# COMPLETE FLOW SUMMARY

```
Step 1:  ANALYZER       → Parse query, identify type: "person"
Step 2:  DISCOVERY      → Search, find entity candidates, confidence: 0.85
Step 3:  CONFIDENCE     → Route to planner (high confidence)
Step 4:  PLANNER v8     → Create research tree with 6 questions
Step 5:  ORCHESTRATOR   → Assign 5 questions to subagents
Step 6:  SUBAGENTS (x5) → Parallel research, each searches + fetches + extracts
Step 7:  SYNTHESIZER    → Combine findings, avg confidence: 0.73
Step 8:  GAP DETECTOR   → Find gaps in Career/Achievements, route: continue
Step 9:  BACKTRACK      → Handle 1 dead end, generate alternatives
Step 10: ORCHESTRATOR   → Iteration 2 with refinement queries
Step 6b: SUBAGENTS      → More parallel research
Step 7b: SYNTHESIZER    → Updated confidence: 0.78
Step 8b: GAP DETECTOR   → Sufficient coverage, route: reduce
Step 11: REDUCE         → Dedupe sources, assign IDs: 28 sources, 45 evidence
Step 12: CREDIBILITY    → E-E-A-T scoring, all pass threshold
Step 13: RANKER         → Filter wrong entities, top 15 sources
Step 14: CLAIMS         → Generate 10 atomic claims
Step 15: CITE           → Attach evidence to claims
Step 16: SPAN VERIFY    → 8/10 verified, 1 unverified
Step 17: CROSS VALIDATE → 3 claims cross-validated
Step 18: CONFIDENCE     → Per-claim scores, overall: 0.74
Step 19: WRITER v8      → Generate report with ✓✓/✓/⚠ indicators

FINAL OUTPUT:
- Report with confidence indicators
- Research metadata
- 74% overall confidence
- 9 verified claims
- 3 cross-validated
- 2 research iterations
```
