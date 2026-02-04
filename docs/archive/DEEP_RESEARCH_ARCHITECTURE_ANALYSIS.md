# Deep Research Agent Architecture: Comprehensive Analysis & Recommendation

## Executive Summary

This document analyzes how OpenAI, Anthropic, Google, Perplexity, xAI (Grok), and LangChain build their deep research agents. Based on this analysis, I propose an optimal architecture for your research-studio that addresses all 5 requirements:

1. **Semantic query understanding** (not pattern-based)
2. **Multi-agent orchestration**
3. **Human-in-the-loop clarification**
4. **Grounded results with citations**
5. **Knowledge gap expansion**

---

## Part 1: Detailed Analysis of Each System

### 1.1 OpenAI Deep Research (o3-based)

**Source**: [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/), [System Card](https://cdn.openai.com/deep-research-system-card.pdf), [PromptLayer Analysis](https://blog.promptlayer.com/how-deep-research-works/)

#### Architecture: Plan-Act-Observe (ReAct) Loop

```
┌─────────────────────────────────────────────────────────┐
│                    o3 REASONING MODEL                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     Extended Chain-of-Thought (100s of steps)   │   │
│  │                                                 │   │
│  │   Plan → Act → Observe → Plan → Act → ...      │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│           ┌─────────────┼─────────────┐                │
│           ▼             ▼             ▼                │
│      ┌────────┐   ┌────────┐   ┌────────┐             │
│      │ Search │   │ Browse │   │ Python │             │
│      │  Tool  │   │  Tool  │   │  Tool  │             │
│      └────────┘   └────────┘   └────────┘             │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How OpenAI Does It |
|-----------|-------------------|
| **Query Understanding** | End-to-end RL training - model learned to understand queries through millions of research tasks, NO hardcoded patterns |
| **Planning** | Implicit in chain-of-thought - model decides search strategy dynamically |
| **Search Strategy** | Iterative refinement - "discovers information in the same iterative manner as a human researcher" |
| **Backtracking** | RL-trained - model learned to "backtrack when paths are unfruitful, and pivot strategies" |
| **Tool Use** | Single unified model with multi-tool access (search, browse, Python, images) |
| **Stopping** | Two-tier: Coverage-based (2+ sources per sub-question, novelty exhausted) + Hard limits (20-30 min, 30-60 queries) |
| **Citations** | "Every factual claim is accompanied by an inline citation—a clickable reference pointing to the exact source" |

#### What Makes It Work
- **End-to-end RL training**: The o3 model was trained on "complex browsing and reasoning tasks" - it learned research strategies from DATA, not rules
- **Extended reasoning**: Can maintain "hundreds of steps" of reasoning without diverging
- **No hardcoded orchestration**: The model itself decides when to search, what to search, when to pivot

#### Limitations
- Proprietary, closed-source
- ~$2-3 per query (expensive)
- 20-30 minute latency
- No human-in-the-loop mid-research

---

### 1.2 Anthropic Multi-Agent Research System

**Source**: [Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system), [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents), [Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

#### Architecture: Orchestrator-Workers Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    LEAD AGENT (Opus 4)                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • Analyzes query & develops strategy           │   │
│  │  • Decomposes into subtasks                     │   │
│  │  • Spawns subagents with detailed instructions  │   │
│  │  • Synthesizes findings                         │   │
│  │  • Decides if more research needed              │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│           ┌─────────────┼─────────────┐                │
│           ▼             ▼             ▼                │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│    │ Subagent │  │ Subagent │  │ Subagent │ (PARALLEL)│
│    │ (Sonnet) │  │ (Sonnet) │  │ (Sonnet) │           │
│    │          │  │          │  │          │           │
│    │ Search   │  │ Search   │  │ Search   │           │
│    │ → Read   │  │ → Read   │  │ → Read   │           │
│    │ → Think  │  │ → Think  │  │ → Think  │           │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│         │             │             │                  │
│         └─────────────┼─────────────┘                  │
│                       ▼                                │
│              ┌─────────────────┐                       │
│              │ Citation Agent  │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How Anthropic Does It |
|-----------|----------------------|
| **Query Understanding** | Lead agent "thinks through approach" using extended thinking - LLM-driven, not rules |
| **Task Decomposition** | Lead agent dynamically determines subtasks based on query complexity |
| **Parallelization** | 3-5 subagents work simultaneously, each with independent context |
| **Context Management** | Lead saves plan to external memory; subagents return compressed findings via "artifact systems" |
| **Error Recovery** | "Resume from where the agent was when errors occurred" - not restart |
| **Citations** | Dedicated CitationAgent "identifies specific citation locations for proper attribution" |
| **Model Tiering** | Opus 4 (lead) + Sonnet 4 (workers) = 90.2% improvement over single-agent Opus |

#### Key Principles
1. **Maintain simplicity** - "Start with simple prompts, optimize, add agentic systems only when simpler solutions fall short"
2. **Prioritize transparency** - Show planning steps explicitly
3. **Careful tool documentation** - "Invest equivalent effort in agent-computer interfaces as human-computer interfaces"

#### Token Economics
- Single agents use **4x more tokens** than chat
- Multi-agent systems use **15x more tokens** than chat
- "Token usage explains 80% of variance in performance"

---

### 1.3 Google Gemini Deep Research

**Source**: [Gemini Deep Research API](https://ai.google.dev/gemini-api/docs/deep-research), [Google Blog](https://blog.google/technology/developers/deep-research-agent-gemini-api/)

#### Architecture: Asynchronous Task Manager

```
┌─────────────────────────────────────────────────────────┐
│              GEMINI 3 PRO (Reasoning Core)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │   Plan → Search → Read → Iterate → Output       │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│              ┌──────────┴──────────┐                   │
│              ▼                     ▼                   │
│    ┌──────────────────┐  ┌──────────────────┐         │
│    │  Asynchronous    │  │   Shared State   │         │
│    │  Task Manager    │◄─►│   (Recovery)     │         │
│    └──────────────────┘  └──────────────────┘         │
│              │                                         │
│    ┌─────────┼─────────┬─────────────┐                │
│    ▼         ▼         ▼             ▼                │
│ ┌──────┐ ┌──────┐ ┌──────┐    ┌───────────┐          │
│ │google│ │ url  │ │ file │    │ 1M token  │          │
│ │search│ │contxt│ │search│    │  context  │          │
│ └──────┘ └──────┘ └──────┘    │  + RAG    │          │
│                               └───────────┘          │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How Google Does It |
|-----------|-------------------|
| **Query Understanding** | Gemini 3 Pro with "multi-step RL for search" - learned iterative planning |
| **Planning** | Explicit research plan shown to user, can be modified |
| **Execution** | Asynchronous - task runs in background, polls for completion |
| **Error Recovery** | "Shared state between planner and task models" - graceful recovery without restart |
| **Context** | 1M token window + RAG for documents exceeding context |
| **Multimodal** | Can process images, PDFs, audio, video as input |
| **Follow-up** | `previous_interaction_id` enables conversation continuation |

#### Unique Features
- **Benchmark-optimized**: 46.4% on Humanity's Last Exam, 66.1% on DeepSearchQA
- **Transparent planning**: Shows research plan before execution
- **Async API**: Long-running tasks don't block

---

### 1.4 Perplexity AI

**Source**: [ByteByteGo Analysis](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google), [Vespa Case Study](https://vespa.ai/perplexity/), [Perplexity Research](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)

#### Architecture: Multi-Stage RAG Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                 PERPLEXITY ARCHITECTURE                 │
│                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌────────┐ │
│  │ Query   │──►│  Live   │──►│ Snippet │──►│Synthesis│ │
│  │ Intent  │   │  Web    │   │Extract  │   │+ Cite  │ │
│  │ Parsing │   │Retrieval│   │         │   │        │ │
│  └─────────┘   └─────────┘   └─────────┘   └────────┘ │
│       │                                         │      │
│       └──────── Conversational Context ─────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              VESPA AI (Retrieval)               │   │
│  │  • 200B unique URLs indexed                     │   │
│  │  • Hybrid: Dense + Sparse + Metadata fusion     │   │
│  │  • Chunk-level retrieval (not full docs)        │   │
│  │  • Multi-stage ranking pipeline                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           MULTI-MODEL ORCHESTRATION             │   │
│  │  • Sonar (in-house) for efficiency              │   │
│  │  • GPT-4, Claude for complexity                 │   │
│  │  • Classifier routes to smallest capable model  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How Perplexity Does It |
|-----------|----------------------|
| **Query Understanding** | LLM-based semantic parsing, NOT keyword matching |
| **Retrieval** | Vespa AI - hybrid search (dense + sparse + metadata), chunk-level |
| **Model Selection** | Classifier routes to "smallest model that gives best UX" |
| **Citations** | Strict rule: "You are not supposed to say anything you didn't retrieve" |
| **Deep Research** | 20-50 targeted queries, 200+ sources, chain-of-thought reasoning |
| **Scale** | 780M monthly queries, 22M active users |

#### Citation Philosophy
Perplexity's core principle: **Every statement must link to a retrieved source**. This is enforced at the system level, not just prompted.

---

### 1.5 xAI Grok DeepSearch / DeeperSearch

**Source**: [xAI Grok 3 Announcement](https://x.ai/news/grok-3), [TryProfound Guide](https://www.tryprofound.com/blog/understanding-grok-a-comprehensive-guide-to-grok-websearch-grok-deepsearch)

#### Architecture: ReAct with Real-Time Data

```
┌─────────────────────────────────────────────────────────┐
│                    GROK 3 + DEEPSEARCH                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │      Chain-of-Thought + ReAct Framework         │   │
│  │  • Generates sub-queries from main question     │   │
│  │  • Reasons about conflicting facts              │   │
│  │  • Distills clarity from complexity             │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│           ┌─────────────┼─────────────┐                │
│           ▼             ▼             ▼                │
│      ┌────────┐   ┌────────┐   ┌────────┐             │
│      │  Web   │   │X (Twitter)│ │Reasoning│            │
│      │ Search │   │  Search   │ │ Engine │             │
│      └────────┘   └────────┘   └────────┘             │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              COLOSSUS SUPERCLUSTER              │   │
│  │  • 100,000+ NVIDIA H100 GPUs                    │   │
│  │  • 10x compute of previous SOTA                 │   │
│  │  • Large-scale RL for reasoning                 │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How xAI Does It |
|-----------|----------------|
| **Query Understanding** | Chain-of-thought reasoning, RL-trained |
| **Data Sources** | Web + X (Twitter) for real-time/trending content |
| **DeeperSearch** | Extended search + reasoning + "presets" for customization |
| **Personalization** | Adapts to user over time, learns from query history |
| **Speed vs Depth** | DeepSearch = faster; DeeperSearch = more thorough |

---

### 1.6 LangChain Open Deep Research

**Source**: [GitHub Repository](https://github.com/langchain-ai/open_deep_research)

#### Architecture: LangGraph State Machine

```
┌─────────────────────────────────────────────────────────┐
│            LANGGRAPH WORKFLOW (Configurable)           │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Research Agent (gpt-4.1) - Main reasoning loop  │   │
│  └─────────────────────────────────────────────────┘   │
│              │                                         │
│              ▼                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Search (Tavily / Native / MCP)          │   │
│  └─────────────────────────────────────────────────┘   │
│              │                                         │
│              ▼                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │     Summarization Model (gpt-4.1-mini)          │   │
│  └─────────────────────────────────────────────────┘   │
│              │                                         │
│              ▼                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │      Compression Module (consolidation)         │   │
│  └─────────────────────────────────────────────────┘   │
│              │                                         │
│              ▼                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Report Generation (gpt-4.1)             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  LEGACY OPTIONS:                                       │
│  • Plan-and-execute with human feedback                │
│  • Supervisor-researcher multi-agent (parallel)        │
└─────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

| Component | How LangChain Does It |
|-----------|----------------------|
| **Query Understanding** | Delegated to research agent (LLM-driven) |
| **Search** | Pluggable: Tavily, Anthropic native, OpenAI native, MCP |
| **Model Tiering** | 4 distinct models for different tasks |
| **Multi-Agent** | Legacy option with supervisor + parallel researchers |
| **Human Feedback** | Plan-and-execute variant supports HITL |
| **Benchmarks** | #6 on Deep Research Bench (RACE score 0.4344) |

---

## Part 2: Comparative Analysis

### 2.1 Query Understanding Approaches

| System | Approach | Pros | Cons |
|--------|----------|------|------|
| **OpenAI** | End-to-end RL | Generalizes to any query | Requires massive training data, expensive |
| **Anthropic** | LLM extended thinking | Transparent reasoning, flexible | Token-heavy |
| **Google** | Multi-step RL + plan | Shows plan to user, modifiable | Async adds complexity |
| **Perplexity** | Semantic LLM parsing | Fast, cost-efficient routing | Less deep reasoning |
| **Your Current** | Regex patterns | Fast, predictable | **FAILS on edge cases** |

**Recommendation**: Replace regex patterns with single LLM call that outputs structured JSON (intent, entity type, topic focus, temporal scope). This is what all major players do.

### 2.2 Multi-Agent Patterns

| System | Pattern | # Agents | Coordination |
|--------|---------|----------|--------------|
| **OpenAI** | Single agent, multi-tool | 1 | Implicit (RL-learned) |
| **Anthropic** | Orchestrator-workers | 1 lead + 3-5 workers | Explicit (lead controls) |
| **Google** | Single agent, async | 1 | Task manager |
| **Perplexity** | Pipeline (not multi-agent) | N/A | Stage-based |
| **LangChain** | Both options | Configurable | LangGraph state |

**Recommendation**: Anthropic's orchestrator-workers pattern is optimal for your needs:
- Parallel execution cuts research time 90%
- Lead agent maintains coherent strategy
- Subagents have clean contexts (no pollution)
- Model tiering (expensive lead, cheap workers) balances cost/quality

### 2.3 Human-in-the-Loop Patterns

| System | HITL Approach |
|--------|--------------|
| **OpenAI** | None mid-research (only initial query) |
| **Anthropic** | Checkpoints for "complex decision trees" |
| **Google** | Shows plan before execution, allows modification |
| **LangChain** | LangGraph interrupt nodes |
| **Your Current** | Clarification for ambiguous entities |

**Recommendation**: Google's approach + Anthropic's checkpoints:
1. Show research plan BEFORE execution
2. Allow user to modify/approve plan
3. Interrupt at ambiguity points (entities, conflicting sources)
4. Optional: progress updates during long research

### 2.4 Citation & Grounding

| System | Approach | Verification Level |
|--------|----------|-------------------|
| **OpenAI** | Inline citations to exact source lines | Claim → Source line |
| **Anthropic** | Dedicated CitationAgent | Post-hoc attribution |
| **Perplexity** | "Don't say anything not retrieved" | System-enforced |
| **Your Current** | Span verification + cross-validation | Claim → Evidence span |

**Recommendation**: Combine approaches:
1. **Perplexity's rule**: Never generate content without retrieved evidence
2. **Your span verification**: Match claims to exact text spans
3. **Cross-validation**: 2+ sources for high confidence
4. **Confidence indicators**: ✓✓ / ✓ / ⚠ in output

### 2.5 Knowledge Gap Expansion

| System | Gap Detection | Iteration Strategy |
|--------|--------------|-------------------|
| **OpenAI** | RL-learned (implicit) | Continuous until stopping criteria |
| **Anthropic** | Lead agent assesses coverage | Request more subagent research |
| **Google** | Multi-step planning | Iterates until plan complete |
| **Your Current** | Section-level confidence | Generate refinement queries |

**Recommendation**: Your current approach is good, enhance with:
1. **Explicit gap categorization**: Missing topics, conflicting info, shallow coverage
2. **Prioritized refinement**: Weight gaps by importance to user's question
3. **Hard limits**: Max iterations to prevent infinite loops
4. **Diminishing returns check**: Stop if improvement < 5%

---

## Part 3: Recommended Architecture

Based on this analysis, here's the optimal architecture for your research-studio:

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RESEARCH-STUDIO V9 ARCHITECTURE                 │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                 PHASE 1: SEMANTIC UNDERSTANDING               │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │              QUERY ANALYZER (LLM-Driven)                │ │ │
│  │  │                                                         │ │ │
│  │  │  Input: Raw user query                                  │ │ │
│  │  │  Output: {                                              │ │ │
│  │  │    intent: "Understand Trump's immigration policies",   │ │ │
│  │  │    query_class: "current_events" | "person_profile" |   │ │ │
│  │  │                 "concept" | "comparison" | "analysis",  │ │ │
│  │  │    primary_subject: "Trump administration",             │ │ │
│  │  │    topic_focus: "immigration policies",                 │ │ │
│  │  │    temporal_scope: "recent",                            │ │ │
│  │  │    complexity: "complex",                               │ │ │
│  │  │    suggested_approach: "news + policy docs + analysis"  │ │ │
│  │  │  }                                                      │ │ │
│  │  │                                                         │ │ │
│  │  │  NO REGEX PATTERNS - Pure LLM understanding             │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   PHASE 2: PLANNING (HITL)                    │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                RESEARCH PLANNER (LLM)                   │ │ │
│  │  │                                                         │ │ │
│  │  │  Generates research plan based on query analysis:       │ │ │
│  │  │  {                                                      │ │ │
│  │  │    research_questions: [                                │ │ │
│  │  │      "What are Trump's key immigration policy changes?",│ │ │
│  │  │      "What executive orders were signed?",              │ │ │
│  │  │      "What is the expert analysis?",                    │ │ │
│  │  │      ...                                                │ │ │
│  │  │    ],                                                   │ │ │
│  │  │    outline: ["Executive Summary", "Key Policies", ...], │ │ │
│  │  │    estimated_depth: "comprehensive"                     │ │ │
│  │  │  }                                                      │ │ │
│  │  │                                                         │ │ │
│  │  │  ════════════════════════════════════════════════════   │ │ │
│  │  │  HUMAN CHECKPOINT: "Here's my research plan. Proceed?"  │ │ │
│  │  │  User can: Approve / Modify / Add questions / Cancel    │ │ │
│  │  │  ════════════════════════════════════════════════════   │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              PHASE 3: MULTI-AGENT RESEARCH                    │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │            LEAD AGENT (gpt-4o / Claude Opus)            │ │ │
│  │  │                                                         │ │ │
│  │  │  • Reviews plan and distributes work                    │ │ │
│  │  │  • Monitors subagent progress                           │ │ │
│  │  │  • Synthesizes findings                                 │ │ │
│  │  │  • Detects gaps and requests more research              │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                         │                                     │ │
│  │         ┌───────────────┼───────────────┐                    │ │
│  │         ▼               ▼               ▼                    │ │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐              │ │
│  │  │ Subagent  │   │ Subagent  │   │ Subagent  │  (PARALLEL)  │ │
│  │  │ (gpt-4o-  │   │ (gpt-4o-  │   │ (gpt-4o-  │              │ │
│  │  │  mini)    │   │  mini)    │   │  mini)    │              │ │
│  │  │           │   │           │   │           │              │ │
│  │  │ Question: │   │ Question: │   │ Question: │              │ │
│  │  │ "Policy   │   │ "Executive│   │ "Expert   │              │ │
│  │  │  changes" │   │  orders"  │   │ analysis" │              │ │
│  │  │           │   │           │   │           │              │ │
│  │  │ [Search]  │   │ [Search]  │   │ [Search]  │              │ │
│  │  │ [Read]    │   │ [Read]    │   │ [Read]    │              │ │
│  │  │ [Extract] │   │ [Extract] │   │ [Extract] │              │ │
│  │  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘              │ │
│  │        │               │               │                     │ │
│  │        └───────────────┼───────────────┘                     │ │
│  │                        ▼                                     │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │              FINDINGS AGGREGATOR                        │ │ │
│  │  │  • Deduplicate sources                                  │ │ │
│  │  │  • Normalize evidence                                   │ │ │
│  │  │  • Calculate coverage per section                       │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               PHASE 4: GAP DETECTION & ITERATION              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                 GAP ANALYZER (LLM)                      │ │ │
│  │  │                                                         │ │ │
│  │  │  Analyzes current findings vs. research plan:           │ │ │
│  │  │  {                                                      │ │ │
│  │  │    overall_confidence: 0.65,                            │ │ │
│  │  │    gaps: [                                              │ │ │
│  │  │      {section: "Legal Challenges", severity: "high"},   │ │ │
│  │  │      {section: "Expert Analysis", severity: "medium"}   │ │ │
│  │  │    ],                                                   │ │ │
│  │  │    conflicts: [...],                                    │ │ │
│  │  │    recommendation: "continue" | "sufficient"            │ │ │
│  │  │  }                                                      │ │ │
│  │  │                                                         │ │ │
│  │  │  IF gaps AND iteration < max:                           │ │ │
│  │  │    → Generate refinement queries                        │ │ │
│  │  │    → Loop back to Phase 3                               │ │ │
│  │  │  ELSE:                                                  │ │ │
│  │  │    → Proceed to Phase 5                                 │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                PHASE 5: TRUST & VERIFICATION                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ CREDIBILITY   │ CLAIMS      │ SPAN        │ CROSS-     │ │ │
│  │  │ SCORER        │ EXTRACTOR   │ VERIFIER    │ VALIDATOR  │ │ │
│  │  │               │             │             │            │ │ │
│  │  │ Domain trust  │ Extract     │ Match claim │ Check if   │ │ │
│  │  │ + LLM quality │ 10-15 atomic│ to exact    │ 2+ sources │ │ │
│  │  │ assessment    │ claims      │ text span   │ agree      │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  BATCHED: These 4 steps run in 2 LLM calls (not 4)           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  PHASE 6: REPORT GENERATION                   │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                   WRITER (gpt-4o)                       │ │ │
│  │  │                                                         │ │ │
│  │  │  Rules:                                                 │ │ │
│  │  │  1. ONLY use verified claims                            │ │ │
│  │  │  2. Include confidence indicators (✓✓/✓/⚠)              │ │ │
│  │  │  3. Inline citations [S1][S2]                           │ │ │
│  │  │  4. Acknowledge gaps honestly                           │ │ │
│  │  │  5. Include "Research Quality" metadata section         │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Changes from Current Implementation

| Current (v8) | Proposed (v9) | Rationale |
|--------------|---------------|-----------|
| Regex pattern matching for query type | Single LLM call with structured output | Fixes misclassification bug, generalizes to any query |
| Hardcoded query templates | LLM generates queries based on intent | No more "Trump LinkedIn" for policy research |
| Fixed outline per type | LLM generates outline based on actual topic | Relevant structure for each query |
| Keyword complexity routing | LLM assesses complexity | More accurate routing |
| Confidence boost for concepts | Remove - let LLM assess naturally | Removes wrong assumptions |
| Pattern → LLM fallback | Always LLM, patterns optional for speed boost | Consistent quality |

### 3.3 The Core Change: Query Analyzer

Replace `pattern_preprocess()` and `_run_llm_analysis()` with a single unified prompt:

```python
QUERY_ANALYZER_SYSTEM = """You are an expert research query analyzer.

Analyze the user's query and determine the optimal research approach.

Output JSON:
{
  "intent": "What the user actually wants to learn (be specific)",
  "query_class": "person_profile | person_topic | concept | current_events | comparison | analysis | technical | general",
  "primary_subject": "The main entity/topic being researched",
  "topic_focus": "Specific aspect or angle (if any)",
  "temporal_scope": "recent | historical | timeless",
  "complexity": "simple | medium | complex",
  "ambiguity_level": "none | low | high",
  "suggested_source_types": ["news", "academic", "official", "analysis", "community"],
  "suggested_questions": ["Question 1", "Question 2", ...]
}

IMPORTANT DISTINCTIONS:
- "person_profile": Learning about who someone IS (bio, background)
- "person_topic": Learning about someone's WORK/OPINIONS on a topic
- "current_events": Recent news, policy changes, developments
- "analysis": Deep dive requiring multiple perspectives

Examples:

Query: "Deep research about Trump's new immigrant related policies"
→ {
    "intent": "Understand recent immigration policy changes under Trump administration",
    "query_class": "current_events",
    "primary_subject": "Trump administration immigration policy",
    "topic_focus": "recent policy changes and their implications",
    "temporal_scope": "recent",
    "complexity": "complex",
    "ambiguity_level": "none",
    "suggested_source_types": ["news", "official", "analysis"],
    "suggested_questions": [
      "What immigration-related executive orders has Trump signed?",
      "What are the key policy changes compared to previous administration?",
      "What do legal experts say about these policies?",
      "What is the humanitarian impact?"
    ]
  }

Query: "Tell me about Elon Musk"
→ {
    "intent": "Learn about Elon Musk's background, companies, and achievements",
    "query_class": "person_profile",
    "primary_subject": "Elon Musk",
    "topic_focus": null,
    "temporal_scope": "timeless",
    "complexity": "medium",
    ...
  }

Query: "What is Elon Musk's stance on AI safety?"
→ {
    "intent": "Understand Musk's opinions and actions regarding AI safety",
    "query_class": "person_topic",
    "primary_subject": "Elon Musk",
    "topic_focus": "AI safety views and advocacy",
    "temporal_scope": "recent",
    ...
  }
"""
```

### 3.4 LLM-Driven Query Generation

Replace template functions with:

```python
QUERY_GENERATOR_SYSTEM = """Generate optimal search queries for the research task.

Based on the query analysis, generate 6-10 search queries that will find the most relevant information.

Guidelines:
1. Be SPECIFIC to the actual topic (not generic templates)
2. Include temporal qualifiers for recent events (2024, 2025)
3. Mix precision levels:
   - 2-3 HIGH PRECISION: Exact phrases, specific sources
   - 2-3 MEDIUM: Key terms + context
   - 1-2 BROAD: Fallback for coverage
4. Consider what sources would have this information

Output JSON array:
[
  {
    "query": "search query text",
    "purpose": "what this query aims to find",
    "expected_sources": ["news", "official", "academic"],
    "priority": "high | medium | low"
  }
]
"""
```

---

## Part 4: Implementation Roadmap

### Phase 1: Core Fix (Immediate)
1. Replace `pattern_preprocess()` with LLM-based query analyzer
2. Replace template query generators with LLM-based generation
3. Remove hardcoded confidence boosts

### Phase 2: Enhanced Planning (Week 1)
1. Add HITL checkpoint after planning
2. Show research plan to user
3. Allow plan modification

### Phase 3: Improved Multi-Agent (Week 2)
1. Upgrade lead agent to gpt-4o (critical decisions)
2. Keep subagents on gpt-4o-mini (cost efficiency)
3. Add better progress tracking

### Phase 4: Better Gap Detection (Week 3)
1. Categorize gap types (missing, shallow, conflicting)
2. Prioritize by importance to user's question
3. Implement diminishing returns check

### Phase 5: Polish (Week 4)
1. Improve report formatting
2. Add confidence visualization
3. Optimize for latency

---

## Part 5: Expected Outcomes

| Metric | Current (v8) | Expected (v9) |
|--------|--------------|---------------|
| Query misclassification | ~10-15% | <2% |
| Irrelevant searches | ~20% | <5% |
| User satisfaction | N/A | Target: 85%+ |
| Cost per query | ~$0.15 | ~$0.12 |
| Research time | ~3-5 min | ~2-4 min |

---

## References

### OpenAI
- [Introducing Deep Research](https://openai.com/index/introducing-deep-research/)
- [o3-deep-research Model](https://platform.openai.com/docs/models/o3-deep-research)
- [How Deep Research Works (PromptLayer)](https://blog.promptlayer.com/how-deep-research-works/)

### Anthropic
- [Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Long-Running Agent Harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

### Google
- [Gemini Deep Research API](https://ai.google.dev/gemini-api/docs/deep-research)
- [Build with Deep Research](https://blog.google/technology/developers/deep-research-agent-gemini-api/)

### Perplexity
- [How Perplexity Built an AI Google (ByteByteGo)](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google)
- [Vespa AI Case Study](https://vespa.ai/perplexity/)

### xAI
- [Grok 3 Announcement](https://x.ai/news/grok-3)
- [DeepSearch Guide](https://www.tryprofound.com/blog/understanding-grok-a-comprehensive-guide-to-grok-websearch-grok-deepsearch)

### LangChain
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)
