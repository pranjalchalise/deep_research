# Design Document

## What is this?

Research Studio is a deep research agent built on LangGraph. You give it a question, it searches the web, reads pages, extracts facts, checks them, and writes a grounded report with citations. There are two pipelines:

- **Standard pipeline** (`src/pipeline/`) — the clean, self-contained version. Supports single-agent (iterative search loop) and multi-agent (parallel workers via LangGraph Send). About 1,200 lines.
- **Advanced pipeline** (`src/advanced/`) — same core flow but adds a trust engine on top: source credibility scoring, claim extraction, span verification, cross-validation, and per-claim confidence indicators. About 4,000 lines across 13+ node modules.

Both are registered in `langgraph.json` and work in LangGraph Studio.

```json
{
  "graphs": {
    "pipeline": "./src/pipeline/graph.py:graph",
    "advanced": "./src/advanced/graph.py:graph"
  }
}
```

This means you can open LangGraph Studio, pick either graph, and interact with it visually and see the state at each node, inspect edges, and use the config knobs. More on that later.

---

## Why two pipelines?

I started with one big pipeline (the advanced one) and kept adding features — credibility scoring, span verification, cross-validation, batched trust engine calls. It worked, but it was hard to reason about. When something went wrong, I'd spend more time debugging the trust engine than the actual research.

So I pulled the core research flow into its own clean pipeline (`src/pipeline/`). No trust engine, no credibility scoring, just: understand the query, plan, search, extract, detect gaps, verify, write. It's easier to demo, easier to test, and honestly covers 80% of what we need.

The advanced pipeline is there to show what's possible when you layer trust and verification on top. It's not always better - it's slower, costs more API calls, and sometimes the extra verification steps don't add much for straightforward factual queries. But for anything where source quality matters (medical, legal, current events), it catches things the standard pipeline doesn't.

---

## How I got here

### Industry research

Before writing any code, I spent time studying how the major players do deep research. I wanted to understand what patterns were universal versus what was specific to each company's stack. The full analysis lives in `docs/archive/DEEP_RESEARCH_AGENT_PLAN.md`, but here's the detailed breakdown.

### OpenAI Deep Research

| Component | Implementation |
|-----------|----------------|
| **Model** | o3 with expanded "attention span" for long chains of thought |
| **Training** | End-to-end RL on simulated research environments with tool access |
| **Loop** | Plan → Act → Observe (ReAct paradigm) with backtracking |
| **Tools** | Web browser, PDF parser, image analyzer, Python code execution |
| **Duration** | Up to 30 minutes, 21+ sources, dozens of search queries |
| **Summarization** | o3-mini used to summarize chains of thought |

The key insight: they trained o3 end-to-end on simulated research environments with tool access. That's not something we can replicate — we don't have the training infrastructure. But the backtracking and iterative refinement patterns are absolutely implementable as graph control flow. When a search path hits a paywall or irrelevant content, pivot and try something else. That's what our `backtrack_handler_node` does in the advanced pipeline, and what the `detect_gaps` loop does more simply in the standard pipeline.

**Performance**: 26.6% on Humanity's Last Exam (3,000 questions across 100 subjects).

### Anthropic's multi-agent research system

| Component | Implementation |
|-----------|----------------|
| **Lead Agent** | Claude Opus — analyzes query, develops strategy, spawns subagents |
| **Subagents** | Claude Sonnet — 3-5 parallel workers with own context windows |
| **Parallelism** | Two levels: subagent spawning + tool calls within subagents |
| **Search** | Brave Search integration via web_search tool |
| **Compression** | Subagents condense findings before returning to lead agent |

This one influenced our architecture the most. The orchestrator-worker pattern maps directly to LangGraph's `Send()` API- the orchestrator node decides how many workers to spawn and what each should focus on, then `Send()` dispatches them in parallel. Each worker gets its own slice of the state and runs independently. LangGraph handles the synchronization so that we don't need to manually wait or lock anything.

Their insight about programmatic tool calling reducing tokens by 37% was practical too. Rather than letting the LLM decide when to call tools via a ReAct loop, we hardcode the tool calls in the node functions. The LLM does the thinking (what to extract, how to summarize), but the search and fetch calls are deterministic.

**Performance**: 90.2% improvement over single-agent Claude Opus on internal evals. Multi-agent uses ~15x more tokens than single chat.

### xAI Grok DeepSearch

| Component | Implementation |
|-----------|----------------|
| **Model** | Grok 3 trained on 200K H100 GPUs |
| **Search** | Real-time web + X/Twitter signals |
| **Reasoning** | "Think" mode for slower, deliberate chain-of-thought |
| **Speed** | 67ms average response latency |
| **Context** | 128K token window |

The complexity routing idea came directly from xAI. Simple queries ("What is CRISPR?") don't need multi-agent overhead — you're wasting API calls spawning 3-5 workers for something one search can answer. In the advanced pipeline, `complexity_router_node` does a keyword + word-count heuristic to classify queries as simple/medium/complex. Simple queries skip straight to a single worker; complex ones get the full orchestrator-subagent treatment. It's a conditional edge in the graph — LangGraph makes this trivial.

### Google Gemini Deep Research

| Component | Implementation |
|-----------|----------------|
| **Model** | Gemini 3 Pro, trained to reduce hallucinations |
| **Context** | 1M token window + RAG for session memory |
| **Task Manager** | Asynchronous with shared state between planner and task models |
| **Sources** | Web + Gmail + Drive + Chat integrations |
| **Multimodal** | Images, PDFs, audio, video as research inputs |

The knowledge gap detection pattern - formulate queries, read content, identify what's missing, search again - is basically what our gap detection loop does. After each search round, the LLM compares what we've found to the research questions from the planning step, identifies gaps, and suggests new search queries. If coverage is below 70% and we haven't hit the iteration cap, we loop back. Google does this with async task management; we do it with LangGraph's conditional edges pointing back to earlier nodes.

**Performance**: 46.4% on Humanity's Last Exam, 66.1% on DeepSearchQA, 59.2% on BrowseComp.

### Perplexity

| Component | Implementation |
|-----------|----------------|
| **Routing** | Dynamic routing to different engines (conversational, research, coding) |
| **Source Ranking** | E-E-A-T scoring (Experience, Expertise, Authority, Trust) |
| **Comet Framework** | Retrieval Agent + Synthesis Agent + Verification Agent |
| **Scale** | 200M daily queries |

E-E-A-T scoring for source ranking came from Perplexity. Their approach of scoring sources on experience, expertise, authority, and trustworthiness is what our `credibility_scorer_node` implements — with a domain-based shortcut (`.edu`/`.gov` = high trust, `quora.com` = low) plus LLM-assessed content quality. The Comet framework (retrieval + synthesis + verification as separate agents) maps to our orchestrator + synthesizer + trust engine split.

### Cross-system comparison

| System | Query Strategy | Refinement | Parallelism |
|--------|---------------|------------|-------------|
| OpenAI | Implicit in CoT | Based on results read | Sequential with backtrack |
| Anthropic | Lead agent generates | Subagents refine independently | 3-5 subagents parallel |
| xAI | Sub-query decomposition | Think mode for depth | Dynamic workflows |
| Google | Research plan from prompt | Gap detection → new queries | Task-level parallel |
| Perplexity | Research tree branching | Multi-step in Pro Search | Comet multi-agent |

### Universal patterns I found

Every system does some version of the same thing:

1. **Break the query into sub-questions** — OpenAI does it implicitly in chain-of-thought, Anthropic has the lead agent generate them, we do it in the orchestrator node
2. **Search iteratively (not just once)** — Google's gap detection loop, OpenAI's backtracking, our `detect_gaps → search_and_extract` cycle
3. **Backtrack when a path is unproductive** — OpenAI pioneered this, our advanced pipeline records dead ends and tries different angles
4. **Score source credibility** — Perplexity's E-E-A-T, our trust engine
5. **Verify citations against actual evidence** — everyone does span-level or claim-level verification
6. **Use multiple models** — cheap models for extraction, expensive ones for reasoning and writing

I built both pipelines around these patterns. The standard pipeline covers 1-2-5-6. The advanced pipeline covers all six.

### What I would have done with more time

| Gap | Industry Practice | Current State | Priority |
|-----|------------------|---------------|----------|
| **Multimodal inputs** | PDF, images, video | Text only | Medium |
| **Code execution** | Python for data analysis | None | Medium |
| **Async operation** | Background + notify | Synchronous only | Low |
| **1M+ context** | Google's session memory | Standard context windows | Low |
| **End-to-end RL** | OpenAI's o3 training | Prompt engineering only | Not feasible |

---

## LangGraph features used (and why)

Before diving into the per-node details, I want to call out the specific LangGraph features I used and why. 

### 1. `StateGraph` with typed state

Both pipelines use `StateGraph` with a TypedDict state class. The state is the single source of truth where every node reads from it and writes back to it. 

```python
# Standard pipeline
g = StateGraph(ResearchState, context_schema=Configuration)

# Advanced pipeline
g = StateGraph(AgentState)
```

The `context_schema=Configuration` parameter on the standard pipeline is worth calling out. It registers a separate TypedDict as the configuration schema, which does two things: (1) it tells LangGraph Studio to render those fields as editable config knobs in the UI, and (2) it keeps configuration out of the state itself. Configuration lives in `RunnableConfig["configurable"]`, not in the graph state, so changing model or report format doesn't pollute the research state.

### 2. `MessagesState` inheritance

Both state classes extend `MessagesState`:

```python
class ResearchState(MessagesState, total=False):
    query: str
    mode: str
    evidence: Annotated[List[Dict], operator.add]
    # ... 20+ fields
```

This gives us the `messages` field with LangGraph's built-in `add_messages` reducer for free. The `messages` list tracks what's happening at each step — the original query comes in as a `HumanMessage`, each node can append status messages, and the final report goes out as an `AIMessage`. This is what makes the graph work seamlessly in LangGraph Studio and chat UIs. Without `MessagesState`, we'd have to manually manage message history.

The `total=False` matters too. It means nodes can return partial updates — you don't have to return every field on every node. If `understand_node` only sets `understanding`, `is_ambiguous`, and `clarification_question`, that's fine. Everything else stays as-is.

### 3. `Annotated[..., operator.add]` for parallel-safe accumulation

This is the feature that makes multi-agent work without race conditions:

```python
# Standard pipeline state
worker_results: Annotated[List[Dict], operator.add]
done_workers: Annotated[int, operator.add]
evidence: Annotated[List[Dict], operator.add]

# Advanced pipeline state
subagent_findings: Annotated[List[SubagentFindings], operator.add]
raw_sources: Annotated[List[RawSource], operator.add]
raw_evidence: Annotated[List[RawEvidence], operator.add]
```

When multiple workers run in parallel via `Send()`, they all write to these fields simultaneously. Without `operator.add`, the last worker to finish would overwrite everyone else's results. With it, LangGraph automatically concatenates the lists and sums the integers. Worker A returns `evidence: [fact1, fact2]`, Worker B returns `evidence: [fact3]`, and the state ends up with `evidence: [fact1, fact2, fact3]`. No locks, no manual merging, no bugs from forgetting to merge.

I use 3 accumulator fields in the standard pipeline and 5 in the advanced one. The `done_workers: Annotated[int, operator.add]` pattern is particularly nice — each worker returns `done_workers: 1`, and LangGraph sums them, so after all workers finish, `done_workers == total_workers`. The `reducer_node` in the advanced pipeline checks this to know when it's safe to proceed with deduplication.

### 4. `Send()` for dynamic parallel fan-out

This is the core of multi-agent mode:

```python
def fanout_workers(state: ResearchState) -> List[Send]:
    """Spawn one parallel worker per sub-question via Send()."""
    return [
        Send("search_worker", {
            "sub_question": sq,
            "query": state["query"],
            "user_clarification": state.get("user_clarification", ""),
        })
        for sq in state.get("sub_questions", [])
    ]
```

The orchestrator node decides how to split the query into 3-5 sub-questions and writes them to state. Then `fanout_workers` creates one `Send()` call per sub-question, each targeting the `search_worker` node with just the data that worker needs. LangGraph runs them all in parallel and waits for all of them to finish before moving to the next node (`collect`).

What makes `Send()` better than, say, spawning threads manually:
- Each worker gets an isolated slice of state — they can't interfere with each other
- The `operator.add` fields handle merging automatically
- LangGraph guarantees all workers complete before the downstream node runs
- The graph visualization shows the fan-out clearly in Studio

The advanced pipeline has two separate fan-out points — `fanout_workers` for the single-agent fallback path and `fanout_subagents` for the orchestrator-subagent path:

```python
# Advanced pipeline — orchestrator-subagent fan-out
def fanout_subagents(state: AgentState):
    assignments = state.get("subagent_assignments") or []
    return [
        Send("subagent", {"subagent_assignment": assignment})
        for assignment in assignments
    ]
```

### 5. `interrupt_before` for human-in-the-loop

When the query is ambiguous ("Python" — language or snake?), the pipeline needs to pause and ask the user for clarification. LangGraph's `interrupt_before` makes this clean:

```python
def build_graph(checkpointer=None, interrupt_on_clarify: bool = True):
    g = StateGraph(ResearchState, context_schema=Configuration)
    # ... add all nodes and edges ...

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]

    return g.compile(**compile_kwargs)
```

When the graph reaches the `clarify` node, it pauses. The caller can inspect the state (which now has `clarification_question` and `clarification_options` from the `understand_node`), show it to the user, get their answer, inject it back via `update_state()`, and resume:

```python
# In run.py — the HITL flow
result = graph.invoke(init, config)

snapshot = graph.get_state(config)
if snapshot.next and "clarify" in snapshot.next:
    # Graph paused! Show clarification to user, get their answer
    user_input = input("Your choice: ")
    graph.update_state(config, {"user_clarification": user_input})
    result = graph.invoke(None, config)  # resume from where we paused
```

This requires a checkpointer (`MemorySaver`) because the graph needs to persist its state across the pause. For automated runs (no HITL), we skip the checkpointer entirely:

```python
graph = build_graph(checkpointer=None, interrupt_on_clarify=False)
```

### 6. `MemorySaver` checkpointing

The checkpointer is what makes pause/resume work. Without it, `graph.invoke()` is fire-and-forget — there's no way to get back into a partially-completed run.

```python
if skip_clarification:
    graph = build_graph(checkpointer=None, interrupt_on_clarify=False)
else:
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer, interrupt_on_clarify=True)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
```

Each run gets a unique `thread_id` so the checkpointer can track multiple concurrent runs. The advanced pipeline provides convenience builders for this:

```python
def build_trust_engine_graph_with_memory(interrupt_on_clarify=True):
    checkpointer = MemorySaver()
    graph = build_trust_engine_graph(checkpointer=checkpointer, ...)
    return graph, checkpointer
```

### 7. Conditional edges for routing logic

Every decision point in the graph is a conditional edge with a routing function:

```python
g.add_conditional_edges("understand", route_after_understand,
                        {"clarify": "clarify", "plan": "plan"})

g.add_conditional_edges("plan", route_after_plan,
                        {"search_and_extract": "search_and_extract",
                         "orchestrate": "orchestrate"})

g.add_conditional_edges("detect_gaps", route_after_gaps,
                        {"search_and_extract": "search_and_extract",
                         "verify": "verify"})
```

The routing functions are pure functions of state — they read a few fields and return the next node name. This makes the control flow explicit and testable. For example, the gap detection loop decision:

```python
def route_after_gaps(state: ResearchState) -> str:
    if state.get("ready_to_write"):
        return "verify"
    if state.get("coverage", 0) >= state.get("min_coverage", 0.7):
        return "verify"
    if not state.get("pending_searches"):
        return "verify"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "verify"
    return "search_and_extract"  # loop back
```

Four exit conditions, all explicit. The advanced pipeline has more complex routing- like `route_by_complexity` which sends simple queries down a fast path and complex ones through the full orchestrator.

### 8. `RunnableConfig` for per-run configuration

Every node receives `config: RunnableConfig` as its second argument. The pipeline reads settings from `config["configurable"]`:

```python
def understand_node(state: ResearchState, config: RunnableConfig) -> Dict:
    cfg = get_configuration(config)
    llm = create_chat_model(cfg["model"])
    # ...
```

This means you can change the model, report format, or system prompt per invocation without rebuilding the graph:

```python
result = graph.invoke(init, config={
    "configurable": {
        "model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "report_structure": "bullet_points",
        "system_prompt": "Write for a technical audience",
    }
})
```

In LangGraph Studio, these show up as editable config knobs because of the `context_schema=Configuration` we set on the `StateGraph`. The advanced pipeline takes a different approach — it uses a frozen dataclass (`ResearchConfig`) with ~50 tunables, which gives better type safety but doesn't integrate as cleanly with Studio's auto-rendered knobs.

### 9. Graph composition patterns

The advanced pipeline provides 6 graph builder functions to handle different deployment scenarios:

| Builder | Checkpointer | HITL | Use case |
|---------|-------------|------|----------|
| `build_trust_engine_graph()` | Optional | Optional | Full control |
| `build_trust_engine_graph_with_memory()` | MemorySaver | Yes | Interactive CLI |
| `build_trust_engine_simple_graph()` | None | No | Automated pipelines |
| `build_optimized_graph()` | Optional | Optional | Full control, batched |
| `build_optimized_graph_with_memory()` | MemorySaver | Yes | Interactive CLI, batched |
| `build_optimized_simple_graph()` | None | No | Automated, batched |

This is more builders than strictly necessary, but it means the caller never has to think about which compile kwargs to pass. `_with_memory` = I want HITL. `_simple` = just run it. The module-level `graph` variable (exposed to `langgraph.json`) uses the trust engine variant with interrupt enabled.

---

## Standard pipeline

The standard pipeline has 11 nodes. Here's what each one does, why it exists, and how it uses the state.

### Flow overview

```
Single-agent:
  START → understand → [clarify] → plan → search_and_extract → detect_gaps
    ↻ (loop back if gaps remain)
  detect_gaps → verify → write_report → END

Multi-agent:
  START → understand → [clarify] → plan → orchestrate → [search_worker ×N] → collect
    → synthesize → (loop back to orchestrate if gaps remain)
  synthesize → verify → write_report → END
```

### 1. `understand_node`

**What it does**: Analyzes the user's query to determine if it's clear enough to research, or if we need to ask for clarification first.

**Reads**: `query`
**Writes**: `understanding`, `is_ambiguous`, `clarification_question`, `clarification_options`, `messages`

**How it works**: Sends the query to the LLM with a structured prompt that checks for seven types of ambiguity: polysemy ("Python" could be a language or a snake), scope (too broad), time period, geography, perspective, specificity, and comparison baseline. The LLM returns JSON with a clarity assessment. If `is_clear` is false, it also returns a clarification question and 3-5 options for the user.

**Why it exists**: Every major deep research system has some form of query understanding. Without it, you risk spending 30 seconds searching for "Python the snake" when the user meant the programming language. The clarification options are important — instead of asking an open-ended "what do you mean?", we give structured choices. This is better UX and makes the HITL flow cleaner.

**LangGraph feature**: This node's output drives the first conditional edge. `route_after_understand` checks `is_ambiguous` and routes to either `clarify` (HITL pause) or `plan` (proceed).

### 2. `clarify_node`

**What it does**: Passthrough node that makes the user's clarification visible in state.

**Reads**: `user_clarification`
**Writes**: `user_clarification`

**How it works**: Almost nothing happens in this node itself. The real work happens *before* it runs — the graph pauses via `interrupt_before=["clarify"]`, the caller shows the clarification question to the user, gets their answer, and injects it into state with `graph.update_state()`. When the graph resumes, this node just passes the clarification through so downstream nodes can read it.

**Why it exists**: It's a control point for HITL. Without a dedicated node to interrupt on, we'd have to interrupt inside `understand_node` (messy) or `plan_node` (too late — we'd lose the clarification context). Having a separate node makes the graph structure clear: understand → (maybe pause here) → plan.

**LangGraph feature**: `interrupt_before=["clarify"]` on the compiled graph. This plus `MemorySaver` checkpointing is what enables the full HITL flow. The `update_state()` API lets the caller inject new data without re-running any nodes.

### 3. `plan_node`

**What it does**: Creates a research plan — what questions to answer, what aspects to cover, and what to search for.

**Reads**: `query`, `user_clarification`
**Writes**: `research_questions`, `aspects_to_cover`, `pending_searches`, `iteration`, `evidence`, `worker_results`, `done_workers`, `sources`, `searches_done`, `messages`

**How it works**: Sends the query (plus clarification if we got one) to the LLM with `PLAN_PROMPT`. Gets back 3-5 research questions, aspects to cover, and 4-6 initial search queries. Also resets all the accumulators (`evidence: []`, `worker_results: []`, `sources: {}`, etc.) so the research loop starts clean.

**Why it exists**: Without a planning step, the pipeline would just search for the raw query. With planning, the LLM can decompose "Compare React vs Vue for a startup in 2024" into specific questions like "What are React's performance characteristics?", "What is Vue's learning curve?", "How do hiring markets compare?" — and generate targeted search queries for each.

**LangGraph feature**: The plan node's output drives the second conditional edge. `route_after_plan` checks `state["mode"]` and routes to either `search_and_extract` (single-agent) or `orchestrate` (multi-agent). This is where the graph splits into two paths using the same nodes.

### 4. `orchestrate_node` (multi-agent only)

**What it does**: Breaks the query into 3-5 independent sub-questions that can be researched in parallel.

**Reads**: `query`, `synthesis` (if looping back after gaps)
**Writes**: `sub_questions`, `total_workers`

**How it works**: Sends the query to the LLM with `ORCHESTRATE_PROMPT`, which asks for independent, non-overlapping sub-questions. On loop-back (when synthesis found gaps), it includes `synthesis.additional_searches` so the LLM can target remaining gaps instead of re-asking the same questions.

**Why it exists**: This is the Anthropic pattern — break the work into pieces, hand each piece to a dedicated worker, run them all in parallel. The orchestrator doesn't do any research itself; it just decides the division of labor.

**LangGraph feature**: The orchestrate node's output feeds directly into `fanout_workers`, which returns `List[Send]`. This is the `add_conditional_edges("orchestrate", fanout_workers)` call — LangGraph sees the list of `Send` objects and dispatches one `search_worker` instance per sub-question in parallel.

### 5. `search_worker_node` (multi-agent only)

**What it does**: Takes a single sub-question and does the full search-fetch-extract cycle for it.

**Reads**: `sub_question`, `query`
**Writes**: `worker_results` (via `operator.add`), `done_workers` (via `operator.add`), `evidence` (via `operator.add`)

**How it works**: Four phases:
1. **Search** — runs up to 3 search queries via Tavily (`cached_search()`)
2. **Fetch** — for each result (max 2 per search), fetches the page and chunks it (max 2 chunks, 3000 chars each)
3. **Extract** — invokes the fast LLM with `WORKER_EXTRACT_PROMPT` to pull facts (max 4 per page)
4. **Summarize** — invokes the smart LLM with `WORKER_SUMMARY_PROMPT` to create a cohesive summary with key findings, confidence, and gaps

Returns the worker summary, raw evidence, and `done_workers: 1`.

**Why it exists**: Each worker focuses on one sub-question, so it can be more targeted than a general search. The caps (3 searches, 2 pages per search, 4 facts per page) keep costs bounded — without them, a broad sub-question could trigger dozens of API calls.

**LangGraph feature**: All three output fields use `operator.add`, which is critical here. Multiple workers run simultaneously and write to the same fields. Without the reducer, the last worker to finish would overwrite all the others. With `operator.add`, evidence from all workers gets concatenated automatically.

### 6. `collect_node` (multi-agent only)

**What it does**: Deduplicates sources from all parallel workers into a single source dictionary.

**Reads**: `sources`, `worker_results`
**Writes**: `sources`

**How it works**: Workers might find the same URL independently. This node walks through all worker evidence, deduplicates by URL, and assigns unique source IDs. LangGraph guarantees this node only runs after *all* `Send()` workers have completed.

**Why it exists**: Without deduplication, the same source might appear 3 times with 3 different IDs, making citations confusing. This is the fan-in step — we've finished the parallel work, now we consolidate.

**LangGraph feature**: LangGraph's implicit fan-in guarantee. The `add_edge("search_worker", "collect")` edge means `collect` only fires once all parallel `search_worker` instances are done. We don't need to check `done_workers == total_workers` manually — the graph structure handles it.

### 7. `synthesize_node` (multi-agent only)

**What it does**: Combines findings from all workers into a unified view and decides if we need another round.

**Reads**: `query`, `worker_results`, `user_clarification`
**Writes**: `synthesis`, `iteration`, `coverage`, `ready_to_write`, `messages`

**How it works**: Formats all worker summaries (question, answer, findings, confidence, gaps) into a single prompt and asks the LLM to synthesize: combine findings, resolve conflicts, identify overall gaps, assess confidence. Sets `ready_to_write` based on whether `synthesis.needs_more_research` is true.

**Why it exists**: Individual workers might find overlapping or contradictory information. The synthesizer resolves conflicts ("Worker A says X but Worker B says Y") and provides a unified confidence score. It also decides whether to loop — if there are significant gaps and we have iteration budget left, we go back to the orchestrator for another round.

**LangGraph feature**: The `route_after_synthesis` conditional edge handles the loop decision. Three conditions exit to `verify` (no more research needed, iteration limit hit, no follow-up searches). Otherwise it routes back to `orchestrate`, creating a cycle in the graph. LangGraph supports cycles natively — you just point a conditional edge back to an earlier node.

### 8. `search_and_extract_node` (single-agent only)

**What it does**: The single-agent equivalent of `search_worker`, but runs sequentially through all pending searches instead of in parallel.

**Reads**: `query`, `pending_searches`, `sources`, `searches_done`, `research_questions`, `user_clarification`
**Writes**: `evidence` (via `operator.add`), `sources`, `searches_done`, `messages`

**How it works**: Iterates through `pending_searches`, runs each through Tavily, fetches pages, chunks content, and extracts facts using the fast LLM. Caps at 20 total sources, max 5 facts per page. Tracks `searches_done` to avoid running the same query twice.

**Why it exists**: Single-agent mode trades parallelism for thoroughness. Instead of splitting the query, it runs all searches sequentially so the extraction context includes the full query and all research questions. This means each extraction call has better context than a worker that only knows about its sub-question.

**LangGraph feature**: The `evidence` field still uses `operator.add` even in single-agent mode. This is defensive — if someone later adds parallel search within this node, the accumulator semantics will still work correctly.

### 9. `detect_gaps_node` (single-agent only)

**What it does**: Assesses research completeness and decides whether to loop back for more searching.

**Reads**: `query`, `evidence`, `research_questions`, `iteration`
**Writes**: `coverage`, `ready_to_write`, `pending_searches`, `gaps`, `iteration`, `messages`

**How it works**: Sends all collected evidence plus the original research questions to the LLM with `GAP_DETECTION_PROMPT`. The LLM assesses coverage per question (well_covered / partial / missing), gives an overall coverage score, identifies gaps, and suggests new search queries. Increments the iteration counter.

**Why it exists**: This is the core loop of single-agent mode, directly inspired by Google Gemini's iterative approach. Without it, we'd do one round of searching and hope for the best. With it, we can discover that "we have good coverage on React's performance but nothing about Vue's developer experience" and generate targeted follow-up searches.

**LangGraph feature**: `route_after_gaps` is the loop control. It checks four conditions (ready_to_write, coverage threshold, pending searches, iteration limit) and either routes back to `search_and_extract` or exits to `verify`. This creates a cycle: `search_and_extract → detect_gaps → search_and_extract → ...`. The iteration counter prevents infinite loops.

### 10. `verify_node`

**What it does**: Cross-checks claims against evidence to calculate a confidence score.

**Reads**: `evidence`
**Writes**: `verified_claims`, `confidence`, `messages`

**How it works**: Takes up to 20 evidence items, formats them as claims, and asks the LLM with `VERIFY_PROMPT` to check each claim against all available evidence. Returns a boolean `supported` flag per claim, plus a confidence score. The overall `confidence` is `supported_count / total_claims`.

**Why it exists**: Before writing the report, we want to know how trustworthy our evidence is. If only 40% of claims are supported by multiple sources, the report should acknowledge uncertainty. This also filters out any hallucinated "facts" that snuck in during extraction.

**LangGraph feature**: Both the single-agent and multi-agent paths converge here — `add_edge("verify", "write_report")`. This is the point where the graph topology reunifies after the mode split.

### 11. `write_report_node`

**What it does**: Generates the final markdown report with citations.

**Reads**: `query`, `evidence`, `sources`, `mode`, `worker_results`, `user_clarification`, `iteration`, `coverage`, `confidence`
**Writes**: `report`, `messages`, `metadata`

**How it works**: This is the most complex node. It:
1. Builds a `url_to_sid` mapping so evidence items reference source IDs (not sequential numbers)
2. Formats evidence with `[Source N]` labels and quotes
3. Selects the prompt based on mode — `WRITE_MULTI_PROMPT` includes worker summaries for thematic organization; `WRITE_PROMPT` is for single-agent
4. Injects the valid citation numbers into the system prompt so the LLM can only reference real sources
5. Applies `report_structure` config (detailed / concise / bullet_points)
6. Appends a sources section as fallback if the LLM forgot one
7. Records metadata (iterations, coverage, confidence, counts)

**Why it exists**: Everything before this node was about gathering and verifying information. This node turns it into a readable report. The citation mapping fix (using source IDs instead of sequential numbers) was a bug I found through the eval suite — the LLM was hallucinating citation numbers like `[19]` when there were only 12 sources.

**LangGraph feature**: Reads from `RunnableConfig` to get `report_structure` and `system_prompt` settings. This is the node where the `context_schema=Configuration` integration pays off — users can change the output format per run without touching any code.

---

## Advanced pipeline: node-by-node

The advanced pipeline has 20+ nodes across several modules. I'll cover the key ones grouped by phase.

### Graph variants

The advanced pipeline ships as 3 graph variants (each with `_with_memory` and `_simple` helpers):

| Variant | Nodes | Trust engine | Key difference |
|---------|-------|-------------|----------------|
| `build_trust_engine_graph` | 18 | Full (7 nodes) | Exhaustive verification |
| `build_optimized_graph` | 16 | Batched (2 nodes) | 60% fewer LLM calls |

### Phase 1: Discovery (3 nodes)

#### `analyzer_node`

**What it does**: Initial query analysis — intent classification, entity detection, temporal scope.

**Why it's different from standard**: The standard pipeline's `understand_node` just checks for ambiguity. The analyzer does full NLP-style query decomposition: what type of query is this (factual, comparative, exploratory)? What entities are mentioned? What time period is relevant? This powers downstream complexity routing.

#### `discovery_node`

**What it does**: Entity disambiguation and discovery search. If the query mentions "Apple," is it the company or the fruit? Runs a quick search to resolve entities and assess confidence.

**Why it exists**: The ERA-CoT (Entity Resolution through Augmented Chain-of-Thought) pattern. Most deep research systems don't do this explicitly — they rely on the search engine to disambiguate. But for ambiguous queries, doing a dedicated discovery search before the main research avoids wasting the entire first iteration on the wrong interpretation.

#### `confidence_check_node` / `complexity_router_node`

**What it does**: Gates the HITL flow. Only asks the user for clarification if discovery confidence is below 0.7. Also classifies query complexity for the optimized graph's routing.

```python
def route_complexity_to_confidence(state: AgentState) -> str:
    if state.get("_fast_path", False):
        return "planner"
    discovery = state.get("discovery") or {}
    return "clarify" if discovery.get("confidence", 0) < 0.7 else "planner"
```

**LangGraph feature**: The optimized graph adds a `complexity_router_node` that sets `query_complexity` and `_fast_path` on the state. Then `route_complexity_to_confidence` uses these to skip the full confidence check for simple queries. This is two conditional edges chained — complexity routing feeds into confidence gating.

### Phase 2: Planning (1 node)

#### `planner_node`

**What it does**: Creates a structured research plan with prioritized questions (primary/secondary/tertiary buckets), search queries, and an outline.

**Why it's different**: The standard pipeline's planner produces a flat list of questions and searches. The advanced planner produces a `ResearchTree` with priority-ranked question buckets, a `Plan` with an outline for the report, and typed `PlanQuery` objects instead of raw strings. This lets the orchestrator make smarter decisions about which sub-questions are most important.

**LangGraph feature**: In the optimized graph, `route_planner_by_complexity` splits after planning:

```python
def route_planner_by_complexity(state: AgentState) -> str:
    return "fast_workers" if state.get("query_complexity") == "simple" else "orchestrator"
```

Simple queries go straight to the single-worker path (just `Send("worker", ...)` directly from the planner). Complex queries go through the orchestrator for sub-question decomposition. This conditional edge is the xAI-inspired complexity routing in action.

### Phase 3: Multi-agent research (3 nodes)

#### `orchestrator_node`

**What it does**: Breaks the query into sub-tasks and assigns them to subagents. Each assignment includes the sub-question, focus area, and specific search queries.

**LangGraph feature**: The output feeds into `fanout_subagents`:

```python
def fanout_subagents(state: AgentState):
    assignments = state.get("subagent_assignments") or []
    return [
        Send("subagent", {"subagent_assignment": assignment})
        for assignment in assignments
    ]
```

This is `add_conditional_edges("orchestrator", fanout_subagents)` — same `Send()` pattern as the standard pipeline but with typed `SubagentAssignment` objects instead of raw dicts.

#### `subagent_node`

**What it does**: Each subagent runs independently — searches, fetches, extracts, and summarizes for its assigned sub-question. Returns `SubagentFindings` with answer, key findings, confidence, and gaps.

**LangGraph feature**: Returns to `Annotated[List[SubagentFindings], operator.add]` so parallel subagents merge cleanly.

#### `synthesizer_node`

**What it does**: Combines findings from all subagents, resolves conflicts, assesses overall confidence. Decides whether to loop back for more research or proceed to the trust engine.

**LangGraph feature**: `route_after_synthesis` controls the loop. On loop-back, routes to `gap_detector` first (not directly to orchestrator), which checks for dead ends before deciding whether to loop or proceed.

### Phase 4: Iterative control (3 nodes)

#### `gap_detector_node`

**What it does**: Assesses what's missing from the research. Similar to the standard pipeline's gap detector but integrated with the multi-agent loop and dead-end tracking.

**LangGraph feature**: Has three possible routing targets — `orchestrator` (more research needed), `backtrack` (dead end detected, need different approach), or `reduce` (ready to proceed). This three-way conditional edge is more complex than the standard pipeline's binary choice.

#### `backtrack_handler_node`

**What it does**: When a search path yields nothing (no results, paywalled content, irrelevant), records it as a dead end and reformulates the approach. Inspired by OpenAI's backtracking.

**LangGraph feature**: Routes back to `orchestrator` via `route_after_backtrack`, creating a cycle with dead-end awareness. The orchestrator sees the dead ends in state and avoids repeating failed paths.

#### `early_termination_check_node` (optimized graph only)

**What it does**: Checks if we should stop iterating because evidence quality has plateaued. Looks at confidence delta between iterations — if the gain is below 5%, we're done.

**LangGraph feature**: `route_after_termination_check` routes to either `reduce` (stop) or `gap_detector` (keep going). This is the early exit heuristic that prevents wasting API calls when we've exhausted useful search results.

### Phase 5: Source processing (2 nodes)

#### `reducer_node`

**What it does**: Waits for all workers to finish, then deduplicates sources and assigns unique IDs.

```python
def reducer_node(state: AgentState) -> Dict:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    if done < total:
        return {}  # workers still running
    # Deduplicate and assign IDs
    deduped = dedup_sources(raw_sources)
    sources = assign_source_ids(deduped)
    # ...
```

**LangGraph feature**: Uses the `done_workers` accumulator (via `operator.add`) as a synchronization check. This is a pattern for fan-in when you need to do post-processing after all parallel work completes.

#### `ranker_node`

**What it does**: Sorts sources by credibility score so the writer prioritizes authoritative ones.

### Phase 6: Trust engine (7 nodes full, 2 batched)

This is what separates the advanced pipeline from the standard one.

#### Full trust engine (7 sequential nodes)

```
credibility → ranker → claims → cite → span_verify → cross_validate → confidence_score → write
```

| Node | What it does | LLM calls |
|------|-------------|-----------|
| `credibility_scorer_node` | E-E-A-T scoring per source (domain trust, freshness, authority) | 1 |
| `ranker_node` | Sort by credibility | 0 |
| `claims_node` | Extract atomic factual claims from evidence | 1 |
| `cite_node` | Link each claim to supporting evidence | 1 |
| `span_verify_node` | Semantic match: does the evidence text actually say what the claim says? (0.6+ threshold) | 1 |
| `cross_validate_node` | Flag claims with only 1 source vs 2+ | 1 |
| `claim_confidence_scorer_node` | Aggregate all signals into per-claim confidence | 1 |

That's 6 LLM calls for the trust engine alone (ranker is deterministic), on top of the research calls. For a query with 20 sources, this adds real cost.

#### Batched trust engine (2 nodes)

```
batched_cred_claims → ranker → batched_verify → write
```

| Node | What it combines | LLM calls |
|------|-----------------|-----------|
| `batched_credibility_claims_node` | Credibility scoring + claim extraction in one prompt | 1 |
| `batched_verification_node` | Span verification + cross-validation + confidence scoring in one prompt | 1 |

Same logic, ~60% fewer API calls. The tradeoff: the batched prompts are longer and more complex, so the LLM occasionally misses nuances that the dedicated nodes would catch. For most queries, the quality difference is negligible.

**LangGraph feature**: The optimized graph simply swaps the trust engine subgraph:

```python
# Full trust engine
g.add_edge("credibility", "ranker")
g.add_edge("ranker", "claims")
g.add_edge("claims", "cite")
# ... 4 more edges

# Batched trust engine
g.add_edge("batched_cred_claims", "ranker")
g.add_edge("ranker", "batched_verify")
g.add_edge("batched_verify", "write")
```

Same graph structure, different node implementations. The routing edges are identical.

### Phase 7: Writing (1 node)

#### `writer_node`

Same concept as the standard pipeline's `write_report_node` but receives trust engine output — per-claim confidence scores, credibility rankings, cross-validation results. Can annotate the report with confidence indicators (high/medium/low per claim).

### Advanced pipeline config

The advanced pipeline uses a frozen dataclass instead of `RunnableConfig`:

```python
@dataclass(frozen=True)
class ResearchConfig:
    max_rounds: int = 2
    queries_per_round: int = 10
    max_research_iterations: int = 2
    min_confidence_to_proceed: float = 0.7

    # Model routing — cheap models for extraction, expensive for reasoning
    model_routing: Tuple[Tuple[str, str], ...] = (
        ("analyzer", "gpt-4o-mini"),
        ("planner", "gpt-4o"),
        ("writer", "gpt-4o"),
        # ... everything else uses mini
    )

    # E-E-A-T inspired source credibility
    trusted_domains: tuple = (".edu", ".gov", "arxiv.org", "nature.com", ...)
    low_trust_domains: tuple = ("pinterest.com", "quora.com", "buzzfeed.com", ...)
    min_source_credibility: float = 0.35

    # Citation verification thresholds
    span_match_threshold: float = 0.6
    cross_validation_threshold: int = 2
    hallucination_threshold: float = 0.3
```

The `frozen=True` means nodes can't accidentally mutate config mid-run. `get_model_for_node()` does the routing lookup so each node gets the right model without hardcoding it.

---

## Standard vs Advanced: side-by-side comparison

Now that both pipelines have been covered in detail, here's how they actually compare.

### Architecture

| Dimension | Standard pipeline | Advanced pipeline |
|-----------|------------------|-------------------|
| **Total nodes** | 11 | 20+ (across 12 modules) |
| **State fields** | ~20 | ~60 (with 25+ typed sub-dicts) |
| **Graph variants** | 1 (`build_graph`) | 3 (`trust_engine`, `optimized`, `simple`) |
| **Config approach** | `RunnableConfig["configurable"]` (Studio knobs) | Frozen dataclass `ResearchConfig` (~50 tunables) |
| **Model routing** | 2 models (smart + fast) | 9 node-level model assignments |

### Flow comparison

| Phase | Standard | Advanced |
|-------|----------|----------|
| **Query understanding** | Ambiguity check (7 dimensions) | Full NLP: intent, entity detection, temporal scope, complexity |
| **Entity disambiguation** | None | ERA-CoT: discovery search → candidate clustering → confidence gating |
| **HITL trigger** | `is_ambiguous == True` | `discovery_confidence < 0.7` (confidence-gated) |
| **Planning** | Flat list of questions + searches | Priority-ranked `ResearchTree` (primary/secondary/tertiary) |
| **Complexity routing** | None (mode is set by caller) | Keyword + word-count heuristic → skip multi-agent for simple queries |
| **Parallel dispatch** | `Send()` per sub-question | `Send()` per sub-question, with query dedup (0.85 threshold) |
| **Gap detection** | LLM-based coverage assessment | Hybrid heuristic + LLM, with dead-end awareness |
| **Backtracking** | None (just loops or stops) | Dead-end categorization → alternative query generation |
| **Early termination** | Iteration cap + coverage threshold | Confidence delta < 5%, source novelty, cost budget |
| **Source scoring** | None | E-E-A-T: domain trust + freshness + authority + content quality |
| **Claim verification** | Bulk verify (supported/unsupported) | Span-level semantic match (0.6+ threshold) |
| **Cross-validation** | None | Flags claims with 2+ independent sources |
| **Confidence output** | Single number (supported/total) | Per-claim confidence with indicators in the report |

### LLM calls per run (typical query, single iteration)

| Step | Standard (single) | Standard (multi, 4 workers) | Advanced (full trust) | Advanced (batched) |
|------|------|------|------|------|
| Query understanding | 1 | 1 | 2 (analyzer + discovery) | 2 |
| Planning | 1 | 1 | 1 | 1 |
| Orchestration | 0 | 1 | 1 | 1 |
| Search + extraction | 4-8 (depends on sources) | 8-16 (4 workers × 2-4 each) | 8-16 | 8-16 |
| Gap detection | 1 | 0 (synthesize instead) | 1 | 1 |
| Synthesis | 0 | 1 | 1 | 1 |
| Verification | 1 | 1 | 0 (trust engine instead) | 0 |
| **Trust engine** | **0** | **0** | **6** (cred + claims + cite + span + cross + confidence) | **2** (batched) |
| Report writing | 1 | 1 | 1 | 1 |
| **Total** | **~9-12** | **~13-21** | **~21-28** | **~17-23** |

The batched trust engine cuts 4 LLM calls per run. On a broad query with 2 iterations, that's 8 fewer calls — roughly 60% fewer trust engine tokens.

### LangGraph features used

| Feature | Standard | Advanced |
|---------|----------|----------|
| `StateGraph` | Yes | Yes |
| `MessagesState` inheritance | Yes | Yes |
| `Annotated[..., operator.add]` | 3 fields | 5 fields |
| `Send()` parallel fan-out | 1 fan-out point | 2 fan-out points |
| `interrupt_before` (HITL) | Yes (`clarify` node) | Yes (`clarify` node) |
| `MemorySaver` checkpointing | Yes | Yes |
| Conditional edges | 5 routing functions | 10+ routing functions |
| `RunnableConfig` / `context_schema` | Yes (Studio knobs) | No (uses frozen dataclass) |
| Cycles (loop-back edges) | 2 (gap loop + synthesis loop) | 3 (gap loop + synthesis loop + backtrack loop) |
| Graph builder helpers | 1 (`build_graph`) | 6 (trust/optimized × full/memory/simple) |

### When to use which

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Quick demo or prototype | Standard (multi) | Faster, simpler, good enough for most queries |
| Narrow factual query | Standard (single) | Iterative depth > parallel breadth for focused topics |
| Broad multi-dimensional query | Standard (multi) or Advanced | Parallelism helps; advanced adds source quality |
| Medical / legal / high-stakes | Advanced (full trust) | Source credibility and span verification matter |
| Cost-sensitive production | Advanced (batched) | 60% fewer trust engine calls with comparable quality |
| Automated pipeline (no user) | Either with `_simple` builder | Skip HITL, no checkpointer overhead |

---

## Design decisions I'd make differently

**Citation numbering (standard pipeline).** The writer node gets evidence labeled with source IDs and a valid citation list. But the LLM sometimes hallucinates citation numbers that don't exist. I fixed this by injecting `VALID CITATION NUMBERS: [1, 3, 5, 7]` into the system prompt, but a proper solution would post-process the report and rewrite any invalid `[N]` markers to the closest valid source. The eval suite catches this.

**Search deduplication (standard pipeline).** Tracks `searches_done` with exact string matching. "quantum computing applications" and "applications of quantum computing" are treated as different queries. The advanced pipeline handles this with `query_similarity_threshold: 0.85` — an embedding-based similarity check. I should have added that to the standard pipeline too.

**Configuration approach.** The standard pipeline uses `RunnableConfig["configurable"]` which integrates nicely with LangGraph Studio (auto-rendered knobs) but isn't type-checked. The advanced pipeline uses `ResearchConfig` (frozen dataclass) which gives IDE autocomplete and catches typos, but doesn't render in Studio. The ideal would be both — a typed config that also integrates with Studio's schema.

**Trust engine cost.** On a broad query with 20 sources, the full trust engine makes 6+ LLM calls just for verification. A smarter approach would verify only the claims that make it into the final report, not every claim from every source. This would cut trust engine costs by probably 60-70% with minimal quality loss.

**Early termination.** Currently checks if confidence gain between iterations is below 5%. A better heuristic would also factor in how many new unique sources the last iteration found — if we're seeing the same sites over and over, there's no point continuing.

---

## State design rationale

Both pipelines follow the same principle: **state is the single source of truth**. Every node reads from state and writes back to state. No hidden globals, no message passing between nodes, no side channels.

The standard pipeline has ~20 state fields. The advanced pipeline has 30+, organized into typed sub-structures:

```python
# Advanced state uses typed sub-dicts for complex data
class QueryAnalysisResult(TypedDict): ...     # 8 fields
class Plan(TypedDict): ...                     # 5 fields
class Source(TypedDict): ...                   # 6 fields
class Evidence(TypedDict): ...                 # 7 fields
class Claim(TypedDict): ...                    # 4 fields
class SourceCredibility(TypedDict): ...        # 6 fields
class VerifiedCitation(TypedDict): ...         # 5 fields
class SubagentAssignment(TypedDict): ...       # 4 fields
class SubagentFindings(TypedDict): ...         # 6 fields
# ... 15+ more
```

This is verbose but it means every piece of data has a clear schema. When a node writes `evidence`, you know exactly what fields each item has. The alternative (just using `Dict[str, Any]` everywhere) would be shorter but would hide bugs until runtime.

---

## Evaluation approach

The eval suite is in `tests/`. Three layers, designed to run standalone without LangSmith.

### Structural evaluators 

10 deterministic checks against the pipeline output:

| Check | What it catches |
|-------|----------------|
| `check_report_has_markdown_structure` | Missing title, sections, or sources section |
| `check_citations_present` | No `[N]` markers in report |
| `check_citations_map_to_sources` | Citation `[19]` but only 12 sources exist |
| `check_metadata_complete` | Missing keys, wrong types in metadata dict |
| `check_evidence_populated` | Empty evidence list |
| `check_scores_in_bounds` | Coverage or confidence outside [0,1] |
| `check_report_minimum_length` | Report under 200 words |
| `check_evidence_fields` | Evidence items missing `fact`, `source_url`, or `source_title` |
| `check_sources_have_urls` | Sources missing `url` or `title` |
| `check_no_empty_report` | Empty or whitespace-only report |

These are fast and catch regressions. The citation mapping check is what originally caught the citation numbering bug.

### LLM-as-judge evaluators (uses gpt-4o)

Five evaluators using Pydantic structured output for reliable, parseable scores:

| Evaluator | What it measures | Output |
|-----------|-----------------|--------|
| **Relevance** | Does the report answer the query? Per-section check. | 1-5 score |
| **Groundedness** | Claim-level extraction. Every factual claim checked against evidence. | grounded/total ratio |
| **Completeness** | Report vs research plan — did we address all planned questions? | 1-5 score + covered/missed lists |
| **Quality** | 5 sub-scores: research depth, source diversity, analytical rigor, clarity, citation integration | Average of five 1-5 scores |
| **Citation faithfulness** | Do inline citations accurately represent their sources? | 1-5 score |

The groundedness evaluator is the most granular — it extracts every factual claim from the report and checks each one against the pipeline's actual evidence. This gives a concrete ratio (e.g., "23/27 claims grounded") rather than a vibes-based score.

All evaluators use `with_structured_output()` on `ChatOpenAI` directly (not our LLMWrapper) because we need Pydantic schema enforcement for reliable parsing.

### Behavioral evaluators (tests the architecture)

These test whether the pipeline's design features actually work, not just whether the output looks good:

| Test | What it validates |
|------|------------------|
| **Config effectiveness** | Same query with `report_structure="detailed"` vs `"bullet_points"` produces different formats |
| **Gap loop improvement** | More iterations yields equal or more evidence (monotonic) |
| **Mode comparison** | Both single and multi-agent produce valid reports |
| **Ambiguity detection** | Known-ambiguous queries trigger `is_ambiguous=True` |
| **State consistency** | `metadata.sources_count` matches `len(state["sources"])` |
| **Iteration convergence** | Coverage is monotonically non-decreasing across iterations |
| **Verified claims ratio** | `confidence` matches actual `supported / total` in verified_claims |

### Evaluation inspiration

Criteria inspired by [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research), which uses 6 LLM-as-judge evaluators (relevance, structure, correctness, groundedness, completeness, overall quality) plus pairwise comparison via Claude Opus with extended thinking.

Key differences in our eval:
- **No LangSmith dependency** — runs standalone with pytest or the `run_eval.py` script
- **Behavioral tests** — config effectiveness, gap loop verification, mode comparison. These test architectural correctness, not just output quality. The reference repo doesn't do these.
- **Groundedness against pipeline evidence** — their groundedness eval checks against `raw_notes`. Ours checks against `state["evidence"]`, the actual structured evidence the pipeline collected.

---

## Running it

### Research

```bash
# Multi-agent (default)
python -m src.run "What is quantum computing?"

# Single-agent with iterative gap detection
python -m src.run --single-agent "Compare React vs Vue"

# Run both and compare metrics side-by-side
python -m src.run --compare "Latest AI developments"

# Custom config
python -m src.run --model gpt-4o-mini --format concise --iterations 3 "Explain CRISPR"

# Custom system prompt
python -m src.run --system-prompt "Write for a technical audience" "Rust vs Go"
```

### Evaluation

```bash
# Fast offline check (mock states, no API calls)
python -m pytest tests/test_structural.py tests/test_behavioral.py -m "not live" -v

# Full live evaluation (runs the pipeline, scores with LLM judge)
python -m tests.run_eval --live --max-iterations 3 --output results.json

# Re-score saved snapshots without re-running the pipeline
python -m tests.run_eval --from-snapshots --output results.json

# Single case for quick testing
python -m tests.run_eval --live --cases factual_simple --output results_quick.json
```

---

## Project structure

```
research-studio/
├── src/
│   ├── pipeline/               # Standard pipeline (~1,200 LOC)
│   │   ├── state.py            # ResearchState (MessagesState) + Configuration
│   │   ├── prompts.py          # All LLM prompts (understand, plan, extract, gap, verify, write)
│   │   ├── nodes.py            # 11 node functions
│   │   └── graph.py            # StateGraph wiring, routing functions, build_graph()
│   ├── advanced/               # Advanced pipeline (~4,000 LOC)
│   │   ├── config.py           # ResearchConfig frozen dataclass (~50 tunables)
│   │   ├── state.py            # AgentState (MessagesState) + 25+ TypedDicts
│   │   ├── graph.py            # 3 graph variants × 3 builder helpers
│   │   └── nodes/              # 13+ node modules
│   │       ├── discovery.py    # analyzer, discovery, clarify, confidence routing
│   │       ├── planner.py      # Research planning with priority buckets
│   │       ├── orchestrator.py # Sub-task decomposition, subagent dispatch, synthesis
│   │       ├── search_worker.py # Parallel search-fetch-extract worker
│   │       ├── reducer.py      # Fan-in deduplication
│   │       ├── iterative.py    # Gap detection, backtracking, early termination
│   │       ├── claims.py       # Atomic claim extraction
│   │       ├── cite.py         # Claim-to-evidence citation mapping
│   │       ├── ranker.py       # Source ranking by credibility
│   │       ├── trust_engine.py # Credibility, span verify, cross-validate, confidence
│   │       ├── trust_engine_batched.py # Batched variant (2 LLM calls instead of 7)
│   │       └── writer.py       # Report generation with trust annotations
│   ├── tools/                  # Tavily search, HTTP fetch
│   ├── utils/                  # LLM wrappers, caching, JSON parsing
│   ├── run.py                  # CLI for standard pipeline
│   └── run_advanced_trust_engine.py  # CLI for advanced pipeline
├── tests/
│   ├── evaluators/
│   │   ├── structural.py       # 10 deterministic checks
│   │   ├── llm_judge.py        # 5 LLM-as-judge evaluators (Pydantic structured output)
│   │   └── behavioral.py       # 7 architectural tests
│   ├── eval_cases.py           # 8 test case definitions
│   ├── conftest.py             # Fixtures, pipeline runner, mock state builder
│   ├── run_eval.py             # Main evaluation runner (--live, --offline, --from-snapshots)
│   ├── test_structural.py      # 18 pytest tests
│   ├── test_llm_judge.py       # 10 pytest tests
│   ├── test_behavioral.py      # 11 pytest tests
│   └── snapshots/              # Saved pipeline outputs for offline re-evaluation
├── langgraph.json              # LangGraph Studio graph registration
└── pyproject.toml
```

---

## Sources and references

- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)
- [How Deep Research Works (PromptLayer)](https://blog.promptlayer.com/how-deep-research-works/)
- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [xAI Grok 3](https://x.ai/news/grok-3)
- [Google Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)
- [Perplexity Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)
- [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) — evaluation criteria inspiration
- [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher) — reference implementation


## AI Tools Used
- Claude Code (mostly for code completion, debugging, deep research and writing docs(specifically the table and code snippet parts of it))
