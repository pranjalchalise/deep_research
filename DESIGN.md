# Design Document

## What this is

Research Studio is a deep research agent I built on LangGraph. You give it a question, it goes and searches the web, reads pages, pulls out facts, checks them, and writes you a report with proper citations. There are two pipelines:

- **Standard pipeline** (`src/pipeline/`) -- the clean version. Supports single-agent (iterative loop) and multi-agent (parallel workers with LangGraph Send). Around 1,200 lines total.
- **Advanced pipeline** (`src/advanced/`) -- same core idea but with a trust engine layered on top: source credibility scoring, claim extraction, span verification, cross-validation, per-claim confidence. Around 4,000 lines across 13+ modules.

Both are registered in `langgraph.json` and work in LangGraph Studio.

```json
{
  "graphs": {
    "pipeline": "./src/pipeline/graph.py:graph",
    "advanced": "./src/advanced/graph.py:graph"
  }
}
```

You can open Studio, pick either graph, see the state at each node, inspect edges, play with the config knobs, etc.

---

## Why two pipelines?

I didn't plan to have two pipelines. I started with what's now the advanced one and just kept bolting things onto it -- credibility scoring, span verification, cross-validation, then a batched version of all that. It worked but it got really hard to debug. When something went wrong with a report I'd spend more time figuring out which trust engine node messed up than actually fixing the research logic.

So at some point I just pulled the core research flow out into its own clean pipeline (`src/pipeline/`). No trust engine, no credibility scores. Just: understand the query, make a plan, search, extract facts, check for gaps, verify, write. It's way easier to demo and test, and honestly it handles like 80% of queries fine.

The advanced pipeline is there to show what you can do when you really care about source quality -- medical stuff, legal topics, current events where misinformation is a real risk. It's not always better though. It's slower, burns more API calls, and for simple factual queries the extra verification doesn't add much.

---

## How I got here

### Industry research

Before I started writing code I spent a while reading about how the big companies do deep research. I wanted to figure out which patterns were universal (everybody does them) vs which ones were specific to having massive infrastructure we can't replicate. The full analysis is in `docs/archive/DEEP_RESEARCH_AGENT_PLAN.md`, but here's what I found.

### OpenAI Deep Research

| Component | Implementation |
|-----------|----------------|
| **Model** | o3 with expanded "attention span" for long chains of thought |
| **Training** | End-to-end RL on simulated research environments with tool access |
| **Loop** | Plan -> Act -> Observe (ReAct paradigm) with backtracking |
| **Tools** | Web browser, PDF parser, image analyzer, Python code execution |
| **Duration** | Up to 30 minutes, 21+ sources, dozens of search queries |
| **Summarization** | o3-mini used to summarize chains of thought |

The big thing here is they trained o3 end-to-end on simulated research environments with actual tool access. Obviously I can't replicate that -- no training infrastructure. But the backtracking and iterative refinement patterns? Those are just control flow. When a search path hits a paywall or gives you junk, pivot and try something else. That's exactly what `backtrack_handler_node` does in the advanced pipeline, and what the `detect_gaps` loop does in the standard pipeline (just simpler).

**Performance**: 26.6% on Humanity's Last Exam (3,000 questions across 100 subjects).

### Anthropic's multi-agent research system

| Component | Implementation |
|-----------|----------------|
| **Lead Agent** | Claude Opus -- analyzes query, develops strategy, spawns subagents |
| **Subagents** | Claude Sonnet -- 3-5 parallel workers with own context windows |
| **Parallelism** | Two levels: subagent spawning + tool calls within subagents |
| **Search** | Brave Search integration via web_search tool |
| **Compression** | Subagents condense findings before returning to lead agent |

This is the one that influenced my architecture the most. The orchestrator-worker pattern maps almost directly to LangGraph's `Send()` API -- orchestrator decides how to split the work, `Send()` dispatches workers in parallel, LangGraph handles the sync. No manual threading or locking.

Their insight about programmatic tool calling reducing tokens by 37% was really practical. Instead of letting the LLM decide when to call tools (ReAct style), I just hardcode the tool calls in the node functions. The LLM handles the thinking (what to extract, how to summarize) but the actual search and fetch calls are deterministic. Way fewer tokens wasted on "I should now call the search tool with query..." reasoning.

**Performance**: 90.2% improvement over single-agent Claude Opus on internal evals. Multi-agent uses ~15x more tokens than single chat.

### xAI Grok DeepSearch

| Component | Implementation |
|-----------|----------------|
| **Model** | Grok 3 trained on 200K H100 GPUs |
| **Search** | Real-time web + X/Twitter signals |
| **Reasoning** | "Think" mode for slower, deliberate chain-of-thought |
| **Speed** | 67ms average response latency |
| **Context** | 128K token window |

The complexity routing idea came from xAI. Simple queries like "What is CRISPR?" really don't need you to spawn 3-5 parallel workers. That's just wasting API calls. In the advanced pipeline, `complexity_router_node` does a quick keyword + word-count check to classify queries as simple/medium/complex. Simple ones skip straight to a single worker, complex ones get the full orchestrator treatment. It's just a conditional edge in the graph -- LangGraph makes this super easy to implement.

### Google Gemini Deep Research

| Component | Implementation |
|-----------|----------------|
| **Model** | Gemini 3 Pro, trained to reduce hallucinations |
| **Context** | 1M token window + RAG for session memory |
| **Task Manager** | Asynchronous with shared state between planner and task models |
| **Sources** | Web + Gmail + Drive + Chat integrations |
| **Multimodal** | Images, PDFs, audio, video as research inputs |

The knowledge gap detection loop came from here. The idea is: search, read, figure out what's still missing, search again with better queries. That's basically what my gap detection does. After each search round, the LLM looks at what we've found vs the research questions from planning, finds the holes, suggests new queries. If coverage is below 70% and we haven't hit the iteration cap, loop back. Google does this with async task management, I do it with LangGraph conditional edges pointing back to earlier nodes.

**Performance**: 46.4% on Humanity's Last Exam, 66.1% on DeepSearchQA, 59.2% on BrowseComp.

### Perplexity

| Component | Implementation |
|-----------|----------------|
| **Routing** | Dynamic routing to different engines (conversational, research, coding) |
| **Source Ranking** | E-E-A-T scoring (Experience, Expertise, Authority, Trust) |
| **Comet Framework** | Retrieval Agent + Synthesis Agent + Verification Agent |
| **Scale** | 200M daily queries |

E-E-A-T scoring for source ranking came from Perplexity. Scoring sources on experience, expertise, authority, trustworthiness -- that's what `credibility_scorer_node` does, with a shortcut for known domains (`.edu`/`.gov` = high trust, `quora.com` = low) plus LLM-assessed content quality on top. Their Comet framework (separate agents for retrieval, synthesis, verification) maps pretty directly to our orchestrator + synthesizer + trust engine split.

### What patterns showed up everywhere

Every system does some version of the same things:

1. **Break queries into sub-questions** -- OpenAI does it implicitly in chain-of-thought, Anthropic has the lead agent generate them, I do it in the orchestrator
2. **Search iteratively** -- Google's gap loop, OpenAI's backtracking, my `detect_gaps -> search_and_extract` cycle
3. **Backtrack when stuck** -- when a search path gives you nothing useful, try a different angle
4. **Score source quality** -- Perplexity's E-E-A-T, my trust engine
5. **Verify citations against evidence** -- span-level or claim-level checking
6. **Use multiple models** -- cheap ones for extraction, expensive ones for reasoning and writing

The standard pipeline covers patterns 1-2-5-6. The advanced pipeline covers all six.

### Cross-system comparison

| System | Query Strategy | Refinement | Parallelism |
|--------|---------------|------------|-------------|
| OpenAI | Implicit in CoT | Based on results read | Sequential with backtrack |
| Anthropic | Lead agent generates | Subagents refine independently | 3-5 subagents parallel |
| xAI | Sub-query decomposition | Think mode for depth | Dynamic workflows |
| Google | Research plan from prompt | Gap detection -> new queries | Task-level parallel |
| Perplexity | Research tree branching | Multi-step in Pro Search | Comet multi-agent |

### What I would have done with more time

| Gap | What they do | Where I am | Priority |
|-----|-------------|------------|----------|
| **Multimodal inputs** | PDF, images, video | Text only | Medium |
| **Code execution** | Python for data analysis | None | Medium |
| **Async operation** | Background + notify | Synchronous only | Low |
| **1M+ context** | Google's session memory | Standard context windows | Low |
| **End-to-end RL** | OpenAI's o3 training | Prompt engineering only | Not feasible |

---

## The development journey (v6 through v9)

This didn't come together all at once. I went through several major versions, each one building on the last.

**v6** was where I added deduplication and a ledger system. Before this, the pipeline would sometimes search for the same thing multiple times and the evidence list would have tons of duplicates. The ledger tracked what we'd already searched for so we wouldn't repeat ourselves. It helped but the overall architecture was still pretty messy.

**v7** is where things got interesting. I added a proper planner node, knowledge gap detection, and gap-based looping. Before v7 the pipeline would just do one round of searching and hope it was enough. Now it could actually look at what it found, figure out what was missing, and go back for more. This was directly inspired by the Google approach. I also refined the extraction prompts a lot because the LLM kept pulling out vague "facts" that weren't actually useful.

**v8** was the big one -- multi-agent orchestration. I added the orchestrator-subagent pattern (from Anthropic), parallel workers via `Send()`, and a synthesizer to merge results. This is also when I added the reflection loop and confidence filtering. v8 was the first version that could actually handle broad queries well because it could research multiple sub-topics simultaneously instead of doing everything sequentially.

**v9** was wrapping everything in LangGraph properly. Before this I had some parts wired up with LangGraph and some parts that were just function calls. v9 made the whole thing a proper `StateGraph` with typed state, conditional edges, the works. That's also when I split into two pipelines because the advanced one was getting too big to reason about.

---

## LangGraph features I used (and why)

I want to go through the specific LangGraph features because some of them were not obvious choices and I had to figure out the right patterns through trial and error.

### 1. `StateGraph` with typed state

Both pipelines build on `StateGraph` with a TypedDict state class. State is the single source of truth -- every node reads from it and writes back to it.

```python
# Standard pipeline
g = StateGraph(ResearchState, context_schema=Configuration)

# Advanced pipeline
g = StateGraph(AgentState)
```

The `context_schema=Configuration` on the standard pipeline is worth calling out. It registers a separate TypedDict as the config schema, which does two things: tells LangGraph Studio to render those fields as editable knobs in the UI, and keeps config separate from state. Config lives in `RunnableConfig["configurable"]`, not in the graph state. So changing the model or report format doesn't pollute the research state. I didn't do this on the advanced pipeline (it uses a frozen dataclass instead) and honestly I wish I had -- the Studio integration is nicer with `context_schema`.

### 2. `MessagesState` inheritance

Both state classes extend `MessagesState`:

```python
class ResearchState(MessagesState, total=False):
    query: str
    mode: str
    evidence: Annotated[List[Dict], operator.add]
    # ... 20+ fields
```

This gives you the `messages` field with LangGraph's built-in `add_messages` reducer for free. I use messages to track what's happening -- original query comes in as `HumanMessage`, nodes can append status messages, final report goes out as `AIMessage`. This is what makes it work in Studio and chat UIs without extra effort.

The `total=False` is important too -- means nodes can return partial updates. If `understand_node` only sets `understanding` and `is_ambiguous`, that's fine. Don't have to return every field every time.

### 3. `operator.add` for parallel-safe accumulation

This is the thing that makes multi-agent mode work without race conditions:

```python
worker_results: Annotated[List[Dict], operator.add]
done_workers: Annotated[int, operator.add]
evidence: Annotated[List[Dict], operator.add]
```

When multiple workers run in parallel via `Send()`, they all write to these fields at the same time. Without `operator.add`, the last worker to finish would overwrite everyone else. With it, LangGraph concatenates the lists and sums the integers automatically. Worker A returns `evidence: [fact1, fact2]`, Worker B returns `evidence: [fact3]`, state ends up with all three. No locks, no manual merging.

I learned this one the hard way. Early versions just had `evidence: List[Dict]` and I kept getting results from only one worker. Took me a while to realize the issue was that workers were overwriting each other, not that they were failing.

The `done_workers: Annotated[int, operator.add]` pattern is nice too -- each worker returns `done_workers: 1`, LangGraph sums them, and after all workers finish `done_workers == total_workers`. The reducer checks this to know when it's safe to proceed.

### 4. `Send()` for parallel fan-out

This is the core of multi-agent mode:

```python
def fanout_workers(state: ResearchState) -> List[Send]:
    return [
        Send("search_worker", {
            "sub_question": sq,
            "query": state["query"],
            "user_clarification": state.get("user_clarification", ""),
        })
        for sq in state.get("sub_questions", [])
    ]
```

Orchestrator decides how to split the query, writes sub-questions to state. Then `fanout_workers` creates one `Send()` per sub-question. LangGraph runs them all in parallel and waits for all of them before moving to the next node.

Why `Send()` and not just threading manually:
- Each worker gets isolated state, can't interfere with each other
- `operator.add` handles merging
- LangGraph guarantees all workers finish before the downstream node runs
- Shows up clearly in Studio's graph visualization

The advanced pipeline has two fan-out points -- `fanout_workers` for the single-agent fallback and `fanout_subagents` for the orchestrator pattern.

### 5. `interrupt_before` for HITL

When the query is ambiguous ("Python" -- the language or the snake?), the pipeline needs to stop and ask the user. LangGraph's `interrupt_before` handles this:

```python
def build_graph(checkpointer=None, interrupt_on_clarify: bool = True):
    g = StateGraph(ResearchState, context_schema=Configuration)
    # ... all nodes and edges ...
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_on_clarify:
        compile_kwargs["interrupt_before"] = ["clarify"]
    return g.compile(**compile_kwargs)
```

Graph hits `clarify`, pauses. Caller inspects state (which has `clarification_question` and `clarification_options`), shows them to the user, gets an answer, injects it with `update_state()`, resumes:

```python
result = graph.invoke(init, config)
snapshot = graph.get_state(config)
if snapshot.next and "clarify" in snapshot.next:
    user_input = input("Your choice: ")
    graph.update_state(config, {"user_clarification": user_input})
    result = graph.invoke(None, config)  # resume
```

Needs a checkpointer (`MemorySaver`) because the graph has to persist state across the pause. For automated runs you skip it entirely:

```python
graph = build_graph(checkpointer=None, interrupt_on_clarify=False)
```

### 6. Conditional edges for routing

Every decision point is a conditional edge with a routing function:

```python
g.add_conditional_edges("understand", route_after_understand,
                        {"clarify": "clarify", "plan": "plan"})

g.add_conditional_edges("detect_gaps", route_after_gaps,
                        {"search_and_extract": "search_and_extract",
                         "verify": "verify"})
```

The routing functions are just pure functions of state -- read a few fields, return the next node name. Makes the control flow explicit and testable:

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

Four exit conditions, all spelled out. The standard pipeline has 5 routing functions, the advanced one has 10+.

### 7. Graph cycles

Both pipelines have loops -- `detect_gaps -> search_and_extract -> detect_gaps` in single-agent mode, and `synthesize -> orchestrate -> ... -> synthesize` in multi-agent mode. LangGraph handles cycles natively, you just point a conditional edge back to an earlier node. The iteration counter prevents infinite loops.

### 8. `RunnableConfig` for per-run settings

Every node gets `config: RunnableConfig` as its second argument. Settings come from `config["configurable"]`:

```python
def understand_node(state: ResearchState, config: RunnableConfig) -> Dict:
    cfg = get_configuration(config)
    llm = create_chat_model(cfg["model"])
```

Change the model, report format, or system prompt per run without rebuilding the graph. In Studio these show up as editable knobs because of `context_schema=Configuration`.

### 9. Multiple graph variants

The advanced pipeline provides 6 builder functions for different deployment scenarios:

| Builder | Checkpointer | HITL | Use case |
|---------|-------------|------|----------|
| `build_trust_engine_graph()` | Optional | Optional | Full control |
| `build_trust_engine_graph_with_memory()` | MemorySaver | Yes | Interactive CLI |
| `build_trust_engine_simple_graph()` | None | No | Automated pipelines |
| `build_optimized_graph()` | Optional | Optional | Full control, batched |
| `build_optimized_graph_with_memory()` | MemorySaver | Yes | Interactive CLI, batched |
| `build_optimized_simple_graph()` | None | No | Automated, batched |

That's probably more builders than I strictly need, but it means the caller never has to think about which compile kwargs to pass. `_with_memory` = HITL. `_simple` = just run it.

---

## Standard pipeline -- the nodes

The standard pipeline has 11 nodes. I'll go through each one -- what it does, why I built it that way, and what went wrong along the way.

### Flow

```
Single-agent:
  START -> understand -> [clarify] -> plan -> search_and_extract -> detect_gaps
    (loop back if gaps remain)
  detect_gaps -> verify -> write_report -> END

Multi-agent:
  START -> understand -> [clarify] -> plan -> orchestrate -> [search_worker x N] -> collect
    -> synthesize -> (loop back to orchestrate if gaps remain)
  synthesize -> verify -> write_report -> END
```

### 1. understand_node

Takes the user's query and figures out if it's clear enough to research or if we need to ask for clarification first. Sends the query to the LLM with a structured prompt that checks 7 types of ambiguity: polysemy, scope, time period, geography, perspective, specificity, comparison baseline. If it's unclear, also returns a clarification question with 3-5 options.

I added this because without it you risk spending 30 seconds researching "Python the snake" when the user meant the programming language. The structured options (instead of open-ended "what do you mean?") make the HITL flow way cleaner.

This node's output drives the first conditional edge -- `route_after_understand` checks `is_ambiguous` and routes to either `clarify` or `plan`.

### 2. clarify_node

Pretty much a passthrough. The real work happens before this node runs -- graph pauses via `interrupt_before`, caller shows the clarification question, gets the user's answer, injects it with `update_state()`. This node just makes the clarification visible in state for downstream nodes.

It exists as a separate node because without it I'd have to interrupt inside `understand_node` (messy) or `plan_node` (too late). Having a dedicated pause point makes the graph clearer.

### 3. plan_node

Creates a research plan -- what questions to answer, what to search for. Takes the query (plus clarification if we got one), sends it to the LLM, gets back 3-5 research questions, aspects to cover, and 4-6 search queries. Also resets all the accumulators (`evidence: []`, `worker_results: []`, etc.) so the research loop starts clean.

Without planning, the pipeline would just search for the raw query string. With it, the LLM can decompose "Compare React vs Vue for a startup in 2024" into targeted questions about performance, learning curve, hiring market, etc.

### 4. orchestrate_node (multi-agent only)

Breaks the query into 3-5 independent sub-questions for parallel research. On loop-back (when synthesis found gaps), it includes the remaining gaps so the LLM targets what's actually missing instead of re-asking the same stuff.

The output feeds into `fanout_workers` which returns `List[Send]` -- one parallel worker per sub-question.

### 5. search_worker_node (multi-agent only)

Each worker takes one sub-question and does the full cycle: search via Tavily (up to 3 queries), fetch pages (max 2 per search), extract facts with the fast LLM (max 4 per page), then summarize with the smart LLM.

The caps (3 searches, 2 pages, 4 facts per page) keep costs bounded. Without them a broad sub-question could trigger dozens of API calls. I tuned these numbers through trial and error -- too low and you miss stuff, too high and a single query burns through your API budget.

All three output fields (`worker_results`, `done_workers`, `evidence`) use `operator.add` so parallel workers don't overwrite each other.

### 6. collect_node (multi-agent only)

Deduplicates sources from all workers into a single source dictionary. Workers researching different sub-questions sometimes find the same URLs -- this merges them and assigns unique source IDs. LangGraph guarantees this only runs after all `Send()` workers complete.

### 7. synthesize_node (multi-agent only)

Combines findings from all workers, resolves conflicts ("Worker A says X but Worker B says Y"), assesses confidence, and decides whether to loop back. If there are significant gaps and we have iteration budget left, routes back to orchestrate for another round.

### 8. search_and_extract_node (single-agent only)

The single-agent equivalent of search_worker, but runs sequentially through all pending searches instead of in parallel. Trades parallelism for context -- each extraction call has the full query and all research questions as context, so it can be more targeted than a worker that only knows about its sub-question.

### 9. detect_gaps_node (single-agent only)

This is the core of the single-agent loop. After each search round, sends all collected evidence plus the original research questions to the LLM. It assesses coverage per question (well_covered / partial / missing), identifies gaps, suggests new search queries. If coverage is below 70% and we haven't hit the iteration cap, we loop back to search_and_extract.

This is the Google-inspired pattern. Without it I was doing one round of searching and hoping for the best. With it the pipeline can discover "we have good stuff on React's performance but nothing about Vue's developer experience" and generate targeted follow-up searches.

### 10. verify_node

Cross-checks claims against evidence to calculate a confidence score. Takes up to 20 evidence items, asks the LLM to check each claim, returns supported/unsupported flags. `confidence = supported_count / total_claims`. This also filters out any hallucinated "facts" that crept in during extraction.

Both paths (single and multi-agent) converge here.

### 11. write_report_node

The most complex node. Generates the final markdown report. It:
1. Builds a `url_to_sid` mapping so evidence references the right source IDs
2. Formats evidence with `[Source N]` labels
3. Picks the prompt based on mode (`WRITE_MULTI_PROMPT` vs `WRITE_PROMPT`)
4. Injects valid citation numbers into the system prompt so the LLM only references real sources
5. Applies the `report_structure` config (detailed / concise / bullet_points)
6. Appends a sources section as fallback
7. Records metadata (iterations, coverage, confidence, counts)

The citation number injection (step 4) was a fix I found through the eval suite. The LLM was hallucinating citation numbers like `[19]` when there were only 12 sources. Adding `VALID CITATION NUMBERS: [1, 3, 5, 7]` to the prompt mostly fixed it, though a proper solution would post-process and rewrite invalid markers.

---

## Advanced pipeline -- key nodes

The advanced pipeline has 20+ nodes. I won't go through every single one (a lot of them are similar to the standard pipeline), but I'll cover the interesting parts.

### Graph variants

| Variant | Trust engine | Key difference |
|---------|-------------|----------------|
| `build_trust_engine_graph` | Full (7 nodes) | Exhaustive verification |
| `build_optimized_graph` | Batched (2 nodes) | 60% fewer LLM calls |

### Discovery phase (3 nodes)

This is where the advanced pipeline starts to diverge from the standard one. Instead of just checking if the query is ambiguous, it does full query analysis:

- **analyzer_node** -- intent classification, entity detection, temporal scope. Way more than the standard pipeline's ambiguity check.
- **discovery_node** -- entity disambiguation. If the query mentions "Apple", is it the company or the fruit? Does a quick search to resolve this before wasting the whole first iteration on the wrong thing.
- **confidence_check_node** -- gates the HITL flow. Only asks for clarification if discovery confidence is below 0.7. The optimized graph also classifies query complexity here for routing.

```python
def route_complexity_to_confidence(state: AgentState) -> str:
    if state.get("_fast_path", False):
        return "planner"
    discovery = state.get("discovery") or {}
    return "clarify" if discovery.get("confidence", 0) < 0.7 else "planner"
```

### Planning

The advanced planner produces a `ResearchTree` with priority-ranked question buckets (primary/secondary/tertiary) and typed `PlanQuery` objects, not just flat lists of strings. This lets the orchestrator make smarter decisions about what's most important.

In the optimized graph, there's complexity routing after planning:

```python
def route_planner_by_complexity(state: AgentState) -> str:
    return "fast_workers" if state.get("query_complexity") == "simple" else "orchestrator"
```

Simple queries skip multi-agent overhead entirely. This is the xAI-inspired bit.

### Iterative control (gap detection + backtracking)

The gap detector here is more sophisticated than the standard pipeline's version. It has three possible routing targets instead of two:
- **orchestrator** -- more research needed
- **backtrack** -- dead end detected, need a totally different approach
- **reduce** -- we're done, move to trust engine

The `backtrack_handler_node` records failed search paths as dead ends and reformulates. When we loop back to the orchestrator, it sees the dead ends and avoids repeating them. This is the OpenAI-inspired pattern.

There's also `early_termination_check_node` in the optimized graph -- if confidence gain between iterations drops below 5%, stop. No point burning more API calls if we're not finding anything new.

### Trust engine (the big one)

This is what separates the advanced pipeline from the standard one. Two variants:

**Full trust engine (7 sequential nodes):**

```
credibility -> ranker -> claims -> cite -> span_verify -> cross_validate -> confidence_score -> write
```

| Node | What it does | LLM calls |
|------|-------------|-----------|
| `credibility_scorer_node` | E-E-A-T scoring per source | 1 |
| `ranker_node` | Sort by credibility | 0 |
| `claims_node` | Extract atomic factual claims | 1 |
| `cite_node` | Link claims to supporting evidence | 1 |
| `span_verify_node` | Semantic match: does the source actually say what the claim says? | 1 |
| `cross_validate_node` | Flag claims with only 1 source vs 2+ | 1 |
| `claim_confidence_scorer_node` | Aggregate all signals into per-claim confidence | 1 |

That's 6 LLM calls just for trust (ranker is deterministic). On a query with 20 sources, it adds up.

**Batched trust engine (2 nodes):**

```
batched_cred_claims -> ranker -> batched_verify -> write
```

| Node | What it combines | LLM calls |
|------|-----------------|-----------|
| `batched_credibility_claims_node` | Credibility + claim extraction in one prompt | 1 |
| `batched_verification_node` | Span verify + cross-validate + confidence in one prompt | 1 |

Same logic, roughly 60% fewer API calls. The batched prompts are longer and more complex so the LLM occasionally misses things the dedicated nodes would catch, but for most queries the quality difference is small.

### Config approach

The advanced pipeline uses a frozen dataclass instead of `RunnableConfig`:

```python
@dataclass(frozen=True)
class ResearchConfig:
    max_rounds: int = 2
    queries_per_round: int = 10
    model_routing: Tuple[Tuple[str, str], ...] = (
        ("analyzer", "gpt-4o-mini"),
        ("planner", "gpt-4o"),
        ("writer", "gpt-4o"),
    )
    trusted_domains: tuple = (".edu", ".gov", "arxiv.org", ...)
    low_trust_domains: tuple = ("pinterest.com", "quora.com", ...)
    span_match_threshold: float = 0.6
    # ... ~50 tunables total
```

`frozen=True` so nodes can't accidentally mutate config mid-run. `get_model_for_node()` handles the routing lookup. The downside is it doesn't integrate as cleanly with Studio's auto-rendered knobs -- that's something I'd fix if I had more time.

---

## Standard vs Advanced comparison

### Architecture

| Dimension | Standard | Advanced |
|-----------|----------|----------|
| **Total nodes** | 11 | 20+ (across 12 modules) |
| **State fields** | ~20 | ~60 (with 25+ typed sub-dicts) |
| **Graph variants** | 1 (`build_graph`) | 3 (`trust_engine`, `optimized`, `simple`) |
| **Config** | `RunnableConfig` (Studio knobs) | Frozen dataclass (~50 tunables) |
| **Model routing** | 2 models (smart + fast) | 9 node-level assignments |

### Flow

| Phase | Standard | Advanced |
|-------|----------|----------|
| **Query understanding** | 7-dimension ambiguity check | Full NLP: intent, entities, temporal scope, complexity |
| **Entity disambiguation** | None | Discovery search + candidate clustering + confidence gating |
| **HITL trigger** | `is_ambiguous == True` | `discovery_confidence < 0.7` |
| **Planning** | Flat list of questions + searches | Priority-ranked ResearchTree |
| **Complexity routing** | None (mode set by caller) | Keyword + word-count heuristic |
| **Gap detection** | LLM-based coverage assessment | Hybrid heuristic + LLM with dead-end tracking |
| **Backtracking** | None (just loops or stops) | Dead-end categorization + alternative query generation |
| **Source scoring** | None | E-E-A-T: domain trust + freshness + authority |
| **Claim verification** | Bulk verify (supported/unsupported) | Span-level semantic match (0.6+ threshold) |
| **Cross-validation** | None | Flags claims with 2+ independent sources |

### LLM calls per run (typical query, single iteration)

| Step | Standard (single) | Standard (multi, 4 workers) | Advanced (full trust) | Advanced (batched) |
|------|------|------|------|------|
| Query understanding | 1 | 1 | 2 | 2 |
| Planning | 1 | 1 | 1 | 1 |
| Orchestration | 0 | 1 | 1 | 1 |
| Search + extraction | 4-8 | 8-16 | 8-16 | 8-16 |
| Gap detection | 1 | 0 | 1 | 1 |
| Synthesis | 0 | 1 | 1 | 1 |
| Verification | 1 | 1 | 0 | 0 |
| **Trust engine** | **0** | **0** | **6** | **2** |
| Report writing | 1 | 1 | 1 | 1 |
| **Total** | **~9-12** | **~13-21** | **~21-28** | **~17-23** |

### LangGraph features

| Feature | Standard | Advanced |
|---------|----------|----------|
| `StateGraph` | Yes | Yes |
| `MessagesState` | Yes | Yes |
| `operator.add` accumulators | 3 fields | 5 fields |
| `Send()` fan-out | 1 point | 2 points |
| `interrupt_before` HITL | Yes | Yes |
| `MemorySaver` | Yes | Yes |
| Conditional edges | 5 routing functions | 10+ |
| `RunnableConfig` / `context_schema` | Yes (Studio knobs) | No (frozen dataclass) |
| Graph cycles | 2 loops | 3 loops |
| Graph builders | 1 | 6 |

### When to use which

| Scenario | Use | Why |
|----------|-----|-----|
| Quick demo | Standard (multi) | Fast, simple, good enough |
| Narrow factual query | Standard (single) | Iterative depth beats parallel breadth |
| Broad multi-topic query | Standard (multi) or Advanced | Parallelism helps |
| Medical / legal / high-stakes | Advanced (full trust) | Source credibility matters |
| Cost-sensitive production | Advanced (batched) | 60% fewer trust engine calls |
| Automated (no user) | Either with `_simple` builder | Skip HITL overhead |

---

## Things I'd do differently

**Citation numbering.** The writer gets evidence labeled with source IDs and a valid citation list. But the LLM still sometimes makes up citation numbers. I fixed it by injecting `VALID CITATION NUMBERS: [1, 3, 5, 7]` into the prompt, but the real fix would be post-processing the report to rewrite any invalid `[N]` markers. The eval suite catches this which is how I found it in the first place.

**Search dedup.** Standard pipeline tracks `searches_done` with exact string matching. "quantum computing applications" and "applications of quantum computing" are treated as different queries. The advanced pipeline does embedding-based similarity (0.85 threshold) which is way better. Should have added that to the standard pipeline too.

**Config approach.** Standard uses `RunnableConfig["configurable"]` which integrates great with Studio but isn't type-checked. Advanced uses `ResearchConfig` (frozen dataclass) which gives IDE autocomplete but doesn't render in Studio. Ideally I'd want both -- typed config that also integrates with Studio's schema.

**Trust engine cost.** On a broad query with 20 sources the full trust engine makes 6+ LLM calls just for verification. Smarter approach: only verify claims that actually make it into the final report, not every claim from every source. Would probably cut trust engine costs by 60-70%.

**Early termination.** Currently checks if confidence gain between iterations is below 5%. Should also look at how many new unique sources we're finding -- if we keep seeing the same URLs, there's no point continuing.

---

## State design

Both pipelines follow the same idea: state is the single source of truth. Every node reads from it and writes back to it. No hidden globals, no message passing between nodes.

The standard pipeline has ~20 fields. The advanced one has ~60, organized into typed sub-structures:

```python
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

It's verbose but every piece of data has a clear schema. When a node writes `evidence`, you know exactly what fields each item has. The alternative (just using `Dict[str, Any]` everywhere) would be shorter but you'd hide bugs until runtime. I've been burned by that before.

---

## Eval suite

Three layers, all in `tests/`. Designed to run standalone without LangSmith.

### Structural evaluators

10 deterministic checks against pipeline output:

| Check | What it catches |
|-------|----------------|
| `check_report_has_markdown_structure` | Missing title, sections, or sources |
| `check_citations_present` | No `[N]` markers in report |
| `check_citations_map_to_sources` | Citation `[19]` but only 12 sources |
| `check_metadata_complete` | Missing keys, wrong types |
| `check_evidence_populated` | Empty evidence list |
| `check_scores_in_bounds` | Coverage or confidence outside [0,1] |
| `check_report_minimum_length` | Report under 200 words |
| `check_evidence_fields` | Evidence missing `fact`, `source_url`, etc. |
| `check_sources_have_urls` | Sources missing URLs or titles |
| `check_no_empty_report` | Empty/whitespace-only report |

These are fast and catch regressions. The citation mapping check is what originally caught the citation numbering bug.

### LLM-as-judge evaluators

Five evaluators using Pydantic structured output (via `ChatOpenAI` directly, not my LLMWrapper, because I need `with_structured_output`):

| Evaluator | What it measures | Output |
|-----------|-----------------|--------|
| **Relevance** | Does the report answer the query? | 1-5 score |
| **Groundedness** | Every factual claim checked against evidence | grounded/total ratio |
| **Completeness** | Report vs research plan coverage | 1-5 score |
| **Quality** | 5 sub-scores: depth, source diversity, rigor, clarity, citations | Average of 1-5s |
| **Citation faithfulness** | Do citations accurately represent their sources? | 1-5 score |

Groundedness is the most useful one. It extracts every factual claim from the report and checks each against the pipeline's actual evidence. Gives you a concrete ratio like "23/27 claims grounded" instead of a vibes-based score.

### Behavioral evaluators

These test whether design features actually work, not just whether the output looks good:

| Test | What it validates |
|------|------------------|
| Config effectiveness | `detailed` vs `bullet_points` produces different formats |
| Gap loop improvement | More iterations -> equal or more evidence |
| Mode comparison | Both modes produce valid reports |
| Ambiguity detection | Known-ambiguous queries trigger `is_ambiguous=True` |
| State consistency | `metadata.sources_count` matches `len(state["sources"])` |
| Iteration convergence | Coverage is monotonically non-decreasing |
| Verified claims ratio | `confidence` matches actual supported/total |

### Eval inspiration

Criteria inspired by [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) which uses LLM-as-judge evaluators for relevance, structure, correctness, groundedness, completeness, quality.

Key differences from theirs:
- **No LangSmith dependency** -- runs standalone with pytest or `run_eval.py`
- **Behavioral tests** -- config effectiveness, gap loop verification, mode comparison. These test the architecture, not just output quality. The reference repo doesn't do these.
- **Groundedness against pipeline evidence** -- theirs checks against `raw_notes`, mine checks against `state["evidence"]`, the actual structured evidence the pipeline collected

---

## Running it

### Research

```bash
# Multi-agent (default)
python -m src.run "What is quantum computing?"

# Single-agent with gap detection loop
python -m src.run --single-agent "Compare React vs Vue"

# Run both and compare metrics
python -m src.run --compare "Latest AI developments"

# Custom config
python -m src.run --model gpt-4o-mini --format concise --iterations 3 "Explain CRISPR"

# Custom system prompt
python -m src.run --system-prompt "Write for a technical audience" "Rust vs Go"
```

### Evaluation

```bash
# Fast offline check (no API calls)
python -m pytest tests/test_structural.py tests/test_behavioral.py -m "not live" -v

# Full live run
python -m tests.run_eval --live --max-iterations 3 --output results.json

# Re-score saved snapshots without re-running pipeline
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
│   │   ├── prompts.py          # All LLM prompts
│   │   ├── nodes.py            # 11 node functions
│   │   └── graph.py            # StateGraph wiring, routing, build_graph()
│   ├── advanced/               # Advanced pipeline (~4,000 LOC)
│   │   ├── config.py           # ResearchConfig frozen dataclass
│   │   ├── state.py            # AgentState + 25+ TypedDicts
│   │   ├── graph.py            # 3 graph variants x 3 builder helpers
│   │   └── nodes/              # 13 node modules
│   ├── tools/                  # Tavily search, HTTP fetch
│   ├── utils/                  # LLM wrappers, caching, JSON parsing
│   ├── run.py                  # CLI for standard pipeline
│   └── run_advanced_trust_engine.py  # CLI for advanced pipeline
├── tests/
│   ├── evaluators/             # structural, llm_judge, behavioral
│   ├── eval_cases.py           # 8 test case definitions
│   ├── conftest.py             # Fixtures, pipeline runner, mock state
│   ├── run_eval.py             # Main eval runner
│   └── test_*.py               # pytest wrappers
├── langgraph.json              # LangGraph Studio registration
└── pyproject.toml
```

---

## Sources

- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)
- [How Deep Research Works (PromptLayer)](https://blog.promptlayer.com/how-deep-research-works/)
- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [xAI Grok 3](https://x.ai/news/grok-3)
- [Google Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)
- [Perplexity Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)
- [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) -- evaluation criteria
- [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher) -- reference implementation


## AI Tools Used
- Claude Code (mostly for code completion, debugging, deep research and writing docs(specifically the table and code snippet parts of it))
