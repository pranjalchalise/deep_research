# Research Studio

A LangGraph-powered deep research agent that accepts a query, searches the web, and returns a grounded report with citations.

Two pipelines are included:

- **Standard pipeline** (`src/pipeline/`) -- clean, self-contained research graph with single-agent and multi-agent modes.
- **Advanced pipeline** (`src/advanced/`) -- adds a trust engine (source credibility scoring, span verification, cross-validation, confidence indicators) on top of the standard flow.

Both pipelines use [Tavily](https://tavily.com/) for web search and OpenAI models for planning, extraction, and report generation.

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd research-studio
pip install -e ".[extract]"
```

The `[extract]` extra installs `trafilatura` and `beautifulsoup4` for better HTML text extraction. The agent works without them (falls back to regex), but results are noticeably better with them.

### 2. Environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY` -- for LLM calls (GPT-4o and GPT-4o-mini)
- `TAVILY_API_KEY` -- for web search

Optional:
- `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` -- enables LangSmith tracing

### 3. Run

**Standard pipeline** (recommended starting point):

```bash
# Multi-agent mode (default) -- parallel workers research sub-questions
python -m src.run "What is quantum computing?"

# Single-agent mode -- iterative search with gap detection
python -m src.run --single-agent "Compare React vs Vue"

# Compare both modes side-by-side
python -m src.run --compare "Latest AI developments"
```

**Advanced pipeline** (trust engine -- slower but more rigorous):

```bash
python -m src.run_advanced_trust_engine "Who is Satya Nadella?"
python -m src.run_advanced_trust_engine --single-agent "Compare Python vs JavaScript"
```

### 4. LangGraph Studio

Both graphs are registered in `langgraph.json`:

```bash
langgraph dev
```

This exposes two graphs:
- `pipeline` -- the standard research graph
- `advanced` -- the trust engine graph

## Architecture

### Standard Pipeline (`src/pipeline/`)

```
START -> understand -> [clarify?] -> plan
                                       |
                    +------------------+------------------+
                    |                                     |
              (single-agent)                       (multi-agent)
                    |                                     |
           search_and_extract                       orchestrate
                    |                                     |
             detect_gaps ←──loop──┐              fanout (Send) -> workers
                    |             |                        |
                    v             |                    collect
                  verify ←────────┘                       |
                    |                                synthesize ←──loop
                    v                                     |
              write_report                             verify
                    |                                     |
                   END                              write_report
                                                         |
                                                        END
```

**Key design choices:**
- `messages` field (via `MessagesState`) stores user query as `HumanMessage` and final report as `AIMessage`
- `operator.add` on `evidence`, `worker_results`, `done_workers` for safe parallel accumulation
- `interrupt_before=["clarify"]` for human-in-the-loop disambiguation
- Gap detection loop re-searches until coverage threshold is met or iteration cap is hit

### Advanced Pipeline (`src/advanced/`)

Adds on top of the standard flow:
- **Discovery phase**: entity detection, query classification, confidence-gated clarification
- **Trust engine** (7 nodes): credibility scoring -> ranking -> claim extraction -> citation mapping -> span verification -> cross-validation -> confidence scoring
- **Batched variant**: same trust pipeline in 2 LLM calls instead of 7 (60% fewer API calls)
- **Dead-end backtracking**: when a search path yields nothing, tries a different angle
- **Complexity routing**: simple queries skip multi-agent overhead

## Project Structure

```
research-studio/
├── src/
│   ├── run.py                          # CLI: standard pipeline
│   ├── run_advanced_trust_engine.py    # CLI: advanced pipeline
│   ├── pipeline/                       # Standard pipeline (self-contained)
│   │   ├── state.py                    #   ResearchState (extends MessagesState)
│   │   ├── prompts.py                  #   All LLM prompts
│   │   ├── nodes.py                    #   All node functions
│   │   └── graph.py                    #   build_graph(), routing, compiled graph
│   ├── advanced/                       # Advanced pipeline (self-contained)
│   │   ├── config.py                   #   ResearchConfig (model routing, thresholds)
│   │   ├── state.py                    #   AgentState (extends MessagesState)
│   │   ├── graph.py                    #   Graph builders, compiled graph
│   │   └── nodes/                      #   13 node modules
│   ├── tools/                          # Shared: Tavily search, HTTP fetch
│   └── utils/                          # Shared: LLM wrappers, caching, JSON parsing
├── langgraph.json                      # LangGraph Studio config
├── pyproject.toml                      # Dependencies
└── .env.example                        # Required environment variables
```

## Configuration

### Standard pipeline (CLI flags)

The standard pipeline accepts configuration via CLI flags or `config={"configurable": {...}}`:

| Flag | Config key | Default | Description |
|------|------------|---------|-------------|
| `--model` | `model` | `gpt-4o` | LLM for planning, synthesis, writing |
| `--fast-model` | `fast_model` | `gpt-4o-mini` | LLM for bulk extraction |
| `--max-results` | `max_search_results` | `5` | Tavily results per query |
| `--format` | `report_structure` | `detailed` | `detailed`, `concise`, or `bullet_points` |
| `--system-prompt` | `system_prompt` | *(none)* | Custom instructions prepended to the report writer |
| `--iterations` | `max_iterations` | `5` (single) / `2` (multi) | Max research loop iterations (state field) |

Examples:

```bash
# Use a cheaper model for the whole pipeline
python -m src.run --model gpt-4o-mini "Explain CRISPR"

# Get a bullet-point summary instead of a full report
python -m src.run --format bullet_points "Latest AI developments"

# Custom writing instructions
python -m src.run --system-prompt "Write for a technical audience" "Rust vs Go"

# Fewer search results for faster/cheaper runs
python -m src.run --max-results 3 --format concise "What is quantum computing?"
```

When invoking the graph programmatically, pass config keys via `configurable`:

```python
from src.pipeline.graph import build_graph

graph = build_graph()
result = graph.invoke(
    {
        "query": "Compare React and Vue",
        "messages": [HumanMessage(content="Compare React and Vue")],
        "mode": "multi",
        # ... other required state fields
    },
    config={"configurable": {
        "model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "max_search_results": 5,
        "report_structure": "detailed",
        "system_prompt": "Focus on developer experience",
    }},
)
```

### Advanced pipeline (`ResearchConfig`)

The advanced pipeline exposes `ResearchConfig` with tunables for:
- Model routing per node (which model handles planning vs extraction vs writing)
- Iteration limits and confidence thresholds
- Source credibility thresholds
- Query deduplication sensitivity
- Cost budgets for early termination
- Cache settings
