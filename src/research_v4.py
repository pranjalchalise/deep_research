from __future__ import annotations

import json
import os
import re
import operator
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from typing_extensions import Annotated

import httpx
import trafilatura

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send


# -----------------------------
# Config
# -----------------------------
FAST_MODE = True  # ✅ True = use Tavily snippets only (fast). False = scrape pages (slow).

MAX_SOURCES = 5
MAX_RESULTS_PER_QUERY = 3
MAX_FETCH_CHARS_PER_SOURCE = 5000

FOLLOWUP_SEARCH_ENABLED = False  # keep false for now; turn on later if you want.
MAX_FOLLOWUP_QUERIES = 3

HARD_BLOCKLIST_DOMAINS = {"oreateai.com"}
SOFT_DOWNRANK_DOMAINS = {
    "reddit.com",
    "medium.com",
    "substack.com",
    "linkgo.dev",
    "latenode.com",
    "sourceforge.net",
    "techaheadcorp.com",
    "softaims.com",
}

SKIP_FETCH_DOMAINS = {"youtube.com", "linkedin.com", "reddit.com"}


# -----------------------------
# Types
# -----------------------------
class PlanQuery(TypedDict):
    qid: str
    query: str


class Plan(TypedDict):
    topic: str
    queries: List[PlanQuery]


class RawSource(TypedDict):
    url: str
    title: str
    snippet: str


class FetchedSource(TypedDict):
    sid: str
    url: str
    title: str
    text: str


# -----------------------------
# State
# -----------------------------
class AgentState(MessagesState):
    plan: Optional[Plan]
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]
    raw_sources: Annotated[List[RawSource], operator.add]
    sources: Optional[List[RawSource]]
    fetched: Optional[List[FetchedSource]]


# -----------------------------
# Helpers
# -----------------------------
def _safe_json_loads(txt: str) -> Any:
    txt = (txt or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(txt[start : end + 1])


def _domain(url: str) -> str:
    try:
        u = url.replace("https://", "").replace("http://", "")
        return u.split("/")[0].lower()
    except Exception:
        return url.lower()


def _soft_penalty(url: str) -> int:
    d = _domain(url)
    return 1 if any(x in d for x in SOFT_DOWNRANK_DOMAINS) else 0


def _clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t).strip()
    return t


def should_skip_url(url: str) -> bool:
    d = _domain(url)
    if any(bad in d for bad in HARD_BLOCKLIST_DOMAINS):
        return True
    return False


def fetch_readable_text(url: str, timeout_s: float = 8.0) -> str:
    """
    Slow mode: download HTML and extract main text with trafilatura.
    Returns empty string on failure.
    """
    try:
        timeout = httpx.Timeout(timeout_s, connect=3.0, read=timeout_s)
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "research-studio/0.1 (+langgraph takehome)"},
        ) as client:
            r = client.get(url)
            if r.status_code >= 400:
                return ""
            html = r.text

        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        return extracted or ""
    except Exception:
        return ""


# -----------------------------
# Nodes
# -----------------------------
def planner_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a research planner. Return ONLY valid JSON:\n"
            '{ "topic": "...", "queries": [{"qid":"Q1","query":"..."}, ...] }\n'
            "Constraints:\n"
            "- 4 to 6 queries\n"
            "- qid must be Q1..Qn\n"
            "- queries must cover distinct aspects needed to answer the user\n"
            "- prefer queries that find primary sources (docs, vendor sites, GitHub)\n"
            "- no extra text\n"
        )
    )

    resp = llm.invoke([system, HumanMessage(content=f"User question:\n{user_q}")])
    plan: Plan = _safe_json_loads(resp.content)

    total = len(plan.get("queries", []))
    print(f"[planner] planned {total} queries")
    return {
        "plan": plan,
        "total_workers": total,
        "done_workers": 0,
        "raw_sources": [],
        "sources": None,
        "fetched": None,
    }


def worker_node(state: AgentState) -> Dict[str, Any]:
    qi = state["query_item"]
    query = qi["query"]
    print(f"[search] {qi.get('qid','?')}: {query}")

    search = TavilySearch(max_results=MAX_RESULTS_PER_QUERY, topic="general")
    raw = search.invoke({"query": query})

    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    out: List[RawSource] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        if should_skip_url(url):
            continue

        title = str(r.get("title", "")).strip() or url
        snippet = str(r.get("content", "") or r.get("snippet", "") or "").strip()
        out.append({"url": url, "title": title, "snippet": snippet[:700]})

    return {"raw_sources": out, "done_workers": 1}


def reducer_node(state: AgentState) -> Dict[str, Any]:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    if done < total:
        return {}

    # Dedup by URL
    seen: Dict[str, RawSource] = {}
    for s in state.get("raw_sources") or []:
        url = s["url"]
        if url not in seen:
            seen[url] = s

    sources = list(seen.values())
    print(f"[reduce] got {len(sources)} unique sources")
    return {"sources": sources}


def rank_sources_node(state: AgentState) -> Dict[str, Any]:
    sources = state.get("sources") or []
    if not sources:
        return {}

    # Heuristic pre-score to reduce LLM load
    scored = []
    for s in sources:
        url = s["url"]
        snippet_len = len(s.get("snippet", "") or "")
        penalty = _soft_penalty(url)
        score = snippet_len - (penalty * 250)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored[: min(len(scored), 25)]]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    system = SystemMessage(
        content=(
            "Rank sources for credibility + usefulness.\n"
            "Return ONLY JSON: {\"order\": [0,1,2,...]}\n"
            "Rank higher:\n"
            "- official docs, org sites, GitHub, reputable publishers\n"
            "Rank lower:\n"
            "- forums, reddit, low-quality SEO blogs\n"
        )
    )

    items = []
    for i, s in enumerate(candidates):
        items.append(f"{i}. {s['title']}\nURL: {s['url']}\nSNIPPET: {(s.get('snippet','')[:240])}")
    resp = llm.invoke([system, HumanMessage(content="\n\n".join(items))])

    data = _safe_json_loads(resp.content)
    order = data.get("order", [])

    ranked = []
    used = set()
    for idx in order:
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in used:
            ranked.append(candidates[idx])
            used.add(idx)

    if not ranked:
        ranked = candidates

    ranked = ranked[:MAX_SOURCES]
    print(f"[rank] selected {len(ranked)} sources")
    return {"sources": ranked}


def fetch_sources_node(state: AgentState) -> Dict[str, Any]:
    sources = state.get("sources") or []
    if not sources:
        return {"fetched": []}

    filtered = [s for s in sources if not should_skip_url(s["url"])]
    filtered = filtered[:MAX_SOURCES]
    print(f"[fetch] preparing {len(filtered)} sources (FAST_MODE={FAST_MODE})")

    fetched: List[FetchedSource] = []

    if FAST_MODE:
        # ✅ Fast: use snippets only
        for i, s in enumerate(filtered, start=1):
            url = s["url"]
            title = s.get("title", "") or url
            txt = _clean_text(s.get("snippet", ""))
            if not txt:
                continue
            fetched.append(
                {
                    "sid": f"S{i}",
                    "url": url,
                    "title": title,
                    "text": txt[:MAX_FETCH_CHARS_PER_SOURCE],
                }
            )
        return {"fetched": fetched}

    # Slow mode: full fetch + extract with concurrency
    import concurrent.futures

    def should_skip_fetch(url: str) -> bool:
        d = _domain(url)
        return any(x in d for x in SKIP_FETCH_DOMAINS)

    def fetch_one(i_s):
        i, s = i_s
        url = s["url"]
        title = s.get("title", "") or url
        if should_skip_fetch(url):
            txt = _clean_text(s.get("snippet", ""))
        else:
            txt = _clean_text(fetch_readable_text(url)) or _clean_text(s.get("snippet", ""))
        if not txt:
            return None
        return {
            "sid": f"S{i}",
            "url": url,
            "title": title,
            "text": txt[:MAX_FETCH_CHARS_PER_SOURCE],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for item in ex.map(fetch_one, list(enumerate(filtered, start=1))):
            if item:
                fetched.append(item)

    return {"fetched": fetched}


def writer_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    plan = state.get("plan") or {"topic": "Research Report", "queries": []}
    fetched = state.get("fetched") or []

    sources_text = "\n\n".join(
        [f"[{s['sid']}] {s['title']}\nURL: {s['url']}\nTEXT: {s['text']}" for s in fetched]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a careful researcher.\n"
            "Write a concise, high-quality report answering the user's question using ONLY the provided sources.\n"
            "Rules:\n"
            "- Every paragraph must include at least one citation like [S1].\n"
            "- Do not invent facts not supported by the sources.\n"
            "- If the sources are insufficient for a point, explicitly say so.\n"
            "- End with a Sources section listing [S#] Title — URL.\n"
        )
    )

    print("[writer] writing report")
    resp = llm.invoke(
        [
            system,
            HumanMessage(
                content=(
                    f"User question:\n{user_q}\n\n"
                    f"Topic: {plan.get('topic')}\n\n"
                    f"Sources:\n{sources_text}"
                )
            ),
        ]
    )

    report = resp.content.strip()
    return {"messages": [AIMessage(content=report)]}


# -----------------------------
# Graph wiring
# -----------------------------
def fanout(state: AgentState):
    plan = state.get("plan") or {}
    qs = plan.get("queries") or []
    return [Send("worker", {"query_item": qi}) for qi in qs]


def route_after_reduce(state: AgentState) -> str:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "rank_sources" if done >= total else END


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)
    g.add_node("rank_sources", rank_sources_node)
    g.add_node("fetch", fetch_sources_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fanout)

    g.add_edge("worker", "reduce")
    g.add_conditional_edges("reduce", route_after_reduce)

    g.add_edge("rank_sources", "fetch")
    g.add_edge("fetch", "writer")
    g.add_edge("writer", END)

    return g.compile()


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    app = build_graph()
    out = app.invoke({"messages": [HumanMessage(content="What is LangGraph and what is it used for?")]})
    print("\n" + out["messages"][-1].content)
