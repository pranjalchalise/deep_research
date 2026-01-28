from __future__ import annotations

import json
import os
import re
import operator
from typing import Any, Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from typing_extensions import Annotated

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send


# -----------------------------
# Config
# -----------------------------
MODEL = os.getenv("RESEARCH_MODEL", "gpt-4o-mini")

MAX_ROUNDS = int(os.getenv("RESEARCH_MAX_ROUNDS", "2"))  # 0 + 1 => 2 rounds total
INITIAL_NUM_QUERIES = int(os.getenv("RESEARCH_INITIAL_QUERIES", "5"))
REFINE_NUM_QUERIES = int(os.getenv("RESEARCH_REFINE_QUERIES", "3"))

MAX_RESULTS_PER_QUERY = int(os.getenv("RESEARCH_RESULTS_PER_QUERY", "5"))
MAX_SOURCES_AFTER_RANK = int(os.getenv("RESEARCH_MAX_SOURCES", "7"))
SNIPPET_CHARS = int(os.getenv("RESEARCH_SNIPPET_CHARS", "900"))


HIGH_TRUST_DOMAINS = {
    "docs.langchain.com",
    "langchain.com",
    "github.com",
    "langchain-ai.github.io",
    "reference.langchain.com",
}

SOFT_DOWNRANK_DOMAINS = {
    "reddit.com",
    "medium.com",
    "substack.com",
    "linkgo.dev",
    "latenode.com",
    "sourceforge.net",
    "techaheadcorp.com",
    "softaims.com",
    "analyticsvidhya.com",
    "geeksforgeeks.org",
}

HARD_BLOCKLIST_DOMAINS = {"oreateai.com"}


# -----------------------------
# Types
# -----------------------------
class PlanQuery(TypedDict):
    qid: str
    query: str


class Plan(TypedDict):
    topic: str
    queries: List[PlanQuery]
    strategy: str  # "official_langchain" or "general_web"


class RawSource(TypedDict):
    url: str
    title: str
    snippet: str


class EvidenceClaim(TypedDict):
    claim: str
    support: str
    sid: str


class EvidenceBundle(TypedDict):
    sid: str
    url: str
    title: str
    claims: List[EvidenceClaim]


class CoverageVerdict(TypedDict):
    coverage_ok: bool
    gaps: List[str]


# -----------------------------
# State
# -----------------------------
class AgentState(MessagesState):
    plan: Optional[Plan]

    round: int
    max_rounds: int

    queries_to_run: Optional[List[PlanQuery]]
    total_workers: int

    completed: Annotated[List[str], operator.add]
    raw_sources: Annotated[List[RawSource], operator.add]

    sources: Optional[List[RawSource]]
    ranked_sources: Optional[List[RawSource]]
    evidence: Optional[List[EvidenceBundle]]

    coverage_ok: Optional[bool]
    gaps: Optional[List[str]]


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
    u = (url or "").strip()
    u = u.replace("https://", "").replace("http://", "")
    return u.split("/")[0].lower()


def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _soft_penalty(url: str) -> int:
    d = _domain(url)
    return 1 if any(x in d for x in SOFT_DOWNRANK_DOMAINS) else 0


def _trust_boost(url: str) -> int:
    d = _domain(url)
    if "github.com/langchain-ai/langgraph" in url:
        return 2500
    if any(ht in d for ht in HIGH_TRUST_DOMAINS):
        return 900
    return 0


def should_skip_url(url: str) -> bool:
    d = _domain(url)
    return any(bad in d for bad in HARD_BLOCKLIST_DOMAINS)


def done_count_for_round(state: AgentState) -> int:
    r = state.get("round", 0)
    tag = f"{r}:"
    return sum(1 for x in (state.get("completed") or []) if isinstance(x, str) and x.startswith(tag))


# -----------------------------
# Nodes
# -----------------------------
def planner_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content

    llm = ChatOpenAI(model=MODEL, temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a research planner.\n"
            "Return ONLY valid JSON:\n"
            '{ "topic": "...", "strategy": "official_langchain"|"general_web", '
            '"queries": [{"qid":"Q1","query":"..."}, ...] }\n'
            "\n"
            f"Constraints:\n"
            f"- {INITIAL_NUM_QUERIES} queries exactly\n"
            "- qid must be Q1..Qn\n"
            "\n"
            "Decide strategy:\n"
            "- If the user question is about LangGraph/LangChain or LangSmith, use strategy=official_langchain.\n"
            "- Otherwise use strategy=general_web.\n"
            "\n"
            "If strategy=official_langchain, heavily use:\n"
            "- site:docs.langchain.com\n"
            "- site:reference.langchain.com\n"
            "- site:langchain.com\n"
            "- site:github.com/langchain-ai/langgraph\n"
            "- site:langchain-ai.github.io\n"
            "\n"
            "If strategy=general_web, DO NOT use site: restrictions.\n"
            "Instead write diverse, specific queries.\n"
            "\n"
            "No extra text."
        )
    )

    resp = llm.invoke([system, HumanMessage(content=f"User question:\n{user_q}")])
    plan: Plan = _safe_json_loads(resp.content)

    qs = plan.get("queries", [])[:INITIAL_NUM_QUERIES]
    total = len(qs)

    print(f"[planner] strategy={plan.get('strategy')} round=0 planned {total} queries")
    return {
        "plan": plan,
        "round": 0,
        "max_rounds": MAX_ROUNDS,
        "queries_to_run": qs,
        "total_workers": total,
        "sources": None,
        "ranked_sources": None,
        "evidence": None,
        "coverage_ok": None,
        "gaps": None,
    }


def worker_node(state: AgentState) -> Dict[str, Any]:
    qi = state["query_item"]
    qid = qi.get("qid", "?")
    query = qi.get("query", "")

    r = state.get("round", 0)
    print(f"[search] round={r} {qid}: {query}")

    search = TavilySearch(max_results=MAX_RESULTS_PER_QUERY, topic="general")
    raw = search.invoke({"query": query})

    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    out: List[RawSource] = []

    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url or should_skip_url(url):
            continue

        title = str(item.get("title", "")).strip() or url
        snippet = str(item.get("content", "") or item.get("snippet", "") or "").strip()
        if not snippet:
            snippet = title

        out.append({"url": url, "title": title, "snippet": snippet[:SNIPPET_CHARS]})

    return {
        "raw_sources": out,
        "completed": [f"{r}:{qid}"],
    }


def reducer_node(state: AgentState) -> Dict[str, Any]:
    done = done_count_for_round(state)
    total = state.get("total_workers", 0)

    if done < total:
        return {}

    seen: Dict[str, RawSource] = {}
    for s in state.get("raw_sources") or []:
        url = s["url"]
        if url not in seen:
            seen[url] = s

    deduped = list(seen.values())
    print(f"[reduce] round={state.get('round',0)} done={done}/{total} -> {len(deduped)} unique sources")
    return {"sources": deduped}


def rank_sources_node(state: AgentState) -> Dict[str, Any]:
    sources = state.get("sources") or []
    if not sources:
        print("[rank] no sources to rank")
        return {"ranked_sources": []}

    scored = []
    for s in sources:
        url = s["url"]
        snippet_len = len(s.get("snippet", "") or "")
        penalty = _soft_penalty(url)
        boost = _trust_boost(url)
        score = snippet_len + boost - (penalty * 450)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [x[1] for x in scored[:MAX_SOURCES_AFTER_RANK]]

    print(f"[rank] selected {len(ranked)} sources")
    return {"ranked_sources": ranked}


def extract_evidence_node(state: AgentState) -> Dict[str, Any]:
    ranked = state.get("ranked_sources") or []
    if not ranked:
        return {"evidence": []}

    llm = ChatOpenAI(model=MODEL, temperature=0.0)
    system = SystemMessage(
        content=(
            "You are an evidence extractor.\n"
            "Given a source title/url/snippet, extract 1-4 factual claims the source supports.\n"
            "For each claim, include a short support snippet based ONLY on the provided snippet.\n"
            "If snippet is weak, return empty claims.\n"
            "Output ONLY JSON: {\"claims\": [{\"claim\":\"...\",\"support\":\"...\"}, ...]}."
        )
    )

    bundles: List[EvidenceBundle] = []
    for i, s in enumerate(ranked, start=1):
        sid = f"S{i}"
        title = s.get("title", "") or s.get("url", "")
        url = s.get("url", "")
        snippet = _clean_text(s.get("snippet", ""))

        resp = llm.invoke(
            [
                system,
                HumanMessage(
                    content=(
                        f"Source ID: {sid}\nTitle: {title}\nURL: {url}\nSNIPPET:\n{snippet}\n"
                    )
                ),
            ]
        )
        data = _safe_json_loads(resp.content)

        claims: List[EvidenceClaim] = []
        for c in (data.get("claims", []) or [])[:4]:
            if not isinstance(c, dict):
                continue
            claim = _clean_text(str(c.get("claim", "")))
            support = _clean_text(str(c.get("support", "")))
            if claim and support:
                claims.append({"claim": claim, "support": support[:240], "sid": sid})

        bundles.append({"sid": sid, "url": url, "title": title, "claims": claims})

    total_claims = sum(len(b.get("claims", [])) for b in bundles)
    print(f"[evidence] extracted {total_claims} claims from {len(bundles)} sources")
    return {"evidence": bundles}


def assess_coverage_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    evidence = state.get("evidence") or []
    sources = state.get("ranked_sources") or []

    if not sources:
        # Force refine / fallback
        print("[assess] no sources found -> coverage_ok=False")
        return {"coverage_ok": False, "gaps": ["No sources found for the current query strategy. Expand search scope."]}

    lines = []
    for b in evidence:
        sid = b["sid"]
        for c in b.get("claims", [])[:3]:
            lines.append(f"- [{sid}] {c['claim']} (support: {c['support']})")
    evidence_text = "\n".join(lines[:60])

    llm = ChatOpenAI(model=MODEL, temperature=0.0)
    system = SystemMessage(
        content=(
            "You are a research coverage judge.\n"
            "Return ONLY JSON:\n"
            "{ \"coverage_ok\": true/false, \"gaps\": [\"...\", ...] }\n"
            "If evidence is insufficient, set coverage_ok=false and provide specific gaps."
        )
    )
    resp = llm.invoke([system, HumanMessage(content=f"User question:\n{user_q}\n\nEvidence:\n{evidence_text}")])
    verdict: CoverageVerdict = _safe_json_loads(resp.content)

    ok = bool(verdict.get("coverage_ok", False))
    gaps = verdict.get("gaps", []) if isinstance(verdict.get("gaps", []), list) else []
    gaps = [str(g).strip() for g in gaps if str(g).strip()][:6]

    print(f"[assess] coverage_ok={ok} gaps={len(gaps)} round={state.get('round',0)}")
    return {"coverage_ok": ok, "gaps": gaps}


def refine_queries_node(state: AgentState) -> Dict[str, Any]:
    r = state.get("round", 0)
    max_r = state.get("max_rounds", MAX_ROUNDS)

    if state.get("coverage_ok") is True:
        return {}
    if r + 1 >= max_r:
        print("[refine] no rounds left")
        return {}

    plan = state.get("plan") or {"strategy": "general_web"}
    strategy = plan.get("strategy", "general_web")
    user_q = state["messages"][-1].content
    gaps = state.get("gaps") or []

    # Fallback: if official_langchain got 0 sources, switch to general_web for refinements
    if "No sources found" in " ".join(gaps) and strategy == "official_langchain":
        strategy = "general_web"
        print("[refine] switching strategy to general_web due to zero sources")

    llm = ChatOpenAI(model=MODEL, temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a query refiner.\n"
            "Return ONLY JSON: {\"queries\": [\"...\", ...]}.\n"
            f"- {REFINE_NUM_QUERIES} queries exactly\n"
            "\n"
            "If strategy=official_langchain, use site: constraints to LangChain/LangGraph official sources.\n"
            "If strategy=general_web, do NOT use site: constraints.\n"
            "No extra text."
        )
    )
    resp = llm.invoke(
        [
            system,
            HumanMessage(
                content=(
                    f"strategy={strategy}\n"
                    f"User question:\n{user_q}\n\n"
                    f"Gaps:\n" + "\n".join([f"- {g}" for g in gaps])
                )
            ),
        ]
    )
    data = _safe_json_loads(resp.content)
    qs = data.get("queries", [])
    if not isinstance(qs, list):
        qs = []
    qs = [str(x).strip() for x in qs if str(x).strip()][:REFINE_NUM_QUERIES]

    new_round = r + 1
    refined: List[PlanQuery] = []
    for i, q in enumerate(qs, start=1):
        refined.append({"qid": f"R{new_round}Q{i}", "query": q})

    print(f"[refine] round={new_round} planned {len(refined)} refined queries")
    return {
        "round": new_round,
        "queries_to_run": refined,
        "total_workers": len(refined),
        "sources": None,
        "ranked_sources": None,
        "evidence": None,
        "coverage_ok": None,
        "gaps": None,
    }


def writer_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    evidence = state.get("evidence") or []
    ranked = state.get("ranked_sources") or []

    # Build evidence text for writer
    evidence_lines = []
    for b in evidence:
        sid = b["sid"]
        evidence_lines.append(f"[{sid}] {b['title']} — {b['url']}")
        for c in b.get("claims", []):
            evidence_lines.append(f"  - CLAIM: {c['claim']}")
            evidence_lines.append(f"    SUPPORT: {c['support']}")
        evidence_lines.append("")

    evidence_text = "\n".join(evidence_lines).strip()

    # If still nothing, produce a clean, honest response
    if not ranked:
        msg = (
            "I couldn't find any web sources for that query with the current research workflow.\n"
            "Try a query that points to public pages (LinkedIn/GitHub/personal site) or ask about a different topic."
        )
        return {"messages": [AIMessage(content=msg)]}

    llm = ChatOpenAI(model=MODEL, temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a careful deep-research report writer.\n"
            "Answer using ONLY the evidence. If evidence is insufficient, say so and explain what is missing.\n"
            "Rules:\n"
            "- Every paragraph must include at least one citation like [S1].\n"
            "- End with Sources list.\n"
        )
    )

    print("[writer] writing final report")
    resp = llm.invoke(
        [
            system,
            HumanMessage(
                content=(
                    f"User question:\n{user_q}\n\n"
                    f"EVIDENCE:\n{evidence_text}"
                )
            ),
        ]
    )
    return {"messages": [AIMessage(content=resp.content.strip())]}


# -----------------------------
# Routing
# -----------------------------
def fanout(state: AgentState):
    qs = state.get("queries_to_run") or []
    return [Send("worker", {"query_item": qi}) for qi in qs]


def route_after_reduce(state: AgentState) -> Literal["rank", "writer"]:
    # Once round is done, always go forward; rank handles empty sources
    done = done_count_for_round(state)
    total = state.get("total_workers", 0)
    return "rank" if done >= total else "writer"


def route_after_assess(state: AgentState) -> Literal["writer", "refine"]:
    ok = state.get("coverage_ok")
    r = state.get("round", 0)
    max_r = state.get("max_rounds", MAX_ROUNDS)
    if ok is True:
        return "writer"
    if r + 1 >= max_r:
        return "writer"
    return "refine"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    g.add_node("rank", rank_sources_node)
    g.add_node("evidence", extract_evidence_node)
    g.add_node("assess", assess_coverage_node)

    g.add_node("refine", refine_queries_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fanout)

    g.add_edge("worker", "reduce")
    g.add_conditional_edges("reduce", route_after_reduce)

    g.add_edge("rank", "evidence")
    g.add_edge("evidence", "assess")
    g.add_conditional_edges("assess", route_after_assess)

    g.add_conditional_edges("refine", fanout)

    g.add_edge("writer", END)
    return g.compile()


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    q = input("Enter your research question: ").strip()
    if not q:
        raise RuntimeError("Empty question")

    app = build_graph()
    out = app.invoke(
        {
            "messages": [HumanMessage(content=q)],
            "completed": [],
            "raw_sources": [],
            "round": 0,
            "max_rounds": MAX_ROUNDS,
            "total_workers": 0,
        }
    )
    print("\n" + out["messages"][-1].content)


if __name__ == "__main__":
    main()
