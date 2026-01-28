import json
import os
import operator
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from typing_extensions import Annotated

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send


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


class Claim(TypedDict):
    cid: str
    claim: str


class Evidence(TypedDict):
    eid: str
    sid: str
    url: str
    evidence: str


# -----------------------------
# State
# -----------------------------
class AgentState(MessagesState):
    plan: Optional[Plan]
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]
    raw_sources: Annotated[List[RawSource], operator.add]
    sources: Optional[List[RawSource]]

    claims: Optional[List[Claim]]
    evidence: Annotated[List[Evidence], operator.add]


# -----------------------------
# Helpers
# -----------------------------
def _safe_json_loads(txt: str) -> Any:
    txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(txt[start : end + 1])


# -----------------------------
# Nodes
# -----------------------------
def planner_node(state: AgentState) -> Dict[str, Any]:
    """Generate a diverse search plan (5-8 queries)."""
    user_q = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a research planner. Return ONLY valid JSON:\n"
            '{ "topic": "...", "queries": [{"qid":"Q1","query":"..."}, ...] }\n'
            "Constraints:\n"
            "- 5 to 8 queries\n"
            "- qid must be Q1..Qn\n"
            "- queries should cover different aspects\n"
            "- do not include commentary\n"
        )
    )
    resp = llm.invoke([system, HumanMessage(content=f"User question:\n{user_q}")])
    plan: Plan = _safe_json_loads(resp.content)

    total = len(plan.get("queries", []))
    return {
        "plan": plan,
        "total_workers": total,
        "done_workers": 0,
        "raw_sources": [],
        "sources": None,
        "claims": None,
        "evidence": [],
    }


def worker_node(state: AgentState) -> Dict[str, Any]:
    """Run Tavily for a single query. Invoked via Send with query_item."""
    qi = state["query_item"]  # injected by Send
    query = qi["query"]

    search = TavilySearch(max_results=5, topic="general")
    raw = search.invoke({"query": query})

    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    out: List[RawSource] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        title = str(r.get("title", "")).strip() or url
        snippet = str(r.get("content", "") or r.get("snippet", "") or "").strip()
        out.append({"url": url, "title": title, "snippet": snippet[:900]})

    return {"raw_sources": out, "done_workers": 1}


def reducer_node(state: AgentState) -> Dict[str, Any]:
    """Deduplicate sources by URL once all workers finish."""
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    if done < total:
        return {}

    seen: Dict[str, RawSource] = {}
    for s in state.get("raw_sources") or []:
        url = s["url"]
        if url not in seen:
            seen[url] = s

    sources = list(seen.values())
    return {"sources": sources}


def claims_node(state: AgentState) -> Dict[str, Any]:
    """Convert the user question into 5-10 atomic claims."""
    user_q = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "Turn the user question into 5-10 atomic claims the report should address.\n"
            "Return ONLY JSON: {\"claims\": [{\"claim\":\"...\"}, ...]}.\n"
            "Rules:\n"
            "- Claims must be answerable with web evidence\n"
            "- Each claim should be specific\n"
            "- No extra text\n"
        )
    )
    resp = llm.invoke([system, HumanMessage(content=user_q)])
    data = _safe_json_loads(resp.content)

    raw_claims = data.get("claims", [])
    out: List[Claim] = []
    for i, c in enumerate(raw_claims, start=1):
        claim_txt = str(c.get("claim", "") if isinstance(c, dict) else c).strip()
        if claim_txt:
            out.append({"cid": f"C{i}", "claim": claim_txt})

    # fallback if model returned nothing
    if not out:
        out = [{"cid": "C1", "claim": user_q.strip()}]

    return {"claims": out}


def evidence_node(state: AgentState) -> Dict[str, Any]:
    """Extract 3-10 concise evidence statements from snippets."""
    sources = state.get("sources") or []
    if not sources:
        return {"evidence": []}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    labeled_sources = []
    for i, s in enumerate(sources, start=1):
        labeled_sources.append(
            {
                "sid": f"S{i}",
                "title": s["title"],
                "url": s["url"],
                "snippet": s.get("snippet", ""),
            }
        )

    src_text = "\n\n".join(
        [f"[{x['sid']}] {x['title']}\nURL: {x['url']}\nSNIPPET: {x['snippet']}" for x in labeled_sources]
    )

    system = SystemMessage(
        content=(
            "Extract evidence statements ONLY from the provided SNIPPET text.\n"
            "Return ONLY valid JSON:\n"
            '{ "evidence": [{"sid":"S1","url":"...","evidence":"..."}, ...] }\n'
            "Rules:\n"
            "- 3 to 10 evidence items total\n"
            "- Each evidence must be a specific factual statement supported by some snippet text\n"
            "- Keep each evidence <= 220 characters\n"
            "- No extra commentary\n"
        )
    )
    resp = llm.invoke([system, HumanMessage(content=src_text)])
    data = _safe_json_loads(resp.content)

    ev: List[Evidence] = []
    for idx, item in enumerate(data.get("evidence", []), start=1):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("sid", "")).strip()
        url = str(item.get("url", "")).strip()
        evidence_txt = str(item.get("evidence", "")).strip()
        if sid and url and evidence_txt:
            ev.append({"eid": f"E{idx}", "sid": sid, "url": url, "evidence": evidence_txt})

    return {"evidence": ev}


def writer_node(state: AgentState) -> Dict[str, Any]:
    """Write report using ONLY evidence ledger. Cite with [E#]."""
    user_q = state["messages"][-1].content
    plan = state.get("plan") or {"topic": "Research Report", "queries": []}
    sources = state.get("sources") or []
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []

    # Build compact text blocks
    claims_text = "\n".join([f"{c['cid']}: {c['claim']}" for c in claims])

    ev_text = "\n".join(
        [f"[{e['eid']}] ({e['sid']}) {e['evidence']}  URL: {e['url']}" for e in evidence]
    )

    sources_text = "\n".join([f"[S{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a careful research writer.\n"
            "Answer the user question using ONLY the evidence ledger.\n"
            "Rules:\n"
            "- Every paragraph must include at least one evidence citation like [E1].\n"
            "- Do not introduce facts not supported by evidence.\n"
            "- If evidence does not support a claim, explicitly say 'Insufficient evidence in retrieved sources'.\n"
            "- Keep the report concise and helpful.\n"
            "- End with:\n"
            "  1) Evidence Used section listing E# -> URL\n"
            "  2) Sources section listing S# -> URL\n"
        )
    )

    resp = llm.invoke(
        [
            system,
            HumanMessage(
                content=(
                    f"Topic: {plan.get('topic')}\n\n"
                    f"User question:\n{user_q}\n\n"
                    f"Claims to address:\n{claims_text}\n\n"
                    f"Evidence ledger:\n{ev_text}\n\n"
                    f"Sources:\n{sources_text}\n"
                )
            ),
        ]
    )
    return {"messages": [AIMessage(content=resp.content.strip())]}


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
    return "claims" if done >= total else END


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    g.add_node("claims", claims_node)
    g.add_node("evidence", evidence_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fanout)  # fan-out to workers
    g.add_edge("worker", "reduce")              # each worker contributes sources, then reduce
    g.add_conditional_edges("reduce", route_after_reduce)

    g.add_edge("claims", "evidence")
    g.add_edge("evidence", "writer")
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
    print(out["messages"][-1].content)