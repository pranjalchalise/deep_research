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
# Config
# -----------------------------
MAX_SOURCES = 10
MAX_RESULTS_PER_QUERY = 5
FOLLOWUP_SEARCH_ENABLED = True
MAX_FOLLOWUP_QUERIES = 3

# If you want: keep reddit/medium sometimes but rank them low (don’t block)
HARD_BLOCKLIST_DOMAINS = {
    "oreateai.com",
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
}


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


class SupportMap(TypedDict):
    cid: str
    eids: List[str]


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
    support: Optional[List[SupportMap]]
    missing_claims: Optional[List[Claim]]


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
    # very small helper to detect domains
    try:
        u = url.replace("https://", "").replace("http://", "")
        return u.split("/")[0].lower()
    except Exception:
        return url.lower()


def _soft_penalty(url: str) -> int:
    d = _domain(url)
    return 1 if any(x in d for x in SOFT_DOWNRANK_DOMAINS) else 0


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
            "- 5 to 8 queries\n"
            "- qid must be Q1..Qn\n"
            "- queries should cover different aspects\n"
            "- no extra text\n"
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
        "support": None,
        "missing_claims": None,
    }


def worker_node(state: AgentState) -> Dict[str, Any]:
    qi = state["query_item"]  # injected by Send
    query = qi["query"]

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

        # hard block
        if any(b in _domain(url) for b in HARD_BLOCKLIST_DOMAINS):
            continue

        title = str(r.get("title", "")).strip() or url
        snippet = str(r.get("content", "") or r.get("snippet", "") or "").strip()
        if not snippet:
            continue

        out.append({"url": url, "title": title, "snippet": snippet[:900]})

    return {"raw_sources": out, "done_workers": 1}


def reducer_node(state: AgentState) -> Dict[str, Any]:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    if done < total:
        return {}

    seen: Dict[str, RawSource] = {}
    for s in state.get("raw_sources") or []:
        url = s["url"]
        if url not in seen:
            seen[url] = s

    return {"sources": list(seen.values())}


def rank_sources_node(state: AgentState) -> Dict[str, Any]:
    sources = state.get("sources") or []
    if not sources:
        return {}

    # add a tiny heuristic: prefer non-downrank domains, longer snippet
    scored = []
    for s in sources:
        url = s["url"]
        snippet_len = len(s.get("snippet", "") or "")
        penalty = _soft_penalty(url)
        # heuristic score: longer snippet better, penalize downrank domains
        score = snippet_len - (penalty * 250)
        scored.append((score, s))

    # pre-sort before LLM rank (gives LLM better candidates first)
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored[: min(len(scored), 20)]]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    system = SystemMessage(
        content=(
            "Rank sources for credibility + usefulness.\n"
            "Return ONLY JSON: {\"order\": [0,1,2,...]}\n"
            "Rank higher:\n"
            "- official docs, org sites, GitHub, reputable publishers\n"
            "Rank lower:\n"
            "- reddit/forums, low-quality SEO blogs\n"
            "Use title+url+snippet.\n"
        )
    )

    items = []
    for i, s in enumerate(candidates):
        items.append(
            f"{i}. {s['title']}\nURL: {s['url']}\nSNIPPET: {(s.get('snippet','')[:260])}"
        )
    prompt = "\n\n".join(items)

    resp = llm.invoke([system, HumanMessage(content=prompt)])
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
    return {"sources": ranked}


def claims_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "Turn the user question into 6-10 atomic claims the report should address.\n"
            "Return ONLY JSON: {\"claims\": [{\"claim\":\"...\"}, ...]}.\n"
            "Rules:\n"
            "- Claims should be concrete and non-overlapping\n"
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

    if not out:
        out = [{"cid": "C1", "claim": user_q.strip()}]

    return {"claims": out}


def evidence_node(state: AgentState) -> Dict[str, Any]:
    sources = state.get("sources") or []
    if not sources:
        return {"evidence": []}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    labeled_sources = []
    for i, s in enumerate(sources, start=1):
        labeled_sources.append(
            {"sid": f"S{i}", "title": s["title"], "url": s["url"], "snippet": s.get("snippet", "")}
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
            "- 8 to 18 evidence items total\n"
            "- Each evidence must be a specific factual statement in the snippet\n"
            "- Each evidence <= 220 characters\n"
            "- No extra text\n"
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


def align_node(state: AgentState) -> Dict[str, Any]:
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []
    if not claims or not evidence:
        return {"support": [], "missing_claims": claims}

    claims_text = "\n".join([f"{c['cid']}: {c['claim']}" for c in claims])
    ev_text = "\n".join([f"{e['eid']}: {e['evidence']} (URL: {e['url']})" for e in evidence])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    system = SystemMessage(
        content=(
            "Match evidence items to claims.\n"
            "Return ONLY JSON:\n"
            "{"
            "\"support\": [{\"cid\":\"C1\",\"eids\":[\"E1\",\"E4\"]}, ...],"
            "\"missing\": [\"C3\", ...]"
            "}\n"
            "Rules:\n"
            "- For each claim, choose 1-4 best evidence IDs that directly support it\n"
            "- If none support, add claim id to missing\n"
            "- No hallucinations\n"
        )
    )
    resp = llm.invoke([system, HumanMessage(content=f"Claims:\n{claims_text}\n\nEvidence:\n{ev_text}")])
    data = _safe_json_loads(resp.content)

    support = data.get("support", [])
    missing_ids = set(data.get("missing", []))

    # normalize support
    out_support: List[SupportMap] = []
    for item in support:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cid", "")).strip()
        eids = item.get("eids", [])
        if cid and isinstance(eids, list):
            eids = [str(e).strip() for e in eids if str(e).strip()]
            out_support.append({"cid": cid, "eids": eids})

    missing_claims = [c for c in claims if c["cid"] in missing_ids]
    return {"support": out_support, "missing_claims": missing_claims}


def followup_planner_node(state: AgentState) -> Dict[str, Any]:
    """If missing claims exist, propose up to 3 follow-up search queries."""
    if not FOLLOWUP_SEARCH_ENABLED:
        return {}

    missing = state.get("missing_claims") or []
    if not missing:
        return {}

    user_q = state["messages"][-1].content
    missing_text = "\n".join([f"{c['cid']}: {c['claim']}" for c in missing])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "Create follow-up web search queries to find evidence for the missing claims.\n"
            "Return ONLY JSON: {\"queries\": [\"...\", \"...\", ...]}\n"
            "Rules:\n"
            f"- max {MAX_FOLLOWUP_QUERIES} queries\n"
            "- queries should be precise and evidence-seeking\n"
            "- no extra text\n"
        )
    )
    resp = llm.invoke(
        [system, HumanMessage(content=f"User question: {user_q}\n\nMissing claims:\n{missing_text}")]
    )
    data = _safe_json_loads(resp.content)
    queries = data.get("queries", [])
    queries = [str(q).strip() for q in queries if str(q).strip()][:MAX_FOLLOWUP_QUERIES]

    if not queries:
        return {}

    # convert into PlanQuery style so we can reuse worker fanout
    followup_plan: Plan = {
        "topic": "Follow-up Evidence",
        "queries": [{"qid": f"F{i+1}", "query": q} for i, q in enumerate(queries)],
    }
    return {"plan": followup_plan, "total_workers": len(followup_plan["queries"]), "done_workers": 0, "raw_sources": []}


def writer_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    plan = state.get("plan") or {"topic": "Research Report", "queries": []}
    sources = state.get("sources") or []
    claims = state.get("claims") or []
    evidence = state.get("evidence") or []
    support = state.get("support") or []

    # index evidence by id
    ev_by_id = {e["eid"]: e for e in evidence}

    # build claim->eids dict
    claim_to_eids: Dict[str, List[str]] = {}
    for s in support:
        claim_to_eids[s["cid"]] = s.get("eids", [])

    claim_blocks = []
    for c in claims:
        eids = claim_to_eids.get(c["cid"], [])
        ev_lines = []
        for eid in eids:
            e = ev_by_id.get(eid)
            if e:
                ev_lines.append(f"[{eid}] {e['evidence']}  URL: {e['url']}")
        block = f"{c['cid']}: {c['claim']}\n" + ("\n".join(ev_lines) if ev_lines else "(NO SUPPORTING EVIDENCE)")
        claim_blocks.append(block)

    claim_support_text = "\n\n".join(claim_blocks)

    sources_text = "\n".join([f"[S{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a careful research writer.\n"
            "Write a report answering the user question.\n"
            "You MUST ONLY use the evidence attached to each claim.\n"
            "Rules:\n"
            "- Write the report claim-by-claim.\n"
            "- Every paragraph must cite evidence like [E3].\n"
            "- If a claim has NO SUPPORTING EVIDENCE, explicitly say: 'Insufficient evidence in retrieved sources.'\n"
            "- Do not invent facts.\n"
            "- End with:\n"
            "  Evidence Used: list E# -> URL\n"
            "  Sources: list S# -> URL\n"
        )
    )

    # compute evidence used list
    used_eids = []
    for cid, eids in claim_to_eids.items():
        for eid in eids:
            if eid in ev_by_id and eid not in used_eids:
                used_eids.append(eid)
    evidence_used_text = "\n".join([f"{eid} -> {ev_by_id[eid]['url']}" for eid in used_eids]) if used_eids else "(none)"

    resp = llm.invoke(
        [
            system,
            HumanMessage(
                content=(
                    f"Topic: {plan.get('topic')}\n\n"
                    f"User question:\n{user_q}\n\n"
                    f"Claim + Evidence mapping:\n{claim_support_text}\n\n"
                    f"Evidence Used:\n{evidence_used_text}\n\n"
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
    return "rank_sources" if done >= total else END


def route_after_align(state: AgentState) -> str:
    missing = state.get("missing_claims") or []
    if FOLLOWUP_SEARCH_ENABLED and missing:
        return "followup_planner"
    return "writer"


def route_after_followup_planner(state: AgentState) -> str:
    # if followup_planner produced a new plan with queries, we fan out again
    plan = state.get("plan") or {}
    qs = plan.get("queries") or []
    return "do_followup_search" if qs else "writer"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)

    g.add_node("rank_sources", rank_sources_node)
    g.add_node("claims", claims_node)
    g.add_node("evidence", evidence_node)
    g.add_node("align", align_node)

    g.add_node("followup_planner", followup_planner_node)
    # We reuse the same worker/reduce for followups; this node is only used to trigger fanout
    g.add_node("do_followup_search", lambda state: {})

    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fanout)

    g.add_edge("worker", "reduce")
    g.add_conditional_edges("reduce", route_after_reduce)

    g.add_edge("rank_sources", "claims")
    g.add_edge("claims", "evidence")
    g.add_edge("evidence", "align")
    g.add_conditional_edges("align", route_after_align)

    # followup
    g.add_conditional_edges("followup_planner", route_after_followup_planner)
    g.add_conditional_edges("do_followup_search", fanout)   # fan out followup queries
    g.add_edge("worker", "reduce")                          # already exists, keeps working
    g.add_edge("reduce", "rank_sources")                    # rank again after followup sources

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
