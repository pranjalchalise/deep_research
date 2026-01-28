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


class AgentState(MessagesState):
    plan: Optional[Plan]
    total_workers: Optional[int]
    done_workers: Annotated[int, operator.add]
    raw_sources: Annotated[List[RawSource], operator.add]
    sources: Optional[List[RawSource]]


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
        )
    )
    resp = llm.invoke([system, HumanMessage(content=f"User question:\n{user_q}")])
    txt = resp.content.strip()
    try:
        plan: Plan = json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        plan = json.loads(txt[start : end + 1])

    total = len(plan.get("queries", []))
    return {"plan": plan, "total_workers": total, "done_workers": 0, "raw_sources": [], "sources": None}


def worker_node(state: AgentState) -> Dict[str, Any]:
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
        out.append({"url": url, "title": title, "snippet": snippet[:700]})

    return {"raw_sources": out, "done_workers": 1}


def reducer_node(state: AgentState) -> Dict[str, Any]:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    if done < total:
        return {}

    seen = {}
    for s in state.get("raw_sources") or []:
        url = s["url"]
        if url not in seen:
            seen[url] = s

    sources = list(seen.values())
    return {"sources": sources}


def writer_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan") or {"topic": "Research Report", "queries": []}
    sources = state.get("sources") or []
    user_q = state["messages"][-1].content

    labeled = []
    for i, s in enumerate(sources, start=1):
        labeled.append(
            {
                "sid": f"S{i}",
                "title": s["title"],
                "url": s["url"],
                "snippet": s.get("snippet", ""),
            }
        )

    sources_text = "\n".join(
        [f"[{s['sid']}] {s['title']}\nURL: {s['url']}\nSNIPPET: {s['snippet']}\n" for s in labeled]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(
        content=(
            "You are a careful researcher.\n"
            "Write a concise report answering the user's question using ONLY the provided sources/snippets.\n"
            "Rules:\n"
            "- Every paragraph must include at least one citation like [S1].\n"
            "- Do not invent facts not supported by sources.\n"
            "- End with a Sources section listing [S#] Title — URL.\n"
        )
    )
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


def fanout(state: AgentState):
    plan = state.get("plan") or {}
    qs = plan.get("queries") or []
    return [Send("worker", {"query_item": qi}) for qi in qs]


def route_after_reduce(state: AgentState) -> str:
    total = state.get("total_workers") or 0
    done = state.get("done_workers") or 0
    return "writer" if done >= total else END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reduce", reducer_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fanout)
    g.add_edge("worker", "reduce")
    g.add_conditional_edges("reduce", route_after_reduce)
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
