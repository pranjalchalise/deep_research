from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.state import AgentState, Plan

PLANNER_SYSTEM = """You are a research planner.

Given a user question, output ONLY valid JSON with:
{
  "topic": "short topic",
  "outline": ["Section 1", ...],
  "queries": [{"qid":"Q1","query":"...","section":"Section 1"}, ...]
}

Constraints:
- outline: 4-8 sections
- queries: 6-12 items
- queries must map to a section in outline
- qid must be Q1..Qn in order
"""

def planner_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    depth = state.get("depth") or 8  # we’ll use as a hint

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    resp = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"User question:\n{user_q}\n\nTarget queries count: {depth} (approx).")
    ])

    txt = resp.content.strip()
    try:
        plan: Plan = json.loads(txt)
    except Exception:
        # crude fallback: extract outermost JSON
        start = txt.find("{")
        end = txt.rfind("}")
        plan = json.loads(txt[start:end+1])

    # minimal guards
    plan["topic"] = plan.get("topic") or "Research Report"
    plan["outline"] = plan.get("outline") or ["Overview", "Key Findings", "Conclusion"]
    plan["queries"] = plan.get("queries") or []

    # set worker counts for map/reduce
    total_workers = len(plan["queries"])

    return {
        "plan": plan,
        "total_workers": total_workers,
        "done_workers": 0,
        "raw_sources": [],
        "raw_evidence": [],
        "issues": [],
    }
