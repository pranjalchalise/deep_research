from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, AssessResult, Lane

ASSESS_SYSTEM = """You assess research coverage.

Return ONLY JSON:
{
  "coverage_ok": true|false,
  "gaps": ["..."],
  "suggested_lanes": ["docs|papers|news|forums|general", ...]
}

Rules:
- coverage_ok must be true only if the report can answer the user question well.
- gaps should be concrete missing items (e.g. "primary source definition", "recent updates", "limitations").
- suggested_lanes should be where to search next to fill gaps.
"""

def assess_node(state: AgentState) -> Dict[str, Any]:
    user_q = state["messages"][-1].content
    plan = state.get("plan") or {}
    sources = state.get("sources") or []
    claims = state.get("claims") or []
    issues = state.get("issues") or []

    # heuristic minimums
    if len(sources) < int(state.get("require_min_sources") or 4):
        last_assess: AssessResult = {
            "coverage_ok": False,
            "gaps": ["Not enough distinct sources to answer confidently."],
            "suggested_lanes": ["docs", "papers", "general"],
        }
        return {"last_assess": last_assess}

    if any(i.get("level") == "block" for i in issues):
        last_assess = {
            "coverage_ok": False,
            "gaps": ["Claims missing citations (blocked). Need more evidence extraction or better sources."],
            "suggested_lanes": ["docs", "papers", "general"],
        }
        return {"last_assess": last_assess}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    resp = llm.invoke(
        [
            SystemMessage(content=ASSESS_SYSTEM),
            HumanMessage(
                content=(
                    f"User question:\n{user_q}\n\n"
                    f"Current topic/outline:\n{json.dumps({'topic': plan.get('topic'), 'outline': plan.get('outline')}, ensure_ascii=False)}\n\n"
                    f"Sources count: {len(sources)}\n"
                    f"Claims count: {len(claims)}\n"
                    f"Issues: {json.dumps(issues, ensure_ascii=False)}\n"
                )
            ),
        ]
    )

    txt = resp.content.strip()
    try:
        out = json.loads(txt)
        if not isinstance(out, dict):
            raise ValueError("bad assess")
    except Exception:
        out = {"coverage_ok": True, "gaps": [], "suggested_lanes": []}

    # sanitize lanes
    lanes = []
    for x in out.get("suggested_lanes", []) or []:
        x = str(x)
        if x in ("docs", "papers", "news", "forums", "general"):
            lanes.append(x)
    last_assess: AssessResult = {
        "coverage_ok": bool(out.get("coverage_ok", False)),
        "gaps": [str(g) for g in (out.get("gaps", []) or [])][:6],
        "suggested_lanes": lanes[:4],
    }
    return {"last_assess": last_assess}
