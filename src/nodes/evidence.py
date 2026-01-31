from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import AgentState, Evidence, Page, Source
from src.utils.text import chunk_text

EVIDENCE_SYSTEM = """You extract concise, factual evidence from text.

Return ONLY JSON list:
[
  {"sid":"S1","section":"...","text":"1-3 factual sentences"},
  ...
]

Rules:
- Each item must be directly supported by the provided text chunk.
- Avoid fluff. Prefer specific definitions, facts, numbers, and explicit statements.
- Do not invent anything not in the chunk.
"""

def evidence_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    outline = plan.get("outline") or []
    pages: List[Page] = state.get("pages") or []
    sources: List[Source] = state.get("sources") or []

    # Map sid -> preferred section via lane/outline is hard; keep it simple:
    # We'll ask the model to choose a best-matching section for each evidence item.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chunk_chars = int(state.get("chunk_chars") or 3500)
    overlap = int(state.get("chunk_overlap") or 350)
    max_chunks = int(state.get("max_chunks_per_source") or 4)
    per_source = int(state.get("evidence_per_source") or 3)

    if state.get("fast_mode", True):
        max_chunks = min(max_chunks, 2)
        per_source = min(per_source, 2)

    extracted: List[dict] = []

    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) < 300:
            continue

        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)[:max_chunks]

        for ch in chunks:
            prompt = (
                f"Outline sections:\n{outline}\n\n"
                f"Source sid: {p['sid']}\n"
                f"Extract up to {per_source} evidence items from this chunk.\n\n"
                f"TEXT CHUNK:\n{ch}"
            )

            resp = llm.invoke([SystemMessage(content=EVIDENCE_SYSTEM), HumanMessage(content=prompt)])
            txt = resp.content.strip()
            try:
                items = json.loads(txt)
                if isinstance(items, list):
                    extracted.extend(items)
            except Exception:
                continue

    # clean + cap
    out: List[Evidence] = []
    for it in extracted:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sid", "")).strip()
        section = str(it.get("section", "")).strip() or "General"
        text = str(it.get("text", "")).strip()
        if sid and text:
            out.append({"eid": "", "sid": sid, "section": section, "text": text[:900]})

    return {"evidence": out}
