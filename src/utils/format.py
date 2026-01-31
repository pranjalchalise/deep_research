from __future__ import annotations

from typing import List

def sources_markdown(sources: List[dict]) -> str:
    lines = []
    for s in sources:
        lines.append(f"[{s['sid']}] {s.get('title','').strip() or s['url']} — {s['url']}")
    return "\n".join(lines)

def evidence_block(evidence: List[dict]) -> str:
    lines = []
    for e in evidence:
        lines.append(f"[{e['eid']}] sid={e['sid']} section={e['section']}\n{e['text']}\n")
    return "\n".join(lines)
