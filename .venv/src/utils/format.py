from __future__ import annotations

from typing import List

from src.core.state import Evidence, Source


def sources_block(sources: List[Source], max_snippet: int = 500) -> str:
    parts = []
    for s in sources:
        parts.append(
            f"[{s['sid']}] {s['title']}\nURL: {s['url']}\nSnippet: {(s.get('snippet') or '')[:max_snippet]}\n"
        )
    return "\n".join(parts)


def evidence_block(evidence: List[Evidence], max_text: int = 900) -> str:
    parts = []
    for e in evidence:
        parts.append(
            f"[{e['eid']}] (Source {e['sid']}) Section: {e['section']}\nURL: {e['url']}\nText: {(e['text'] or '')[:max_text]}\n"
        )
    return "\n".join(parts)
