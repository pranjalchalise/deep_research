"""
Formatters that turn internal source/evidence dicts into human-readable text.

Used by prompt templates and the final report writer.
"""
from __future__ import annotations

from typing import List

def sources_markdown(sources: List[dict]) -> str:
    """Render the sources ledger as a numbered reference list."""
    lines = []
    for s in sources:
        lines.append(f"[{s['sid']}] {s.get('title','').strip() or s['url']} — {s['url']}")
    return "\n".join(lines)

def evidence_block(evidence: List[dict]) -> str:
    """Render evidence items with their source cross-references."""
    lines = []
    for e in evidence:
        lines.append(f"[{e['eid']}] sid={e['sid']} section={e['section']}\n{e['text']}\n")
    return "\n".join(lines)
