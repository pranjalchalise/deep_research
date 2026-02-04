"""
Low-level text helpers: HTML stripping, chunking, whitespace cleanup.

These are the building blocks used by http_fetch and the extraction nodes.
"""
from __future__ import annotations

import re
from typing import List

def strip_html_naive(html: str) -> str:
    """Regex-based HTML removal. Only use this as a last resort when no real parser is available."""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    """Split text into fixed-size chunks with overlap so we don't lose context at boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def normalize_ws(s: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", (s or "")).strip()
