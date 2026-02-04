"""
Low-level text helpers: HTML stripping, chunking, and whitespace normalization.

These are the building blocks used by http_fetch and the extraction nodes.
"""
from __future__ import annotations

import re
from typing import List

def strip_html_naive(html: str) -> str:
    """Regex-based HTML tag removal. Use only when no real parser is available."""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    """Split text into fixed-size chunks with overlap so we don't cut mid-sentence."""
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
    """Collapse all runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", (s or "")).strip()
