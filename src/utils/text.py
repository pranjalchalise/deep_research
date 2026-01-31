from __future__ import annotations

import re
from typing import List

def strip_html_naive(html: str) -> str:
    # very rough fallback if no better parser available
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
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
    return re.sub(r"\s+", " ", (s or "")).strip()
