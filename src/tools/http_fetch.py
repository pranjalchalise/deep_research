# src/tools/http_fetch.py
"""
HTTP fetching utilities for retrieving and extracting text from web pages.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

from src.utils.cache import FileCache, get_page_cache, make_page_key
from src.utils.text import strip_html_naive, normalize_ws, chunk_text


# Domains that often block scrapers or have complex JS rendering
SKIP_DOMAINS = {
    "twitter.com", "x.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
}

# Domains that work well
GOOD_DOMAINS = {
    "arxiv.org",
    "github.com",
    "stackoverflow.com",
    "reddit.com",
    "medium.com",
    "dev.to",
    "wikipedia.org",
    "pubmed.ncbi.nlm.nih.gov",
}


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped (e.g., social media that blocks scrapers)."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lower().replace("www.", "")
        return host in SKIP_DOMAINS
    except Exception:
        return False


def fetch_html(url: str, timeout_s: float = 12.0) -> str:
    """
    Fetch raw HTML from a URL.

    Returns empty string on failure.
    """
    if should_skip_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Try requests first (better handling)
    try:
        import requests
        resp = requests.get(url, timeout=timeout_s, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        pass

    # Fallback to urllib
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def html_to_text(html: str) -> str:
    """
    Extract clean text from HTML.

    Tries multiple extractors in order of quality:
    1. trafilatura (best for articles)
    2. BeautifulSoup (good general purpose)
    3. Naive regex (fallback)
    """
    if not html or len(html) < 100:
        return ""

    # Try trafilatura first (best quality for articles)
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted and len(extracted) > 200:
            return normalize_ws(extracted)
    except Exception:
        pass

    # Try BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Try to find main content
        main = soup.find("main") or soup.find("article") or soup.find(class_="content") or soup.body
        if main:
            text = main.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)

        text = normalize_ws(text)
        if len(text) > 200:
            return text
    except Exception:
        pass

    # Naive fallback
    return normalize_ws(strip_html_naive(html))


def fetch_page_text(
    url: str,
    timeout_s: float = 12.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Fetch a URL and extract clean text content.

    Args:
        url: URL to fetch
        timeout_s: Request timeout in seconds
        use_cache: Whether to use cache
        cache_dir: Custom cache directory

    Returns:
        Extracted text content (empty string on failure)
    """
    cache_key = make_page_key(url)

    # Check cache
    if use_cache:
        cache = get_page_cache(cache_dir) if cache_dir else get_page_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # Fetch and extract
    html = fetch_html(url, timeout_s)
    text = html_to_text(html)

    # Cache result (even empty to avoid re-fetching failures)
    if use_cache and text:
        cache = get_page_cache(cache_dir) if cache_dir else get_page_cache()
        cache.set(cache_key, text)

    return text


def fetch_and_chunk(
    url: str,
    chunk_chars: int = 3500,
    chunk_overlap: int = 350,
    max_chunks: int = 4,
    timeout_s: float = 12.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Fetch a URL and return chunked text content.

    Args:
        url: URL to fetch
        chunk_chars: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        max_chunks: Maximum chunks to return
        timeout_s: Request timeout
        use_cache: Whether to use cache
        cache_dir: Custom cache directory

    Returns:
        List of text chunks (may be empty on failure)
    """
    text = fetch_page_text(url, timeout_s, use_cache, cache_dir)
    if not text:
        return []

    chunks = chunk_text(text, chunk_chars, chunk_overlap)
    return chunks[:max_chunks]


def fetch_multiple(
    urls: List[str],
    chunk_chars: int = 3500,
    chunk_overlap: int = 350,
    max_chunks_per_url: int = 4,
    timeout_s: float = 12.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Fetch multiple URLs and return chunked content for each.

    Args:
        urls: List of URLs to fetch
        chunk_chars: Size of each chunk
        chunk_overlap: Overlap between chunks
        max_chunks_per_url: Maximum chunks per URL
        timeout_s: Request timeout per URL
        use_cache: Whether to use cache
        cache_dir: Custom cache directory

    Returns:
        Dict mapping URL -> list of text chunks
    """
    results: Dict[str, List[str]] = {}

    for url in urls:
        chunks = fetch_and_chunk(
            url=url,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks_per_url,
            timeout_s=timeout_s,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        results[url] = chunks

    return results


__all__ = [
    "fetch_html",
    "html_to_text",
    "fetch_page_text",
    "fetch_and_chunk",
    "fetch_multiple",
    "should_skip_url",
]
