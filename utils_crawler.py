"""
utils_crawler.py
----------------
Utilities for crawling URLs and processing content using Crawl4AI.
Extracted from insert_docs.py to avoid circular imports.
"""

import re
import time
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

def smart_chunk_markdown(markdown: str, max_len: int = 1000, overlap_chars: int = 150) -> List[str]:
    """Hierarchically split markdown by #, ##, ### then by characters with overlap."""
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    chunks: List[str] = []

    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            step = max(1, max_len - max(0, overlap_chars))
                            i = 0
                            while i < len(h3):
                                piece = h3[i:i+max_len].strip()
                                if piece:
                                    chunks.append(piece)
                                i += step
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)

    final_chunks: List[str] = []

    for c in chunks:
        if len(c) > max_len:
            step = max(1, max_len - max(0, overlap_chars))
            i = 0
            while i < len(c):
                piece = c[i:i+max_len].strip()
                if piece:
                    final_chunks.append(piece)
                i += step
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if c]

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt') or url.endswith('.md') or url.endswith('.markdown')

async def crawl_recursive_internal_links(start_urls, max_depth=3, max_concurrent=10) -> List[Dict[str,Any]]:
    """Recursive crawl using logic from 5-crawl_recursive_internal_links.py. Returns list of dicts with url and markdown."""
    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            results = await _crawl_many_with_retry(crawler, urls_to_crawl)
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.url, 'markdown': result.markdown, 'title': (result.metadata or {}).get('title', '')})
                    internal_links = result.links.get("internal", []) if isinstance(result.links, dict) else []
                    for link in internal_links:
                        href = link.get("href") if isinstance(link, dict) else link
                        if href:
                            next_url = normalize_url(href)
                            if next_url not in visited:
                                next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    async with AsyncWebCrawler(headless=True) as crawler:
        result = await _crawl_one_with_retry(crawler, url)
        if result and result.success and result.markdown:
            return [{'url': result.url, 'markdown': result.markdown, 'title': (result.metadata or {}).get('title', '')}]
        print(f"[ingest] Crawl failed for {url}; falling back to static parse")
        md = _static_fetch_to_markdown(url)
        if md:
            return [{'url': url, 'markdown': md, 'title': ''}]
        return []

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str,Any]]:
    """Batch crawl using logic from 3-crawl_sitemap_in_parallel.py."""
    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        results = await _crawl_many_with_retry(crawler, urls)
        out: List[Dict[str, Any]] = []
        for r in results:
            if r.success and r.markdown:
                out.append({'url': r.url, 'markdown': r.markdown})
            else:
                print(f"[ingest] Crawl failed for {r.url}; falling back to static parse")
                md = _static_fetch_to_markdown(r.url)
                if md:
                    out.append({'url': r.url, 'markdown': md})
        return out

async def _crawl_many_with_retry(crawler: AsyncWebCrawler, urls: List[str]):
    results = await crawler.arun_many(urls=urls, bypass_cache=True)
    failed = [u for u, r in zip(urls, results) if not (r.success and (r.markdown or r.html))]
    if not failed:
        return results
    print(f"[ingest] Some URLs failed to crawl; retrying with backoff (count={len(failed)})")
    time.sleep(1.5)
    retry_results = await crawler.arun_many(urls=failed, bypass_cache=True)
    url_to_res = {u: r for u, r in zip(urls, results)}
    for u, rr in zip(failed, retry_results):
        url_to_res[u] = rr
    return [url_to_res[u] for u in urls]

async def _crawl_one_with_retry(crawler: AsyncWebCrawler, url: str):
    res = await crawler.arun(url=url, bypass_cache=True)
    if res.success and (res.markdown or res.html):
        return res
    print("[ingest] Crawl failed; retrying with backoff")
    time.sleep(1.5)
    res2 = await crawler.arun(url=url, bypass_cache=True)
    return res2

def _static_fetch_to_markdown(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=15)
        ctype = (r.headers.get('content-type') or '').lower()
        if 'text/html' not in ctype and 'markdown' not in ctype and 'text/plain' not in ctype:
            print(f"[ingest] Static fetch non-HTML/text content for {url}; skipping")
            return None
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text('\n')
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        md = "\n\n".join(lines)
        return md
    except Exception as e:
        print(f"[ingest] Static fetch failed for {url}: {e}")
        return None

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def is_url(path: str) -> bool:
    """Check if the path is a URL (starts with http://, https://, file://, or raw:)."""
    return path.startswith(('http://', 'https://', 'file://', 'raw:'))
