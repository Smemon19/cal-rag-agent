"""
insert_docs.py
--------------
Command-line utility to crawl URLs or process local files using Crawl4AI, detect content type (sitemap, .txt, or regular page),
use the appropriate crawl method, chunk the resulting Markdown into <1000 character blocks by header hierarchy,
and insert all chunks into ChromaDB with metadata.

Usage:
    python insert_docs.py <PATH> [--collection ...] [--db-dir ...] [--embedding-model ...]
    
    PATH can be:
    - A local file path (e.g., /path/to/document.pdf)
    - A URL (e.g., https://example.com)
    - A file:// URL (e.g., file:///path/to/document.pdf)
    - A raw: URL (e.g., raw:https://example.com)
"""
import argparse
import sys
import re
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
import requests
import time
from bs4 import BeautifulSoup
from utils import (
    get_chroma_client,
    get_or_create_collection,
    add_documents_to_collection,
    resolve_collection_name,
    normalize_source_url,
    build_section_path,
    make_chunk_metadata,
    resolve_embedding_backend_and_model,
)
from rag_agent import RagAgent

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
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            results = await _crawl_many_with_retry(crawler, urls_to_crawl, run_config, dispatcher)
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.final_url or result.url, 'markdown': result.markdown, 'title': (result.metadata or {}).get('title', '')})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await _crawl_one_with_retry(crawler, url, crawl_config)
        if result and result.success and result.markdown:
            return [{'url': result.final_url or url, 'markdown': result.markdown, 'title': (result.metadata or {}).get('title', '')}]
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
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await _crawl_many_with_retry(crawler, urls, crawl_config, dispatcher)
        out: List[Dict[str, Any]] = []
        for r in results:
            if r.success and r.markdown:
                out.append({'url': r.final_url or r.url, 'markdown': r.markdown})
            else:
                print(f"[ingest] Crawl failed for {r.url}; falling back to static parse")
                md = _static_fetch_to_markdown(r.url)
                if md:
                    out.append({'url': r.url, 'markdown': md})
        return out

async def _crawl_many_with_retry(crawler: AsyncWebCrawler, urls: List[str], config: CrawlerRunConfig, dispatcher: MemoryAdaptiveDispatcher):
    results = await crawler.arun_many(urls=urls, config=config, dispatcher=dispatcher)
    failed = [u for u, r in zip(urls, results) if not (r.success and (r.markdown or r.html))]
    if not failed:
        return results
    print(f"[ingest] Some URLs failed to crawl; retrying with backoff (count={len(failed)})")
    time.sleep(1.5)
    retry_results = await crawler.arun_many(urls=failed, config=config, dispatcher=dispatcher)
    url_to_res = {u: r for u, r in zip(urls, results)}
    for u, rr in zip(failed, retry_results):
        url_to_res[u] = rr
    return [url_to_res[u] for u in urls]

async def _crawl_one_with_retry(crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig):
    res = await crawler.arun(url=url, config=config)
    if res.success and (res.markdown or res.html):
        return res
    print("[ingest] Crawl failed; retrying with backoff")
    time.sleep(1.5)
    res2 = await crawler.arun(url=url, config=config)
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

def main():
    parser = argparse.ArgumentParser(description="Insert crawled docs or local files into ChromaDB")
    parser.add_argument("path", help="Path to crawl (local file or URL - regular, .txt/.md, or sitemap)")
    parser.add_argument("--collection", default=None, help="ChromaDB collection name (overrides env)")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Chunk overlap between adjacent slices (chars)")
    parser.add_argument("--max-depth", type=int, default=3, help="Recursion depth for regular URLs")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max parallel browser sessions")
    parser.add_argument("--batch-size", type=int, default=100, help="ChromaDB insert batch size")
    # PDF-specific options
    parser.add_argument("--pdf-pages", type=str, default=None, help="PDF page selection like '10-20,35,40-42' (1-based)")
    parser.add_argument("--pdf-render-dpi", type=int, default=300, help="DPI to render pages for OCR (0 to disable)")
    parser.add_argument("--pdf-ocr-lang", type=str, default="eng", help="Tesseract languages, e.g., 'eng' or 'eng+osd'")
    parser.add_argument("--pdf-ocr-psm", type=int, default=6, help="Tesseract PSM mode (layout assumption)")
    parser.add_argument("--pdf-ocr-oem", type=int, default=3, help="Tesseract OEM mode")
    parser.add_argument("--pdf-diagnostic-dir", type=str, default=None, help="Directory to write per-page diagnostic artifacts")
    args = parser.parse_args()

    # Resolve final collection and log once
    resolved_collection = resolve_collection_name(args.collection)
    print(f"[ingest] Using ChromaDB collection: '{resolved_collection}'")

    # Create RagAgent instance
    agent = RagAgent(
        collection_name=resolved_collection,
        db_directory=args.db_dir,
        embedding_model=args.embedding_model
    )

    # Determine if input is a URL or local file
    path = args.path
    
    async def process_input():
        if is_url(path):
            print(f"Processing URL: {path}")
            await agent.insert_from_url(
                path,
                chunk_size=args.chunk_size,
                max_depth=args.max_depth,
                max_concurrent=args.max_concurrent,
                batch_size=args.batch_size,
                overlap_chars=args.overlap_chars
            )
        else:
            print(f"Processing local file: {path}")
            await agent.insert_from_file(
                path,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                overlap_chars=args.overlap_chars,
                pdf_pages=args.pdf_pages,
                pdf_render_dpi=(args.pdf_render_dpi if args.pdf_render_dpi and args.pdf_render_dpi > 0 else None),
                pdf_ocr_lang=args.pdf_ocr_lang,
                pdf_ocr_psm=args.pdf_ocr_psm,
                pdf_ocr_oem=args.pdf_ocr_oem,
                pdf_diagnostic_dir=args.pdf_diagnostic_dir,
            )

    # Run the async function
    asyncio.run(process_input())

if __name__ == "__main__":
    main()
