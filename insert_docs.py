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
from crawl4ai import AsyncWebCrawler
import requests
import time
from bs4 import BeautifulSoup
from utils import (
    resolve_collection_name,
    normalize_source_url,
    build_section_path,
    make_chunk_metadata,
    resolve_embedding_backend_and_model,
)
from rag_agent import RagAgent

from utils_crawler import (
    is_url,
    is_sitemap,
    is_txt,
    crawl_recursive_internal_links,
    crawl_markdown_file,
    parse_sitemap,
    crawl_batch,
    extract_section_info,
    smart_chunk_markdown
)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return result


async def _crawl_many_with_retry(crawler: AsyncWebCrawler, urls: List[str]):
    # New API: arun_many(urls, ...)
    # cache_mode might be a string or enum, let's try simple first
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
    _run_async(process_input())

if __name__ == "__main__":
    main()
