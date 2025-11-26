"""Pydantic AI agent that leverages RAG with vector stores (ChromaDB or BigQuery)."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio
import re
from pathlib import Path

import dotenv
# Guarded imports for pydantic_ai to tolerate version differences
try:
    from pydantic_ai.agent import Agent  # preferred path
    from pydantic_ai import RunContext  # type context for tools
except Exception:
    try:
        from pydantic_ai import Agent, RunContext  # fallback path
    except Exception:
        Agent = None  # type: ignore
        RunContext = None  # type: ignore
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # type: ignore

from utils import (
    resolve_embedding_backend_and_model,
    normalize_source_url,
    build_section_path,
    make_chunk_metadata,
    get_env_file_path,
    ensure_appdata_scaffold,
    sanitize_and_validate_openai_key,
    resolve_collection_name,
    create_embedding_function,
    increment_query_count,
    format_results_as_context,
)
from utils_vectorstore import get_vector_store, resolve_vector_backend, VectorStore
from rag_components import RetrievalOrchestrator

from tools_calc import (
    deflection_limit,
    vehicle_barrier_reaction,
    fall_anchor_design_load,
    wind_speed_category,
    machinery_impact_factor,
)
from rules_loader import load_all_rules
from verify import verify_answer
import json as _json
import uuid as _uuid

SECTION_PATTERN = re.compile(r"\b\d{3,4}(?:\.\d+)+\b")
IRC_SECTION_PATTERN = re.compile(r"\bR\d{3}(?:\.\d+)+\b")
QUESTION_STOPWORDS = {
    "tell", "about", "what", "should", "know", "please", "thank", "thanks",
    "need", "want", "hello", "hey", "hi", "help", "info", "information",
    "building", "home", "house", "structure", "story", "stories", "bath",
    "room", "rooms", "with", "for", "this", "that", "the", "and", "like",
    "two", "three", "four"
}

# Ensure appdata scaffold; only load .env in local dev (not in container/App Hosting)
# Default collection and repo DB directory unless CLI/env override
try:
    # Removed ChromaDB specific env var check
    pass
except Exception:
    pass
os.environ.setdefault("RAG_COLLECTION_NAME", "docs_ibc_v2")

ensure_appdata_scaffold()
if not (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    try:
        # Try repo .env first, then fall back to appdata .env
        repo_env = Path(__file__).resolve().parent / ".env"
        if repo_env.exists():
            dotenv.load_dotenv(dotenv_path=repo_env, override=True)
        else:
            dotenv.load_dotenv(dotenv_path=get_env_file_path(), override=True)
    except Exception:
        pass

# Defer strict API key validation to runtime callers (Streamlit UI or CLI).
try:
    sanitize_and_validate_openai_key()
except Exception:
    pass


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    vector_store: VectorStore
    embedding_model: str
    collection_name: str = "docs_ibc_v2"
    vector_backend: str = "chroma"
    header_contains: Optional[str] = None
    source_contains: Optional[str] = None


class RagAgent:
    """RAG Agent for handling document insertion from URLs and local files."""

    def __init__(self, collection_name: Optional[str] = None, db_directory: str = "",
                 embedding_model: Optional[str] = None):
        # Resolve collection name with precedence: CLI > env > default
        resolved_collection = resolve_collection_name(collection_name)
        self.collection_name = resolved_collection
        self.db_directory = db_directory

        # Resolve vector backend and create appropriate store
        backend_name, backend_config = resolve_vector_backend()
        print(f"[agent] Vector backend: '{backend_name}'")

        # Log which embedding backend/model are active once during agent init
        emb_backend, emb_model = resolve_embedding_backend_and_model()
        self.embedding_model = embedding_model or emb_model
        print(f"[agent] Embeddings backend='{emb_backend}', model='{self.embedding_model}'")

        # Create vector store based on backend
        self.vector_store = get_vector_store(backend="bigquery")
    
    def smart_chunk_markdown(self, markdown: str, max_len: int = 1000, overlap_chars: int = 150) -> List[str]:
        """Hierarchically split markdown by headers, then by characters with small overlap.

        Overlap is applied only between adjacent slices within the same header region.
        """
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
                                    chunk = h3[i:i+max_len].strip()
                                    if chunk:
                                        chunks.append(chunk)
                                    i += step
                            else:
                                chunks.append(h3)
                    else:
                        chunks.append(h2)
            else:
                chunks.append(h1)

        # Fallback: if no headers found, treat entire text as one chunk
        if not chunks:
            chunks = [markdown]

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
    
    def extract_section_info(self, chunk: str) -> Dict[str, Any]:
        """Extracts headers and stats from a chunk."""
        headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
        header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

        return {
            "headers": header_str,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        }
    
    async def insert_from_url(self, url: str, chunk_size: int = 1000, max_depth: int = 3, 
                             max_concurrent: int = 10, batch_size: int = 100, overlap_chars: int = 150) -> None:
        """Insert documents from a URL into the collection."""
        from insert_docs import (
            is_sitemap, is_txt, crawl_recursive_internal_links, 
            crawl_markdown_file, parse_sitemap, crawl_batch
        )
        
        # Detect URL type and crawl
        if is_txt(url):
            print(f"Detected .txt/markdown file: {url}")
            crawl_results = await crawl_markdown_file(url)
        elif is_sitemap(url):
            print(f"Detected sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                print("No URLs found in sitemap.")
                return
            crawl_results = await crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
        else:
            print(f"Detected regular URL: {url}")
            crawl_results = await crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent)
        
        # Process and insert chunks
        self._process_and_insert_chunks(crawl_results, chunk_size, batch_size, overlap_chars)
    
    async def insert_from_file(
        self,
        file_path: str,
        *,
        chunk_size: int = 1000,
        batch_size: int = 100,
        overlap_chars: int = 150,
        pdf_pages: Optional[str] = None,
        pdf_render_dpi: Optional[int] = None,
        pdf_ocr_lang: str = "eng",
        pdf_ocr_psm: int = 6,
        pdf_ocr_oem: int = 3,
        pdf_diagnostic_dir: Optional[str] = None,
    ) -> None:
        """Insert documents from a local file into the collection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist.")
            return
        
        print(f"Processing local file: {file_path}")
        
        # Read the file content based on file type
        try:
            if file_path.suffix.lower() == '.pdf':
                pdf_chunks = self._read_pdf(
                    file_path,
                    page_range=pdf_pages,
                    render_dpi=pdf_render_dpi,
                    ocr_lang=pdf_ocr_lang,
                    ocr_psm=pdf_ocr_psm,
                    ocr_oem=pdf_ocr_oem,
                    diagnostic_dir=(Path(pdf_diagnostic_dir) if pdf_diagnostic_dir else None),
                )
                # Build crawl_results in a uniform shape with per-chunk page numbers
                crawl_results = [
                    {"url": str(file_path), "text": ch.get("text", ""), "page_number": ch.get("page"), "title": ch.get("title", "")}
                    for ch in pdf_chunks
                ]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                crawl_results = [{"url": str(file_path), "markdown": content, "title": file_path.stem}]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return
        
        # Process and insert chunks
        self._process_and_insert_chunks(crawl_results, chunk_size, batch_size, overlap_chars)
    
    def _read_pdf(
        self,
        file_path: Path,
        *,
        page_range: Optional[str] = None,
        render_dpi: Optional[int] = 300,
        ocr_lang: str = "eng",
        ocr_psm: int = 6,
        ocr_oem: int = 3,
        diagnostic_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Extract text and images from a PDF file using enhanced processing.

        Returns a list of chunk dicts with keys: text, page, title.
        """
        try:
            from pdf_loader.pdf_loader import process_pdf
            import tempfile
            import json
            
            # Create temporary directories for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                output_json = temp_path / "pdf_chunks.json"
                image_dir = temp_path / "images"
                
                # Process PDF with image extraction and OCR
                chunks = process_pdf(
                    file_path,
                    output_json,
                    image_dir,
                    chunk_size=1000,
                    page_range=page_range,
                    render_pages_dpi=render_dpi,
                    ocr_language=ocr_lang,
                    ocr_psm=ocr_psm,
                    ocr_oem=ocr_oem,
                    diagnostic_dir=diagnostic_dir,
                )
                # Discover title from PDF metadata if available
                pdf_title = ""
                try:
                    import fitz  # PyMuPDF
                    with fitz.open(file_path) as doc:
                        pdf_title = (doc.metadata or {}).get("title") or ""
                except Exception:
                    pdf_title = ""
                if not pdf_title:
                    pdf_title = file_path.stem

                # Attach title to each chunk
                for ch in chunks:
                    ch["title"] = pdf_title
                return chunks
                
        except ImportError as e:
            print(f"Error: Required dependencies not installed: {e}")
            print("Please install: pip install PyMuPDF pytesseract Pillow")
            raise ImportError("PDF processing dependencies not installed")
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            # Fallback to basic text extraction
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return [{"text": text, "page": None, "title": file_path.stem}]
            except Exception as fallback_error:
                print(f"Fallback text extraction also failed: {fallback_error}")
                raise
    
    def _process_and_insert_chunks(self, crawl_results: List[Dict[str, Any]], chunk_size: int, batch_size: int, overlap_chars: int) -> None:
        """Process crawl results and insert chunks into the collection.

        Standardizes metadata across sources and writes stable ids.
        """
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Track per-source summary
        per_source_ids: Dict[str, List[str]] = {}

        # Determine embedding info for metadata
        backend, model = resolve_embedding_backend_and_model()

        # Per-section counters for stable local indices
        section_counters: Dict[str, int] = {}

        for doc in crawl_results:
            raw_url = str(doc.get('url', ''))
            normalized_url = normalize_source_url(raw_url)

            # Determine source type and mime
            lower_url = raw_url.lower()
            if lower_url.startswith('http://') or lower_url.startswith('https://'):
                source_type = 'web'
                mime_type = 'text/html'
            else:
                suffix = Path(raw_url).suffix.lower()
                if suffix == '.pdf':
                    source_type = 'pdf'
                    mime_type = 'application/pdf'
                elif suffix in {'.md', '.markdown'}:
                    source_type = 'markdown'
                    mime_type = 'text/markdown'
                elif suffix == '.txt':
                    source_type = 'txt'
                    mime_type = 'text/plain'
                else:
                    source_type = 'other'
                    mime_type = 'text/plain'

            # Title preference: doc-provided title, else first H1, else filename stem
            provided_title = str(doc.get('title') or '')
            content_for_title = str(doc.get('markdown') or doc.get('text') or '')
            h1_match = re.search(r'^#\s+(.+)$', content_for_title, re.MULTILINE)
            default_title = Path(raw_url).stem if not (lower_url.startswith('http') or lower_url.startswith('https')) else ''
            title = provided_title or (h1_match.group(1).strip() if h1_match else '') or default_title

            # Decide chunking vs pre-chunked
            prechunked = 'text' in doc and ('page_number' in doc or 'page' in doc)
            chunk_texts: List[str] = []
            page_numbers: List[Optional[int]] = []

            if prechunked:
                # single pre-chunked entry
                chunk_texts = [str(doc.get('text') or '')]
                page_numbers = [int(doc.get('page_number') or doc.get('page') or 0) or None]
            else:
                content = str(doc.get('markdown') or '')
                if source_type == 'pdf':
                    # Fallback in case; treat as plain text
                    chunk_texts = self._chunk_text_simple(content, max_len=chunk_size, overlap_chars=overlap_chars)
                else:
                    chunk_texts = self.smart_chunk_markdown(content, max_len=chunk_size, overlap_chars=overlap_chars)
                page_numbers = [None] * len(chunk_texts)

            # Build per-chunk metadata
            for chunk in chunk_texts:
                headers_info = self.extract_section_info(chunk)
                headers_text = headers_info.get('headers', '')
                section_path = build_section_path(chunk)
                if source_type == 'pdf' and not section_path:
                    section_path = title or ''
                page_num = page_numbers[0] if page_numbers else None

                # Increment local index per (source, section, page)
                ctr_key = f"{normalized_url}\n{section_path}\n{page_num if page_num is not None else ''}"
                local_idx = section_counters.get(ctr_key, 0)
                section_counters[ctr_key] = local_idx + 1

                meta = make_chunk_metadata(
                    source_url=raw_url,
                    source_type=source_type,
                    section_path=section_path,
                    headers=headers_text,
                    page_number=page_num,
                    chunk_text=chunk,
                    title=title,
                    mime_type=mime_type,
                    embedding_backend=backend,
                    embedding_model=model,
                    section_local_index=local_idx,
                )

                ids.append(meta['chunk_id'])
                documents.append(chunk)
                metadatas.append(meta)
                per_source_ids.setdefault(normalized_url, []).append(meta['chunk_id'])

        if not documents:
            print("No documents found to insert.")
            return

        # Check if vector store supports upsert (BigQuery does not support real-time upserts)
        try:
            # Deduplicate by chunk_id against existing collection
            existing_result = self.vector_store.get_by_ids(ids)
            existing = set(existing_result.get("ids", []))
            keep_indices = [i for i, cid in enumerate(ids) if cid not in existing]

            if not keep_indices:
                print(f"[ingest] All {len(ids)} chunks already exist; skipping insert.")
            else:
                ids_new = [ids[i] for i in keep_indices]
                docs_new = [documents[i] for i in keep_indices]
                metas_new = [metadatas[i] for i in keep_indices]

                print(f"Inserting {len(docs_new)} chunks into vector store '{self.collection_name}' (skipped {len(ids)-len(keep_indices)} duplicates)...")

                self.vector_store.upsert(
                    ids=ids_new,
                    documents=docs_new,
                    metadatas=metas_new,
                    batch_size=batch_size
                )

                print(f"Successfully added {len(docs_new)} chunks to vector store '{self.collection_name}'.")

            # Per-source summary logs
            for src, cids in per_source_ids.items():
                stype = 'web' if src.startswith('http') else ('pdf' if src.endswith('.pdf') else 'file')
                inserted = sum(1 for cid in cids if cid not in existing)
                skipped = len(cids) - inserted
                print(f"[ingest] source_type={stype} url={src} inserted={inserted} skipped_duplicates={skipped}")

        except NotImplementedError:
            # BigQuery doesn't support real-time upserts
            print(f"[ingest] Warning: Real-time insertion not supported for this vector backend.")
            print(f"[ingest] Please use batch ingestion scripts for BigQuery.")
            print(f"[ingest] Total chunks prepared: {len(documents)}")
            # Log the would-be insertions
            for src, cids in per_source_ids.items():
                stype = 'web' if src.startswith('http') else ('pdf' if src.endswith('.pdf') else 'file')
                print(f"[ingest] source_type={stype} url={src} chunks={len(cids)}")
    
    def _chunk_text_simple(self, text: str, max_len: int = 1000, overlap_chars: int = 150) -> List[str]:
        """Simple text chunking for non-markdown content."""
        if not text:
            return []
        
        # Simple character-based splitting with overlap applied between adjacent slices
        if not text:
            return []
        step = max(1, max_len - max(0, overlap_chars))
        pieces: List[str] = []
        i = 0
        while i < len(text):
            piece = text[i:i+max_len].strip()
            if piece:
                pieces.append(piece)
            i += step
        return pieces


def create_agent():
    """Create the RAG agent with proper environment variable loading."""
    if Agent is None:
        raise ImportError(
            "pydantic_ai is required to create the agent. Install it or avoid calling create_agent() at import time."
        )
    
    # Ensure correct model choice is loaded
    model_choice = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
    print(f"[agent] Initializing agent with model: {model_choice}")
    
    return Agent(
        model_choice,
        deps_type=RAGDeps,
        model_settings={"tool_choice": "required"},
        system_prompt="You are the CAL Engineering Assistant. Always rely on the supplied context and tools; if the context lacks the answer, say so."
    )

# Initialize agent as None and track the last key used to recreate if it changes
agent = None
_last_openai_key = None
_last_model_choice = None
_last_retrieve_context: str = ""


def build_retrieval_context(
    question: str,
    deps: RAGDeps,
    *,
    n_results: int = 5,
    header_contains: Optional[str] = None,
    source_contains: Optional[str] = None,
) -> str:
    """Retrieve context synchronously using the orchestrator and stash for verification."""
    embed_fn = create_embedding_function()
    orchestrator = RetrievalOrchestrator(deps.vector_store, embed_fn)

    def _norm(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    header_filter = _norm(header_contains) or _norm(getattr(deps, "header_contains", None))
    source_filter = _norm(source_contains) or _norm(getattr(deps, "source_contains", None))

    filters: Dict[str, str] = {}
    if header_filter:
        filters["header_contains"] = header_filter
    if source_filter:
        filters["source_contains"] = source_filter

    section_tokens = orchestrator.preprocessor.extract_section_tokens(question)
    table_ctx = orchestrator.lookup.lookup_tables(question, deps.vector_store, n_results)
    rule_ctx = orchestrator.lookup.lookup_rules(section_tokens, question)

    query_embedding = embed_fn([question])[0]
    vec_res = deps.vector_store.vector_search(query_embedding, n_results=n_results, filters=filters or None)
    kw_res = deps.vector_store.keyword_search(
        orchestrator.preprocessor.build_search_terms(question),
        max_results=max(3, n_results),
    )
    merged = orchestrator._merge_results(vec_res, kw_res, n_results)
    formatted_context = format_results_as_context(merged)

    structured_text, structured_entries = _build_structured_extracts(
        question,
        merged,
        rule_text=rule_ctx
    )

    parts: List[str] = []
    if structured_text:
        parts.append(structured_text)
    if rule_ctx:
        parts.append(rule_ctx)
    if table_ctx:
        parts.append(table_ctx)
    parts.append(formatted_context)

    context_text = "\n\n".join(part for part in parts if part)

    global _last_retrieve_context
    _last_retrieve_context = context_text
    increment_query_count()
    return context_text, structured_entries


def _build_structured_extracts(
    question: str,
    merged_results: Dict[str, Any],
    max_entries: int = 10,
    rule_text: str = "",
) -> tuple[str, List[Dict[str, str]]]:
    """Synthesize concise section/quote pairs from merged retrieval outputs."""
    docs = merged_results.get("documents", [[]])[0]
    metas = merged_results.get("metadatas", [[]])[0]

    entries: List[str] = []
    entry_objs: List[Dict[str, str]] = []
    seen_sections: set[str] = set()

    question_terms: set[str] = set()
    if question:
        raw_terms = re.findall(r"[a-zA-Z]{3,}", question.lower())
        question_terms = {term for term in raw_terms if term not in QUESTION_STOPWORDS}

    def _derive_code_label(meta: Dict[str, Any]) -> str:
        title = (meta or {}).get("title", "") or ""
        source_url = (meta or {}).get("source_url", "") or ""
        text = f"{title} {source_url}".lower()
        if "residential" in text or "irc" in text:
            return "IRC 2018"
        return "IBC 2018"

    def _clean_quote(snippet: str) -> str:
        snippet = snippet.replace(" | ", " ").replace("|", " ")
        snippet = " ".join(snippet.split())
        return snippet.strip()

    def _extract_quote_for_token(text: str, token: str) -> str:
        idx = text.lower().find(token.lower())
        if idx == -1:
            return _clean_quote(text[:220])
        line_start = text.rfind("\n", 0, idx)
        line_end = text.find("\n", idx)
        if line_start == -1:
            line_start = 0
        if line_end == -1:
            line_end = len(text)
        line_snippet = text[line_start:line_end]
        if len(line_snippet.strip()) > 40 and token in line_snippet:
            snippet = line_snippet
        else:
            start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx))
            if start == -1:
                start = max(0, idx - 140)
            else:
                start = max(0, start + 1)
            end_period = text.find(".", idx)
            end_newline = text.find("\n", idx)
            candidates = [pos for pos in [end_period, end_newline] if pos != -1]
            end = min(candidates) if candidates else idx + 200
            snippet = text[start:end]
            if len(snippet) < 40:
                snippet = text[idx - 120: idx + 120]
        return _clean_quote(snippet)

    for doc, meta in zip(docs or [], metas or []):
        if not doc:
            continue
        sections = SECTION_PATTERN.findall(doc)
        sections += IRC_SECTION_PATTERN.findall(doc)
        for section in sections:
            if section in seen_sections:
                continue
            quote = _extract_quote_for_token(doc, section)
            if not quote:
                continue
            if question_terms:
                haystack = f"{quote} {(meta or {}).get('section_path', '')}".lower()
                if not any(term in haystack for term in question_terms):
                    continue
            code_label = _derive_code_label(meta)
            source = meta.get("title") or meta.get("source_url") or "Unknown source"
            entries.append(f"- [{code_label} §{section}] \"{quote}\" (Source: {source})")
            entry_objs.append({
                "code_label": code_label,
                "section": section,
                "quote": quote,
                "source": source,
                "section_path": (meta or {}).get("section_path", ""),
            })
            seen_sections.add(section)
            if len(entries) >= max_entries:
                break
        if len(entries) >= max_entries:
            break

    if not entries:
        entries = []
        entry_objs = []

    if rule_text:
        for rule_entry in _parse_rule_snippets(rule_text):
            section = rule_entry["section"]
            if section in seen_sections:
                continue
            entries.append(
                f"- [{rule_entry['code_label']} §{section}] \"{rule_entry['quote']}\" (Source: {rule_entry['source']})"
            )
            entry_objs.append(rule_entry)
            seen_sections.add(section)
            if len(entries) >= max_entries:
                break

    if not entries:
        return "", entry_objs

    return "STRUCTURED CONTEXT EXTRACTS:\n" + "\n".join(entries), entry_objs


def _parse_rule_snippets(rule_text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not rule_text:
        return entries
    blocks = rule_text.split("RULE SNIPPET (")
    for block in blocks[1:]:
        header, _, body = block.partition("):")
        header = header.strip()
        body = body.strip()
        if not header or not body:
            continue
        section_match = re.search(r"§([0-9R\.]+)", header)
        if not section_match:
            continue
        section = section_match.group(1)
        code_label = header.split("§")[0].strip()
        bullets = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("- "):
                bullets.append(line[2:])
        if not bullets:
            continue
        quote = "; ".join(bullets)
        entries.append({
            "code_label": code_label,
            "section": section,
            "quote": quote,
            "source": "rules/snippets",
        })
    return entries


def get_agent():
    """Get or create the agent instance, recreating if critical env changes."""
    global agent, _last_openai_key, _last_model_choice

    current_key = os.getenv("OPENAI_API_KEY")
    current_model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")

    # Recreate the agent if not present or if the API key/model changed
    if agent is None or _last_openai_key != current_key or _last_model_choice != current_model:
        agent = create_agent()
        _register_tools(agent)
        _last_openai_key = current_key
        _last_model_choice = current_model
    
    return agent


async def retrieve(
    context: RunContext[RAGDeps],
    search_query: str,
    n_results: int = 5,
    header_contains: Optional[str] = None,
    source_contains: Optional[str] = None,
) -> str:
    """Retrieve relevant documents from vector store based on a search query.

    Args:
        context: The run context containing RAGDeps dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        header_contains: Optional filter for header content.
        source_contains: Optional filter for source content.

    Returns:
        Formatted context information from the retrieved documents.
    """
    # Guardrails and circuit breaker configuration
    try:
        max_n_results = max(1, int(str(os.getenv("RETRIEVAL_N_RESULTS", str(n_results))).strip()))
    except Exception:
        max_n_results = max(1, n_results)
    
    # Initialize orchestrator
    embed_fn = create_embedding_function()
    orchestrator = RetrievalOrchestrator(context.deps.vector_store, embed_fn)
    
    # Ensure rules are loaded (no-op if already)
    try:
        load_all_rules("rules")
    except Exception:
        pass

    # Coalesce filter values
    def _norm(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = s.strip()
        return s2 if s2 else None

    header_filter = _norm(header_contains) or _norm(getattr(context.deps, "header_contains", None))
    source_filter = _norm(source_contains) or _norm(getattr(context.deps, "source_contains", None))
    
    filters = {}
    if header_filter: filters["header_contains"] = header_filter
    if source_filter: filters["source_contains"] = source_filter

    try:
        final_ctx = orchestrator.retrieve(
            query=search_query,
            n_results=min(n_results, max_n_results),
            filters=filters
        )
        
        # Stash for verification step
        global _last_retrieve_context
        _last_retrieve_context = final_ctx
        
        increment_query_count()
        return final_ctx
        
    except Exception as e:
        print(f"[retrieve] Error: {e}")
        return "Error retrieving documents. Please try again."


def _register_tools(agent_instance: Agent) -> None:
    """Register all tools on the provided agent instance."""
    agent_instance.tool(retrieve)
    # Calculation tools (pure functions; no RunContext)
    agent_instance.tool_plain(deflection_limit)
    agent_instance.tool_plain(vehicle_barrier_reaction)
    agent_instance.tool_plain(fall_anchor_design_load)
    agent_instance.tool_plain(wind_speed_category)
    agent_instance.tool_plain(machinery_impact_factor)


async def run_rag_agent(
    question: str,
    collection_name: Optional[str] = None,
    db_directory: str = "",
    embedding_model: Optional[str] = None,
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question.

    Args:
        question: The question to answer.
        collection_name: Name of the collection to use (for Chroma backend).
        db_directory: Directory where vector store data is stored (for Chroma backend).
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.

    Returns:
        The agent's response.
    """
    # Resolve collection once for this run
    resolved_collection = resolve_collection_name(collection_name)

    # Resolve vector backend and create appropriate store
    backend_name, backend_config = resolve_vector_backend()

    if backend_name == "bigquery":
        vector_store = get_vector_store(backend="bigquery")
    else:
        # Fallback to BigQuery if unknown backend is resolved, or error out.
        # Since we removed Chroma, we default to BigQuery.
        vector_store = get_vector_store(backend="bigquery")

    # Create dependencies
    _, detected_model = resolve_embedding_backend_and_model()
    effective_model = embedding_model or detected_model

    deps = RAGDeps(
        vector_store=vector_store,
        embedding_model=effective_model,
        collection_name=resolved_collection if backend_name == "chroma" else "docs_ibc_v2",
        vector_backend=backend_name
    )

    context_text, context_entries = build_retrieval_context(
        question,
        deps,
        n_results=n_results,
        header_contains=getattr(deps, "header_contains", None),
        source_contains=getattr(deps, "source_contains", None),
    )

    if not context_text.strip():
        fallback_prompt = (
            "You do not have any retrieved reference material to cite. Respond to the user's question "
            "using your general building-code expertise. If the question requires specific code references "
            "that you cannot confirm, acknowledge that while still providing helpful guidance."
        )
        result = await get_agent().run(
            f"{fallback_prompt}\n\nUser question: {question}",
            deps=deps,
            model_settings={"temperature": 0.4},
        )
        return result.data

    augmented_question = (
        "Answer the user using ONLY the context provided. Cite sections in plain text (e.g., "
        "\"IBC 2018 §1507.2\") when possible. If the context does not contain the answer, reply "
        "with \"I don't have that information in the current documentation.\" Use a natural tone.\n\n"
        f"Context:\n{context_text}\n\nUser question: {question}"
    )
    result = await get_agent().run(augmented_question, deps=deps, model_settings={"temperature": 0.2})
    answer_text = result.data

    if context_entries:
        verification_context = "\n".join(
            f"{entry.get('code_label', '')} §{entry.get('section', '')} {entry.get('quote', '')}"
            for entry in context_entries
        )
    else:
        verification_context = context_text

    global _last_retrieve_context
    _last_retrieve_context = verification_context

    # Verify against last retrieval context
    ctx_text = verification_context or ""
    try:
        ver = verify_answer(answer_text, ctx_text)
    except Exception as e:
        print(f"[verify] error: {e}")
        return answer_text

    if ver.get("ok"):
        print("[verify] ok")
        return answer_text

    missing = ver.get("missing", {}) or {}
    mis_sections = missing.get("sections", []) or []
    mis_nums = missing.get("nums", []) or []
    mis_stds = missing.get("standards", []) or []
    print(f"[verify] missing sections={mis_sections} nums={mis_nums} standards={mis_stds}")

    if context_entries:
        # Deterministic extract already returned; skip LLM retry to avoid hallucinations.
        return answer_text

    # One focused retry with hint and keyword expansion
    retry_tokens: List[str] = [*mis_sections, *mis_nums, *mis_stds]
    retry_tokens = [t for t in retry_tokens if t]
    if retry_tokens:
        hint = (
            "Your previous draft omitted these tokens found in source: "
            + ", ".join(retry_tokens[:8])
            + ". You must include them verbatim and cite the section/table."
        )
        hinted_q = f"{hint}\n\n{question}"

        # Light keyword search to broaden candidates (non-blocking)
        try:
            _ = vector_store.keyword_search(retry_tokens, max_results=5)
        except Exception:
            pass

        result2 = await get_agent().run(hinted_q, deps=deps)
        answer2 = result2.data
        try:
            ver2 = verify_answer(answer2, _last_retrieve_context or ctx_text)
        except Exception:
            ver2 = {"ok": False}
        if ver2.get("ok"):
            print("[verify] ok (retry)")
        else:
            print("[verify] retry still missing")
        return answer2

    # If nothing actionable, return first draft
    return answer_text


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default=None, help="Name of the ChromaDB collection (overrides env)")
    parser.add_argument("--db-dir", default="", help="Directory where ChromaDB data is stored (default: AppData)")
    parser.add_argument("--embedding-model", default=None, help="Name of the embedding model to use (auto if omitted)")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Log resolved collection once for visibility
    resolved_name = resolve_collection_name(args.collection)
    print(f"[agent] Using ChromaDB collection: '{resolved_name}'")

    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=resolved_name,
        db_directory=args.db_dir,
        embedding_model=(args.embedding_model or None),
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
