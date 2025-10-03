"""Pydantic AI agent that leverages RAG with a local ChromaDB for Pydantic documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio
import chromadb
import re
from pathlib import Path

import dotenv
# Guarded imports for pydantic_ai to tolerate version differences
try:
    from pydantic_ai.agent import Agent  # preferred path
except Exception:
    try:
        from pydantic_ai import Agent  # fallback path
    except Exception:
        Agent = None  # type: ignore
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # type: ignore

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context,
    add_documents_to_collection,
    keyword_search_collection,
    build_section_search_terms,
    resolve_collection_name,
    resolve_embedding_backend_and_model,
    normalize_source_url,
    build_section_path,
    make_chunk_metadata,
    get_env_file_path,
    ensure_appdata_scaffold,
    get_default_chroma_dir,
    sanitize_and_validate_openai_key,
    get_key_diagnostics,
)
from utils_tables import parse_markdown_table, pick_underlayment
import itertools
from tools_calc import (
    deflection_limit,
    vehicle_barrier_reaction,
    fall_anchor_design_load,
    wind_speed_category,
    machinery_impact_factor,
)
from rules_loader import load_all_rules, find_rules_by_section, find_rules_by_keywords
from verify import verify_answer
import json as _json
import uuid as _uuid
from utils import increment_query_count

# Ensure appdata scaffold; only load .env in local dev (not in container/App Hosting)
# Default collection and repo DB directory unless CLI/env override
try:
    repo_chroma = str((Path(__file__).resolve().parent / "chroma_db").absolute())
    if not os.getenv("CHROMA_DIR") and Path(repo_chroma).exists():
        os.environ.setdefault("CHROMA_DIR", repo_chroma)
except Exception:
    pass
os.environ.setdefault("RAG_COLLECTION_NAME", "docs_ibc_v2")

ensure_appdata_scaffold()
if not (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    try:
        dotenv.load_dotenv(dotenv_path=get_env_file_path(), override=False)
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
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str
    header_contains: Optional[str] = None
    source_contains: Optional[str] = None


class RagAgent:
    """RAG Agent for handling document insertion from URLs and local files."""
    
    def __init__(self, collection_name: Optional[str] = None, db_directory: str = "", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        # Resolve collection name with precedence: CLI > env > default
        resolved_collection = resolve_collection_name(collection_name)
        self.collection_name = resolved_collection
        self.db_directory = db_directory or get_default_chroma_dir()
        self.embedding_model = embedding_model
        self.client = get_chroma_client(db_directory)
        # Log which embedding backend/model are active once during agent init
        backend, model = resolve_embedding_backend_and_model()
        print(f"[agent] Embeddings backend='{backend}', model='{model}'")
        self.collection = get_or_create_collection(
            self.client, 
            resolved_collection, 
            embedding_model_name=embedding_model
        )
    
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

        # Deduplicate by chunk_id against existing collection
        from utils import get_existing_ids
        existing = get_existing_ids(self.collection, ids)
        keep_indices = [i for i, cid in enumerate(ids) if cid not in existing]
        if not keep_indices:
            print(f"[ingest] All {len(ids)} chunks already exist; skipping insert.")
        else:
            ids_new = [ids[i] for i in keep_indices]
            docs_new = [documents[i] for i in keep_indices]
            metas_new = [metadatas[i] for i in keep_indices]

            print(f"Inserting {len(docs_new)} chunks into ChromaDB collection '{self.collection_name}' (skipped {len(ids)-len(keep_indices)} duplicates)...")

            add_documents_to_collection(
                self.collection,
                ids_new,
                docs_new,
                metas_new,
                batch_size=batch_size
            )

            print(f"Successfully added {len(docs_new)} chunks to ChromaDB collection '{self.collection_name}'.")

        # Per-source summary logs
        for src, cids in per_source_ids.items():
            stype = 'web' if src.startswith('http') else ('pdf' if src.endswith('.pdf') else 'file')
            inserted = sum(1 for cid in cids if cid not in existing)
            skipped = len(cids) - inserted
            print(f"[ingest] source_type={stype} url={src} inserted={inserted} skipped_duplicates={skipped}")
    
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


# Create the RAG agent
def create_agent():
    """Create the RAG agent with proper environment variable loading."""
    if Agent is None:
        raise ImportError(
            "pydantic_ai is required to create the agent. Install it or avoid calling create_agent() at import time."
        )
    return Agent(
        os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
        deps_type=RAGDeps,
        system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                      "Use the retrieve tool to get relevant information from the documentation before answering. "
                      "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                      "in the current documentation and provide your best general knowledge response. "
                      "When you call a calculation tool, echo the input and output clearly in your answer, e.g., "
                      "'For a 30 ft span with L/180: Max deflection = 2.0 in (IBC Table 1604.3). "
                      "If a RULE SNIPPET is present, quote its values verbatim first, cite the section number, "
                      "and avoid unrelated standards unless explicitly asked. "
                      "If the retrieved context contains a section/table token and exact numeric terms, you MUST include those exact tokens in the answer and cite the section/table (e.g., 'IBC 2018 §1607.9' or 'Table 1604.3'). Prefer brevity and precision."
    )

# Initialize agent as None and track the last key used to recreate if it changes
agent = None
_last_openai_key = None
_last_model_choice = None
_last_retrieve_context: str = ""


def get_agent():
    """Get or create the agent instance, recreating if critical env changes."""
    global agent, _last_openai_key, _last_model_choice

    current_key = os.getenv("OPENAI_API_KEY")
    current_model = os.getenv("MODEL_CHOICE", "gpt-4.1-mini")

    # Recreate the agent if not present or if the API key/model changed
    if agent is None or _last_openai_key != current_key or _last_model_choice != current_model:
        agent = create_agent()
        _register_tools(agent)
        _last_openai_key = current_key
        _last_model_choice = current_model
    
    return agent

from typing import Any as _Any

async def retrieve(
    context: _Any,
    search_query: str,
    n_results: int = 5,
    header_contains: Optional[str] = None,
    source_contains: Optional[str] = None,
) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    # Guardrails and circuit breaker configuration
    try:
        max_n_results = max(1, int(str(os.getenv("RETRIEVAL_N_RESULTS", str(n_results))).strip()))
    except Exception:
        max_n_results = max(1, n_results)
    try:
        max_chunks_for_context = max(1, int(str(os.getenv("MAX_CHUNKS_FOR_CONTEXT", "12")).strip()))
    except Exception:
        max_chunks_for_context = 12
    try:
        max_context_tokens = max(256, int(str(os.getenv("MAX_CONTEXT_TOKENS", "6000")).strip()))
    except Exception:
        max_context_tokens = 6000

    import time
    start_time = time.time()
    breaker_triggered = False
    request_id = str(_uuid.uuid4())

    # Get ChromaDB client and collection (reused via utils caches)
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    # Ensure rules are loaded (no-op if already)
    try:
        load_all_rules("rules")
    except Exception:
        pass
    # ===== Pre-vector: Specialized table lookup for certain materials and wind speeds =====
    table_prefix_context: str = ""
    try:
        ql = (search_query or "").lower()
        known_materials = [
            # Slate
            "slate shingles",
            "slate shingle",
            "slate shingle roof",
            # Asphalt
            "asphalt shingles",
            "asphalt shingle",
            "asphalt shingle roof",
            # Vehicle barriers
            "vehicle barrier",
            "vehicle barriers",
        ]
        material_hit = next((m for m in known_materials if m in ql), None)
        mph_match = re.search(r"(\d{2,3})\s*mph", ql)
        wind_mph_val = int(mph_match.group(1)) if mph_match else None

        if material_hit and wind_mph_val is not None:
            # Scan candidate chunks by keyword for tables
            substrings = [material_hit, "table", "underlayment", "1507"]
            scan = keyword_search_collection(collection, substrings, max_results=max(15, n_results))
            docs = scan.get("documents", [[]])[0]
            metas = scan.get("metadatas", [[]])[0]

            chosen_value: Optional[str] = None
            citation: str = ""

            for doc, meta in zip(docs, metas):
                for df in parse_markdown_table(doc or ""):
                    val = pick_underlayment(df, material_hit, wind_mph_val)
                    if val:
                        chosen_value = val
                        # Build citation: try to detect table number and code name
                        table_num = None
                        m = re.search(r"\btable\s+([0-9]+(?:\.[0-9]+)*)", (doc or ""), flags=re.IGNORECASE)
                        if m:
                            table_num = m.group(1)
                        else:
                            sp = str((meta or {}).get("section_path") or (meta or {}).get("headers") or "")
                            m2 = re.search(r"([0-9]+(?:\.[0-9]+)+)", sp)
                            if m2:
                                table_num = m2.group(1)

                        title = str((meta or {}).get("title") or "")
                        src = str((meta or {}).get("source_url") or "")
                        code_short = ""
                        mapping = [
                            ("international building code", "IBC"),
                            ("international existing building code", "IEBC"),
                            ("international residential code", "IRC"),
                        ]
                        hay = f"{title} {src}".lower()
                        for needle, short in mapping:
                            if needle in hay:
                                code_short = short
                                break
                        if code_short and table_num:
                            citation = f"{code_short} Table {table_num}"
                        elif table_num:
                            citation = f"Table {table_num}"
                        elif code_short:
                            citation = code_short
                        else:
                            citation = "Table lookup"

                        # Compose prefix context
                        mat_pretty = material_hit
                        table_prefix_context = (
                            f"TABLE LOOKUP RESULT: Underlayment for '{mat_pretty}' at wind speed {wind_mph_val} mph: {chosen_value} ({citation})\n\n"
                        )
                        print(f"[table_lookup] match material='{mat_pretty}' wind_mph={wind_mph_val} -> '{chosen_value}' [{citation}]")
                        break
                if chosen_value:
                    break
    except Exception as e:
        # Fail open; do not block retrieval on table parsing errors
        print(f"[table_lookup] Error during table scan: {e}")

    # ===== Section locking: restrict candidates to chunks matching section tokens =====
    section_tokens = re.findall(r"\b\d+(?:\.\d+)+\b", search_query or "")
    lock_applied = False
    lock_candidates = None
    if section_tokens:
        # Build ordered fallbacks: for each token, include itself then shorter prefixes
        def prefixes(tok: str) -> List[str]:
            parts = tok.split('.')
            out = [tok]
            if len(parts) >= 2:
                out.append('.'.join(parts[:2]))
            if len(parts) >= 1:
                out.append(parts[0])
            # Dedup preserve order
            seen = set()
            res = []
            for t in out:
                if t not in seen:
                    seen.add(t)
                    res.append(t)
            return res

        token_fallbacks = [prefixes(t) for t in section_tokens]
        # Flatten by priority tiers across tokens: first try full sections for all, etc.
        tiers = list(itertools.zip_longest(*token_fallbacks))
        tiers_flat: List[str] = []
        for tier in tiers:
            for t in tier:
                if t:
                    tiers_flat.append(t)

        # Initial candidate pool: vector + keyword merge similar to below but larger for filtering
        base_results = query_collection(collection, search_query, n_results=max(n_results, 25))
        base_ids = [*base_results.get("ids", [[]])[0]]
        base_docs = [*base_results.get("documents", [[]])[0]]
        base_metas = [*base_results.get("metadatas", [[]])[0]]

        # Add keyword scan candidates to broaden the pool
        kw_scan = keyword_search_collection(collection, [search_query], max_results=100)
        for i, id_ in enumerate(kw_scan.get("ids", [[]])[0]):
            if id_ not in base_ids:
                base_ids.append(id_)
                base_docs.append(kw_scan.get("documents", [[]])[0][i])
                base_metas.append(kw_scan.get("metadatas", [[]])[0][i])

        total_before = len(base_ids)

        kept_ids: List[str] = []
        kept_docs: List[str] = []
        kept_metas: List[Dict[str, Any]] = []

        # Try tiers in order until we have any matches
        for tier_token in tiers_flat:
            ids_t: List[str] = []
            docs_t: List[str] = []
            metas_t: List[Dict[str, Any]] = []
            tt_l = tier_token.lower()
            for id_, doc, meta in zip(base_ids, base_docs, base_metas):
                meta = meta or {}
                header_text = str(meta.get("section_path") or meta.get("headers") or meta.get("title") or meta.get("header") or "")
                text = f"{header_text}\n{doc}".lower()
                if tt_l in text:
                    ids_t.append(id_)
                    docs_t.append(doc)
                    metas_t.append(meta)
            if ids_t:
                kept_ids, kept_docs, kept_metas = ids_t, docs_t, metas_t
                lock_applied = True
                print(f"[retrieve] section_lock applied: tokens=['{tier_token}']; kept={len(kept_ids)}/{total_before} candidates")
                break

        if lock_applied:
            lock_candidates = {
                "ids": [kept_ids[:n_results]],
                "documents": [kept_docs[:n_results]],
                "metadatas": [kept_metas[:n_results]],
                "distances": [[0.0] * min(n_results, len(kept_ids))],
            }

    # ===== Rule snippets injection (before formatting/vector context) =====
    rules_prefix_context: str = ""
    try:
        # Prefer exact section tokens; else keyword-based search
        rule_hits: List[Dict[str, Any]] = []
        if section_tokens:
            # Try exact then prefixes
            for tok in section_tokens:
                # exact
                rh = find_rules_by_section(tok)
                if rh:
                    rule_hits = rh
                    break
                # prefix fallbacks
                parts = tok.split('.')
                for i in range(len(parts)-1, 0, -1):
                    prefix = '.'.join(parts[:i])
                    rh = find_rules_by_section(prefix)
                    if rh:
                        rule_hits = rh
                        break
                if rule_hits:
                    break
        if not rule_hits:
            # Keyword-based lookup with simple tokens from the query
            kw_tokens = re.findall(r"[a-zA-Z]+", search_query or "")
            candidates = find_rules_by_keywords(kw_tokens)
            if candidates:
                # take the highest-scoring rule group
                top_rule = candidates[0][0]
                rule_hits = [top_rule]

        if rule_hits:
            # For simplicity, display the first matching rule (could be extended)
            r = rule_hits[0]
            sec = str(r.get("sec") or "").strip()
            title = str(r.get("title") or "").strip()
            ver = str(r.get("version") or "").strip()
            items = r.get("items", []) or []
            header = f"RULE SNIPPET ({ver} §{sec} – {title}):\n"
            lines = []
            for it in items:
                label = str(it.get("label") or "").strip()
                value = str(it.get("value") or "").strip()
                note = str(it.get("note") or "").strip()
                if note:
                    lines.append(f"- {label}: {value} ({note})")
                else:
                    lines.append(f"- {label}: {value}")
            rules_prefix_context = header + "\n".join(lines) + "\n\n"

            # Log injected snippet
            tokens_preview = re.findall(r"[a-zA-Z]+", search_query or "")[:5]
            print(f"[rules] injected §{sec} ({len(items)} items) for query tokens={tokens_preview}")
    except Exception as e:
        print(f"[rules] Error during snippet injection: {e}")

    # Vector query first (bounded)
    n_results = min(n_results, max_n_results)
    query_results = query_collection(collection, search_query, n_results=n_results)

    # Hybrid fallback for numeric/table/section queries
    section_terms = build_section_search_terms(search_query)
    kw_results = keyword_search_collection(collection, section_terms, max_results=max(3, n_results))

    # Merge vector and keyword hits (dedupe by id, prefer vector order)
    # Apply candidate discipline: cap candidates before rerank/format
    try:
        candidate_cap_raw = os.getenv("VECTOR_CANDIDATE_CAP", "50")
        candidate_cap = max(1, int(str(candidate_cap_raw).strip()))
    except Exception:
        candidate_cap = 50
    vec_ids = [*query_results.get("ids", [[]])[0]]
    vec_docs = [*query_results.get("documents", [[]])[0]]
    vec_metas = [*query_results.get("metadatas", [[]])[0]]

    kw_ids = kw_results.get("ids", [[]])[0]
    kw_docs = kw_results.get("documents", [[]])[0]
    kw_metas = kw_results.get("metadatas", [[]])[0]

    seen = set(vec_ids)
    for i, id_ in enumerate(kw_ids):
        if id_ not in seen:
            vec_ids.append(id_)
            vec_docs.append(kw_docs[i])
            vec_metas.append(kw_metas[i])
            seen.add(id_)

    # Cap candidates feeding downstream steps
    candidates_from_vector = len(vec_ids)
    if len(vec_ids) > candidate_cap:
        vec_ids = vec_ids[:candidate_cap]
        vec_docs = vec_docs[:candidate_cap]
        vec_metas = vec_metas[:candidate_cap]
    try:
        print({
            "where": "retrieval",
            "action": "candidate_cap",
            "candidates_from_vector": candidates_from_vector,
            "candidates_after_cap": len(vec_ids),
        })
    except Exception:
        pass

    # Coalesce filter values: tool args override deps; treat blank/whitespace as None
    def _norm(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = s.strip()
        return s2 if s2 else None

    header_filter = _norm(header_contains) or _norm(getattr(context.deps, "header_contains", None))
    source_filter = _norm(source_contains) or _norm(getattr(context.deps, "source_contains", None))

    before_count = len(vec_ids)

    # Apply case-insensitive filtering if any filter is active
    if header_filter or source_filter:
        hf = header_filter.lower() if header_filter else None
        sf = source_filter.lower() if source_filter else None

        filtered_ids: List[str] = []
        filtered_docs: List[str] = []
        filtered_metas: List[Dict[str, Any]] = []

        for id_, doc, meta in zip(vec_ids, vec_docs, vec_metas):
            meta = meta or {}
            # Header fallback: section_path -> headers -> title -> header
            header_text = str(meta.get("section_path") or meta.get("headers") or meta.get("title") or meta.get("header") or "")
            header_text_l = header_text.lower()
            # Source fallback: source_url -> source -> file_path
            source_text = str(meta.get("source_url") or meta.get("source") or meta.get("file_path") or "")
            source_text_l = source_text.lower()

            ok_header = True if hf is None else (hf in header_text_l)
            ok_source = True if sf is None else (sf in source_text_l)

            if ok_header and ok_source:
                filtered_ids.append(id_)
                filtered_docs.append(doc)
                filtered_metas.append(meta)

        vec_ids, vec_docs, vec_metas = filtered_ids, filtered_docs, filtered_metas

        after_count = len(vec_ids)
        short_warn = ""
        if (header_filter and len(header_filter) < 2) or (source_filter and len(source_filter) < 2):
            short_warn = " (warning: very short filters can be noisy)"
        print(
            f"[retrieve] Filters header_contains='{header_filter or ''}', source_contains='{source_filter or ''}' candidates={before_count} -> filtered={after_count}{short_warn}"
        )

    # Handle no results after filtering
    if not vec_ids:
        notice = "No results matched the current filters. Consider removing or relaxing filters."
        if header_filter or source_filter:
            details = []
            if header_filter:
                details.append(f"header_contains='{header_filter}'")
            if source_filter:
                details.append(f"source_contains='{source_filter}'")
            if details:
                notice += " (" + ", ".join(details) + ")"
        query_results = {
            "ids": [["no-matches"]],
            "documents": [[notice]],
            "metadatas": [[{}]],
            "distances": [[0.0]],
        }
        return format_results_as_context(query_results)

    # Trim to n_results after optional filtering or section lock
    if lock_candidates:
        query_results = lock_candidates
    else:
        query_results = {
            "ids": [vec_ids[:n_results]],
            "documents": [vec_docs[:n_results]],
            "metadatas": [vec_metas[:n_results]],
            "distances": [[0.0] * min(n_results, len(vec_ids))],  # placeholder
        }
    
    # Format the results as context (prepend rules and table lookup if available)
    # Guard against huge contexts by limiting number of chunks
    if len(query_results.get("documents", [[]])[0]) > max_chunks_for_context:
        # Trim
        query_results = {
            "ids": [query_results["ids"][0][:max_chunks_for_context]],
            "documents": [query_results["documents"][0][:max_chunks_for_context]],
            "metadatas": [query_results["metadatas"][0][:max_chunks_for_context]],
            "distances": [query_results.get("distances", [[0.0]])[0][:max_chunks_for_context]],
        }
    # Optional dedup by (source_url, page_number)
    docs = query_results.get("documents", [[]])[0]
    metas = query_results.get("metadatas", [[]])[0]
    seen_pairs = set()
    dedup_docs: List[str] = []
    dedup_metas: List[Dict[str, Any]] = []
    for d, m in zip(docs, metas):
        key = (str((m or {}).get("source_url", "")), str((m or {}).get("page_number", "")))
        if key not in seen_pairs:
            seen_pairs.add(key)
            dedup_docs.append(d)
            dedup_metas.append(m)
    deduped_count = len(docs) - len(dedup_docs)
    query_results = {
        "ids": [query_results.get("ids", [[]])[0][:len(dedup_docs)]],
        "documents": [dedup_docs],
        "metadatas": [dedup_metas],
        "distances": [query_results.get("distances", [[0.0]])[0][:len(dedup_docs)]],
    }
    base_ctx = format_results_as_context(query_results)

    # Rough token estimate (~4 chars per token)
    # Trim overly long individual chunks prior to token cap to reduce spikes
    try:
        per_chunk_char_limit_raw = os.getenv("PER_CHUNK_CHAR_LIMIT", "4000")
        per_chunk_char_limit = max(500, int(str(per_chunk_char_limit_raw).strip()))
    except Exception:
        per_chunk_char_limit = 4000
    trimmed_chunks_count = 0
    if any(len(d or "") > per_chunk_char_limit for d in dedup_docs):
        trimmed_docs = []
        for d in dedup_docs:
            if len(d or "") > per_chunk_char_limit:
                trimmed_docs.append((d or "")[:per_chunk_char_limit])
                trimmed_chunks_count += 1
            else:
                trimmed_docs.append(d)
        query_results["documents"] = [trimmed_docs]
        base_ctx = format_results_as_context(query_results)

    est_tokens = max(1, len(base_ctx) // 4)
    if est_tokens > max_context_tokens:
        breaker_triggered = True
        elapsed = time.time() - start_time
        try:
            print({
                "where": "retrieval",
                "action": "circuit_breaker",
                "n_results_requested": n_results,
                "n_results_returned": len(query_results.get("documents", [[]])[0]),
                "chunks_used": len(query_results.get("documents", [[]])[0]),
                "tokens_in_context": est_tokens,
                "circuit_breaker_triggered": True,
                "elapsed_sec": round(elapsed, 3),
            })
        except Exception:
            pass
        # Per-request structured log without PII
        try:
            print(_json.dumps({
                "ts": time.time(),
                "request_id": request_id,
                "elapsed_ms": int(elapsed * 1000),
                "query_chars": len(search_query or ""),
                "embed_cache_hit": None,  # unknown here
                "embed_batches": None,
                "batch_size": None,
                "prefilter_count": None,
                "prefilter_limit": os.getenv("KEYWORD_PREFILTER_LIMIT", ""),
                "prefilter_fallback": None,
                "n_results_requested": n_results,
                "n_results_returned": len(query_results.get("documents", [[]])[0]),
                "candidates_from_vector": None,
                "candidates_after_cap": None,
                "deduped_count": None,
                "trimmed_chunks_count": None,
                "chunks_used": len(query_results.get("documents", [[]])[0]),
                "tokens_in_context": est_tokens,
                "circuit_breaker_triggered": True,
                "errors": "context_limit",
            }))
        except Exception:
            pass
        return (
            "Your request would exceed our current context limits."
            " Please narrow the question or specify a section/table so I can retrieve fewer chunks."
        )
    prefix = ""
    if rules_prefix_context:
        prefix += rules_prefix_context
    if table_prefix_context:
        prefix += table_prefix_context
    final_ctx = (prefix + base_ctx) if prefix else base_ctx
    # Stash for verification step
    global _last_retrieve_context
    _last_retrieve_context = final_ctx
    elapsed = time.time() - start_time
    try:
        print({
            "where": "retrieval",
            "action": "stats",
            "n_results_requested": n_results,
            "n_results_returned": len(query_results.get("documents", [[]])[0]),
            "chunks_used": len(query_results.get("documents", [[]])[0]),
            "tokens_in_context": est_tokens,
            "circuit_breaker_triggered": breaker_triggered,
            "elapsed_sec": round(elapsed, 3),
            "deduped_count": deduped_count,
            "trimmed_chunks_count": trimmed_chunks_count,
        })
    except Exception:
        pass
    # Per-request structured log without PII
    try:
        print(_json.dumps({
            "ts": time.time(),
            "request_id": request_id,
            "elapsed_ms": int(elapsed * 1000),
            "query_chars": len(search_query or ""),
            "embed_cache_hit": None,
            "embed_batches": None,
            "batch_size": None,
            "prefilter_count": None,
            "prefilter_limit": os.getenv("KEYWORD_PREFILTER_LIMIT", ""),
            "prefilter_fallback": None,
            "n_results_requested": n_results,
            "n_results_returned": len(query_results.get("documents", [[]])[0]),
            "candidates_from_vector": candidates_from_vector,
            "candidates_after_cap": len(query_results.get("documents", [[]])[0]),
            "deduped_count": deduped_count,
            "trimmed_chunks_count": trimmed_chunks_count,
            "chunks_used": len(query_results.get("documents", [[]])[0]),
            "tokens_in_context": est_tokens,
            "circuit_breaker_triggered": breaker_triggered,
            "errors": None,
        }))
    except Exception:
        pass
    increment_query_count()
    return final_ctx


def _register_tools(agent_instance: Agent) -> None:
    """Register all tools on the provided agent instance."""
    agent_instance.tool(retrieve)
    # Calculation tools (pure functions; no RunContext)
    agent_instance.tool(deflection_limit)
    agent_instance.tool(vehicle_barrier_reaction)
    agent_instance.tool(fall_anchor_design_load)
    agent_instance.tool(wind_speed_category)
    agent_instance.tool(machinery_impact_factor)


async def run_rag_agent(
    question: str,
    collection_name: Optional[str] = None,
    db_directory: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        collection_name: Name of the ChromaDB collection to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.
        
    Returns:
        The agent's response.
    """
    # Resolve collection once for this run
    resolved_collection = resolve_collection_name(collection_name)

    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory or get_default_chroma_dir()),
        collection_name=resolved_collection,
        embedding_model=embedding_model
    )
    
    # First draft
    result = await get_agent().run(question, deps=deps)
    answer_text = result.data

    # Verify against last retrieval context
    ctx_text = _last_retrieve_context or ""
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
            coll = get_or_create_collection(get_chroma_client(db_directory), resolved_collection, embedding_model_name=embedding_model)
            _ = keyword_search_collection(coll, retry_tokens, max_results=5)
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
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Log resolved collection once for visibility
    resolved_name = resolve_collection_name(args.collection)
    print(f"[agent] Using ChromaDB collection: '{resolved_name}'")

    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=resolved_name,
        db_directory=args.db_dir or get_default_chroma_dir(),
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
