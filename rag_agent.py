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
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

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
    
)

# Load environment variables from .env file, overriding any existing values
dotenv.load_dotenv(override=True)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


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
    
    def __init__(self, collection_name: Optional[str] = None, db_directory: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        # Resolve collection name with precedence: CLI > env > default
        resolved_collection = resolve_collection_name(collection_name)
        self.collection_name = resolved_collection
        self.db_directory = db_directory
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
    return Agent(
        os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
        deps_type=RAGDeps,
        system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                      "Use the retrieve tool to get relevant information from the documentation before answering. "
                      "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                      "in the current documentation and provide your best general knowledge response."
    )

# Initialize agent as None and track the last key used to recreate if it changes
agent = None
_last_openai_key = None
_last_model_choice = None


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

async def retrieve(
    context: RunContext[RAGDeps],
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
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Vector query first
    query_results = query_collection(collection, search_query, n_results=n_results)

    # Hybrid fallback for numeric/table/section queries
    section_terms = build_section_search_terms(search_query)
    kw_results = keyword_search_collection(collection, section_terms, max_results=max(3, n_results))

    # Merge vector and keyword hits (dedupe by id, prefer vector order)
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

    # Trim to n_results after optional filtering
    query_results = {
        "ids": [vec_ids[:n_results]],
        "documents": [vec_docs[:n_results]],
        "metadatas": [vec_metas[:n_results]],
        "distances": [[0.0] * min(n_results, len(vec_ids))],  # placeholder
    }
    
    # Format the results as context
    return format_results_as_context(query_results)


def _register_tools(agent_instance: Agent) -> None:
    """Register all tools on the provided agent instance."""
    agent_instance.tool(retrieve)


async def run_rag_agent(
    question: str,
    collection_name: Optional[str] = None,
    db_directory: str = "./chroma_db",
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
        chroma_client=get_chroma_client(db_directory),
        collection_name=resolved_collection,
        embedding_model=embedding_model
    )
    
    # Run the agent
    result = await get_agent().run(question, deps=deps)
    
    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default=None, help="Name of the ChromaDB collection (overrides env)")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
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
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
