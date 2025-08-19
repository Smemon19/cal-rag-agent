"""Utility functions for text processing and ChromaDB operations."""

import os
import pathlib
from typing import List, Dict, Any, Optional, Iterable, Set

import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched
import re
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import datetime, timezone
from pathlib import Path

_embedding_factory_logged: bool = False
_DEFAULT_OVERLAP_CHARS: int = 150

def resolve_overlap_chars(cli_value: Optional[int] = None) -> int:
    """Resolve chunk overlap in characters from CLI or env.

    - CLI value (if provided) takes precedence
    - Else use env OVERLAP_CHARS
    - Default to 150 when unset/invalid
    - Clamp to >= 0
    """
    if cli_value is not None:
        try:
            return max(0, int(cli_value))
        except Exception:
            return _DEFAULT_OVERLAP_CHARS
    raw = os.getenv("OVERLAP_CHARS", str(_DEFAULT_OVERLAP_CHARS))
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return _DEFAULT_OVERLAP_CHARS


def is_web_url(url: str) -> bool:
    s = (url or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def is_file_url(url: str) -> bool:
    s = (url or "").strip().lower()
    return s.startswith("file://")

def resolve_embedding_backend_and_model() -> tuple[str, str]:
    """Resolve embedding backend and model from environment with safe defaults.

    - EMBEDDING_BACKEND: "sentence" (default) or "openai"; invalid values fall back to "sentence" with a warning
    - SENTENCE_MODEL: default "all-MiniLM-L6-v2"
    - OPENAI_EMBED_MODEL: default "text-embedding-3-large"
    """
    backend_raw = os.getenv("EMBEDDING_BACKEND", "sentence")
    backend = (backend_raw or "").strip().lower() or "sentence"
    if backend not in {"sentence", "openai"}:
        print(f"[embeddings] Warning: Unknown EMBEDDING_BACKEND='{backend_raw}'. Falling back to 'sentence'.")
        backend = "sentence"

    if backend == "openai":
        model = (os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large") or "").strip() or "text-embedding-3-large"
    else:
        model = (os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2") or "").strip() or "all-MiniLM-L6-v2"

    return backend, model


def create_embedding_function():
    """Create and return a Chroma embedding function based on resolved backend/model.

    For 'openai', requires OPENAI_API_KEY to be set and non-empty.
    Logs the chosen backend/model once per process startup.
    """
    global _embedding_factory_logged
    backend, model = resolve_embedding_backend_and_model()

    if not _embedding_factory_logged:
        print(f"[embeddings] Using backend='{backend}', model='{model}'")
        _embedding_factory_logged = True

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai. Set it in your environment or .env."
            )
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    # Default: sentence-transformers
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)


def normalize_source_url(raw: str) -> str:
    """Normalize a source URL or filesystem path deterministically.

    - For http/https: lowercase scheme and netloc, remove fragment, remove common tracking params (utm_*, gclid, fbclid, ref),
      strip trailing slash (except root), preserve other query params.
    - For file paths: convert to absolute, prefix with file://, resolve symlinks where possible.
    """
    if not raw:
        return ""
    raw = str(raw).strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        # strip trailing slash except root
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # filter tracking params
        tracking_keys = {"gclid", "fbclid"}
        params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not (k.startswith("utm_") or k in tracking_keys or k == "ref")]
        query = urlencode(params)
        return urlunparse((scheme, netloc, path, "", query, ""))
    # file path
    p = Path(raw)
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    return f"file://{p.as_posix()}"


def build_section_path(markdown_chunk: str) -> str:
    """Build a hierarchy path like 'H1 > H2 > H3' from a markdown chunk, if present.

    Uses the first occurrence of #, ##, ### headers found in the chunk.
    """
    if not markdown_chunk:
        return ""
    headers = re.findall(r"^(#{1,3})\s+(.+)$", markdown_chunk, re.MULTILINE)
    # Map to levels 1..3
    levels: dict[int, str] = {}
    for hashes, text in headers:
        level = len(hashes)
        if 1 <= level <= 3 and level not in levels:
            levels[level] = text.strip()
    parts = [levels.get(1, ""), levels.get(2, ""), levels.get(3, "")]
    parts = [p for p in parts if p]
    return " > ".join(parts)


def compute_chunk_id(source_url: str, section_path: str, page_number: Optional[int], section_local_index: int) -> str:
    """Compute a deterministic chunk id from source + section + page + index.

    Returns a short hex string with a stable prefix.
    """
    base = f"{source_url}\n{section_path}\n{page_number if page_number is not None else ''}\n{section_local_index}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"c_{digest}"


def make_chunk_metadata(
    *,
    source_url: str,
    source_type: str,
    section_path: str,
    headers: str,
    page_number: Optional[int],
    chunk_text: str,
    title: str,
    mime_type: str,
    embedding_backend: str,
    embedding_model: str,
    section_local_index: int,
) -> Dict[str, Any]:
    """Construct standardized metadata for a chunk.

    All fields are strings unless specified; empty strings when missing.
    """
    normalized_url = normalize_source_url(source_url)
    char_count = len(chunk_text or "")
    word_count = len((chunk_text or "").split()) if chunk_text else 0
    content_preview = (chunk_text or "")[:180]
    chunk_id = compute_chunk_id(normalized_url, section_path or "", page_number, section_local_index)
    meta: Dict[str, Any] = {
        "source_url": normalized_url,
        "source_type": source_type or "",
        "section_path": section_path or "",
        "headers": headers or "",
        "page_number": page_number if page_number is not None else "",
        "char_count": char_count,
        "word_count": word_count,
        "chunk_id": chunk_id,
        "inserted_at": datetime.now(timezone.utc).isoformat(),
        "content_preview": content_preview,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "mime_type": mime_type or "",
        "title": title or "",
    }
    return meta

def get_default_collection_name() -> str:
    """Return the default ChromaDB collection name.

    Resolution order for default (no CLI override):
    - Use the environment variable `RAG_COLLECTION_NAME` if it is set to a non-empty, non-whitespace value
    - Otherwise, fall back to the literal default "docs"
    """
    env_value = os.getenv("RAG_COLLECTION_NAME", "")
    if env_value is None or env_value.strip() == "":
        return "docs"
    return env_value


def resolve_collection_name(cli_value: Optional[str] = None) -> str:
    """Resolve the final collection name with precedence: CLI > env > default.

    - If `cli_value` is provided and non-blank (not empty and not whitespace-only), use it as-is
    - Else use `get_default_collection_name()`
    """
    if cli_value is not None and cli_value.strip() != "":
        return cli_value
    return get_default_collection_name()


def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """Get a ChromaDB client with the specified persistence directory.
    
    Args:
        persist_directory: Directory where ChromaDB will store its data
        
    Returns:
        A ChromaDB PersistentClient
    """
    # Create the directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Return the client
    return chromadb.PersistentClient(persist_directory)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    distance_function: str = "cosine",
) -> chromadb.Collection:
    """Get an existing collection or create a new one if it doesn't exist.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection. If None or blank, resolves via `get_default_collection_name()`.
        embedding_model_name: Name of the embedding model to use
        distance_function: Distance function to use for similarity search
        
    Returns:
        A ChromaDB Collection
    """
    # Resolve name first
    resolved_name = collection_name if (collection_name is not None and collection_name.strip() != "") else get_default_collection_name()

    # Create embedding function via centralized factory
    embedding_func = create_embedding_function()
    backend, model = resolve_embedding_backend_and_model()
    
    # Try to get the collection, create it if it doesn't exist
    try:
        col = client.get_collection(
            name=resolved_name,
            embedding_function=embedding_func
        )
        # Warn if potential mixed embeddings (compare stored metadata when available)
        meta = getattr(col, "metadata", {}) or {}
        stored_backend = meta.get("embedding_backend")
        stored_model = meta.get("embedding_model")
        if stored_backend and (stored_backend != backend or (stored_model and stored_model != model)):
            print(
                f"[embeddings] Warning: Existing collection '{resolved_name}' was created with backend='{stored_backend}', model='{stored_model}'. "
                f"Current session uses backend='{backend}', model='{model}'. Mixing embeddings in one collection can degrade retrieval. "
                f"Consider using a different collection name."
            )
        else:
            # If metadata missing, still advise cautiously
            print(
                f"[embeddings] Opened existing collection '{resolved_name}'. If it was created with a different embedding backend, retrieval may degrade."
            )
        return col
    except Exception:
        return client.create_collection(
            name=resolved_name,
            embedding_function=embedding_func,
            metadata={
                "hnsw:space": distance_function,
                "embedding_backend": backend,
                "embedding_model": model,
            }
        )


def add_documents_to_collection(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 100,
) -> None:
    """Add documents to a ChromaDB collection in batches.
    
    Args:
        collection: ChromaDB collection
        ids: List of document IDs
        documents: List of document texts
        metadatas: Optional list of metadata dictionaries for each document
        batch_size: Size of batches for adding documents
    """
    # Create default metadata if none provided
    if metadatas is None:
        metadatas = [{}] * len(documents)
    
    # Create document indices
    document_indices = list(range(len(documents)))
    
    # Add documents in batches
    for batch in batched(document_indices, batch_size):
        # Get the start and end indices for the current batch
        start_idx = batch[0]
        end_idx = batch[-1] + 1  # +1 because end_idx is exclusive
        
        # Add the batch to the collection
        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )


def get_existing_ids(collection: chromadb.Collection, candidate_ids: Iterable[str], batch_size: int = 500) -> Set[str]:
    """Return the subset of candidate_ids that already exist in the collection.

    Uses batched collection.get(ids=[...]) calls to avoid loading the entire collection.
    """
    existing: Set[str] = set()
    batch: List[str] = []
    for cid in candidate_ids:
        batch.append(cid)
        if len(batch) >= batch_size:
            try:
                res = collection.get(ids=batch, include=[])
                existing.update(res.get("ids", []))
            except Exception:
                # Some providers may error on unknown ids; fallback to scanning
                pass
            batch = []
    if batch:
        try:
            res = collection.get(ids=batch, include=[])
            existing.update(res.get("ids", []))
        except Exception:
            pass
    return existing


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query a ChromaDB collection for similar documents.
    
    Args:
        collection: ChromaDB collection
        query_text: Text to search for
        n_results: Number of results to return
        where: Optional filter to apply to the query
        
    Returns:
        Query results containing documents, metadatas, distances, and ids
    """
    # Query the collection
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )


def format_results_as_context(query_results: Dict[str, Any]) -> str:
    """Format query results as a context string for the agent.
    
    Args:
        query_results: Results from a ChromaDB query
        
    Returns:
        Formatted context string
    """
    context = "CONTEXT INFORMATION:\n\n"
    
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
        query_results["distances"][0]
    )):
        # Add document information
        context += f"Document {i+1} (Relevance: {1 - distance:.2f}):\n"
        
        # Add metadata if available
        if metadata:
            for key, value in metadata.items():
                context += f"{key}: {value}\n"
        
        # Add document content
        context += f"Content: {doc}\n\n"
    
    return context


def keyword_search_collection(
    collection: chromadb.Collection,
    substrings: List[str],
    max_results: int = 5,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Search a collection by simple substring matching over documents.

    This is a hybrid fallback for numeric/section queries (e.g., "TABLE 1507.9.6").

    Args:
        collection: The ChromaDB collection to search.
        substrings: List of case-insensitive substrings to look for.
        max_results: Maximum number of matches to return.
        batch_size: Number of items to scan per batch.

    Returns:
        A dict similar to collection.query output shape with keys: documents, metadatas, ids.
    """
    normalized = [s.lower() for s in substrings if s]
    total = collection.count()
    results_docs: List[str] = []
    results_metas: List[Dict[str, Any]] = []
    results_ids: List[str] = []

    offset = 0
    while offset < total and len(results_docs) < max_results:
        res = collection.get(include=["documents", "metadatas"], limit=min(batch_size, total - offset), offset=offset)
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        ids = res.get("ids", [])
        for doc, meta, id_ in zip(docs, metas, ids):
            text = (doc or "").lower()
            if any(sub in text for sub in normalized):
                results_docs.append(doc)
                results_metas.append(meta)
                results_ids.append(id_)
                if len(results_docs) >= max_results:
                    break
        offset += batch_size

    return {
        "documents": [results_docs],
        "metadatas": [results_metas],
        "ids": [results_ids],
    }


def build_section_search_terms(query: str) -> List[str]:
    """Generate helpful substrings for section/table lookups.

    Extract sequences like 1507.9.6 and also partials (1507.9, 1507) to improve recall.
    """
    terms: List[str] = []
    q = query.strip()
    terms.append(q)

    # Common prefixes
    terms.extend([
        q.replace("section", "", 1).strip(),
        q.replace("Section", "", 1).strip(),
        q.replace("TABLE", "", 1).strip(),
        q.replace("Table", "", 1).strip(),
    ])

    # Extract dotted numeric references (e.g., 1507.9.6)
    matches = re.findall(r"\b\d+(?:\.\d+)+\b", q)
    for m in matches:
        parts = m.split(".")
        # full, then progressively shorter prefixes
        for i in range(len(parts), 0, -1):
            terms.append(".".join(parts[:i]))

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            unique_terms.append(t)
    return unique_terms
