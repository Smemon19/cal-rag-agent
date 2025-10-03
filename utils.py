"""Utility functions for text processing and ChromaDB operations."""

import os
import json
import pathlib
import time
import random
import psutil
from typing import List, Dict, Any, Optional, Iterable, Set, Tuple
from collections import OrderedDict

import chromadb
from more_itertools import batched
import re
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import datetime, timezone
from pathlib import Path
from utils_normalize import normalize_inequalities, expand_numeric_tokens, pair_unit_synonyms

_embedding_factory_logged: bool = False
_DEFAULT_OVERLAP_CHARS: int = 150
_EMBEDDING_FUNCTION_CACHE: Dict[Tuple[str, str], Any] = {}
_EMBEDDING_WRAPPER_CACHE: Dict[Tuple[str, str], Any] = {}
_CHROMA_CLIENT_CACHE: Dict[str, chromadb.PersistentClient] = {}
_COLLECTION_CACHE: Dict[Tuple[int, str], Any] = {}
_LAST_VECTOR_STORE_STATUS: Dict[str, Any] = {}
_PROCESS_START_TS: float = time.time()
_QUERY_COUNT: int = 0
_EMBED_LAST_JOB: Dict[str, Any] = {
    "batches": 0,
    "batch_size": 0,
    "cache_hit": False,
}

# --- Centralized API key and embedding diagnostics ---
_KEY_SANITIZED_PREFIX: str = ""
_KEY_VALIDATED: bool = False
_KEY_VALIDATION_ERROR: str = ""
_EMBEDDING_FINGERPRINT: Dict[str, Any] = {}


def get_appdata_base_dir() -> str:
    """Return the base writable application data directory for this app.

    - On Windows: %LOCALAPPDATA%\\CalRAG
    - On Cloud Run: /tmp/.calrag (writable ephemeral storage)
    - On other OS: ~/.calrag
    """
    # Prefer /tmp when running on Cloud Run (filesystem is read-only except /tmp)
    try:
        if os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN"):
            return str(Path("/tmp") / ".calrag")
    except Exception:
        pass
    try:
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return str(Path(local_appdata) / "CalRAG")
    except Exception:
        pass
    # Fallback for non-Windows
    return str(Path.home() / ".calrag")


def get_env_file_path() -> str:
    """Return the absolute path to the app's .env file in the appdata dir."""
    return str(Path(get_appdata_base_dir()) / ".env")


def get_default_chroma_dir() -> str:
    """Return the default path for Chroma persistence.

    Resolution order:
    1) Environment variable CHROMA_DIR (expanded, if set and non-empty)
    2) Repository-local 'chroma_db' directory (if exists and non-empty)
    3) Appdata directory fallback: ~/.calrag/chroma (or %LOCALAPPDATA%\\CalRAG\\chroma on Windows)
    """
    # 1) Explicit override via environment
    try:
        env_dir = os.getenv("CHROMA_DIR", "")
        if env_dir and str(env_dir).strip() != "":
            return str(Path(env_dir).expanduser())
    except Exception:
        pass

    # 2) Repository-local directory next to this file
    try:
        repo_root = Path(__file__).resolve().parent
        repo_chroma = repo_root / "chroma_db"
        if repo_chroma.exists():
            # Prefer repo path if it contains any files (i.e., a populated DB)
            has_any = False
            try:
                next(repo_chroma.iterdir())
                has_any = True
            except StopIteration:
                has_any = False
            except Exception:
                # If listing fails, be conservative and do not choose repo path
                has_any = False
            if has_any:
                return str(repo_chroma)
    except Exception:
        pass

    # 3) Fallback to appdata location
    return str(Path(get_appdata_base_dir()) / "chroma")


def ensure_appdata_scaffold() -> None:
    """Ensure base appdata directories exist and a template .env is present on first run."""
    base_dir = Path(get_appdata_base_dir())
    chroma_dir = Path(get_default_chroma_dir())
    base_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    env_path = Path(get_env_file_path())
    if not env_path.exists():
        # Minimal template for first run
        template = (
            "# Cal RAG Agent configuration\n"
            "# Fill in your API key if using OpenAI models.\n"
            "MODEL_CHOICE=gpt-4.1-mini\n"
            "OPENAI_API_KEY=\n"
            "# Embeddings backend: sentence (default) or openai\n"
            "EMBEDDING_BACKEND=sentence\n"
            "# Optional sentence-transformers model\n"
            "SENTENCE_MODEL=all-MiniLM-L6-v2\n"
            "# Default collection name for UI and ingest\n"
            "RAG_COLLECTION_NAME=docs\n"
        )
        try:
            env_path.write_text(template, encoding="utf-8")
        except Exception:
            # Best-effort; continue without blocking
            pass

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


def sanitize_and_validate_openai_key() -> None:
    """Sanitize OPENAI_API_KEY from environment and validate basic format.

    - Do not read from repo .env; only environment or OPENAI_API_KEY_FILE if provided by platform.
    - Trim whitespace and quotes; persist sanitized value to os.environ.
    - Validate prefix: sk-live- or sk-proj-; on invalid, record error for readiness gate.
    - Record masked prefix (first 5 chars + '…') for diagnostics only; never log full key.
    """
    global _KEY_SANITIZED_PREFIX, _KEY_VALIDATED, _KEY_VALIDATION_ERROR
    try:
        key = os.getenv("OPENAI_API_KEY", "")
        key_file = os.getenv("OPENAI_API_KEY_FILE", "")
        if (not key) and key_file:
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    key = f.read()
            except Exception:
                key = key
        if key:
            sanitized = key.strip().strip('"').strip("'")
            if sanitized and sanitized != key:
                os.environ["OPENAI_API_KEY"] = sanitized
                key = sanitized
        if not key:
            _KEY_VALIDATED = False
            _KEY_VALIDATION_ERROR = "OPENAI_API_KEY missing"
            _KEY_SANITIZED_PREFIX = ""
            return
        # Prefix validation
        if not (key.startswith("sk-live-") or key.startswith("sk-proj-")):
            _KEY_VALIDATED = False
            _KEY_VALIDATION_ERROR = "OPENAI_API_KEY has unexpected prefix"
        else:
            _KEY_VALIDATED = True
            _KEY_VALIDATION_ERROR = ""
        # Masked prefix for UI (first 5 chars)
        try:
            _KEY_SANITIZED_PREFIX = (key[:5] + "…") if key else ""
        except Exception:
            _KEY_SANITIZED_PREFIX = ""
    except Exception as e:
        _KEY_VALIDATED = False
        _KEY_VALIDATION_ERROR = f"sanitization_error: {e}"
        _KEY_SANITIZED_PREFIX = ""


def get_key_diagnostics() -> Dict[str, Any]:
    return {
        "present": _KEY_VALIDATED or bool(os.getenv("OPENAI_API_KEY")),
        "masked_prefix": _KEY_SANITIZED_PREFIX,
        "valid": _KEY_VALIDATED,
        "error": _KEY_VALIDATION_ERROR,
    }


def compute_embedding_fingerprint() -> Dict[str, Any]:
    """Compute a stable fingerprint for embedding settings.

    Returns dict with backend, model, dim (best-effort), and home path.
    """
    global _EMBEDDING_FINGERPRINT
    backend, model = resolve_embedding_backend_and_model()
    home = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME") or ""
    dim = None
    try:
        if backend == "sentence":
            from sentence_transformers import SentenceTransformer as _ST
            st_model = _ST(model)
            dim = int(getattr(getattr(st_model, "get_sentence_embedding_dimension", lambda: None)(), "__int__", lambda: None)() or 384)
        else:
            # Known dims for common OpenAI models (fallbacks)
            guess = {
                "text-embedding-3-large": 3072,
                "text-embedding-3-small": 1536,
            }
            dim = guess.get(os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"), 1536)
    except Exception:
        dim = 384 if backend == "sentence" else 1536
    _EMBEDDING_FINGERPRINT = {
        "backend": backend,
        "model": model,
        "dim": int(dim) if dim else None,
        "home": home,
    }
    return _EMBEDDING_FINGERPRINT


def create_embedding_function():
    """Create and return a Chroma embedding function based on resolved backend/model.

    For 'openai', requires OPENAI_API_KEY to be set and non-empty.
    Logs the chosen backend/model once per process startup.
    """
    global _embedding_factory_logged, _EMBEDDING_FUNCTION_CACHE
    backend, model = resolve_embedding_backend_and_model()

    if not _embedding_factory_logged:
        print(f"[embeddings] Using backend='{backend}', model='{model}'")
        _embedding_factory_logged = True

    # Return a cached embedding function instance when available to avoid
    # repeatedly loading heavy models (e.g., sentence-transformers/torch).
    cache_key = (backend, model)
    if cache_key in _EMBEDDING_FUNCTION_CACHE:
        return _EMBEDDING_FUNCTION_CACHE[cache_key]

    # Ensure Hugging Face caches are writable (Cloud Run: only /tmp is writable).
    # Route caches under the app's writable data dir to avoid read-only $HOME issues.
    try:
        hf_root = Path(get_appdata_base_dir()) / "hf_cache"
        hf_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_root))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_root / "sentence-transformers"))
        os.environ.setdefault("TORCH_HOME", str(hf_root / "torch"))
    except Exception:
        # Best effort; if this fails, downstream may still attempt defaults
        pass

    # Lazy-import to avoid importing heavy deps (e.g., torch via sentence-transformers)
    # during Streamlit startup/module scanning. This prevents watcher crashes.
    from chromadb.utils import embedding_functions as _embedding_functions

    # Determine optional bounded cache size for embeddings
    # Set EMBED_CACHE_SIZE=0 to disable caching
    try:
        embed_cache_size_raw = os.getenv("EMBED_CACHE_SIZE", "2048")
        embed_cache_size = max(0, int(str(embed_cache_size_raw).strip()))
    except Exception:
        embed_cache_size = 2048

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai. Set it in your environment or .env."
            )
        fn = _embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model)
        if embed_cache_size > 0:
            fn = _wrap_embedding_with_lru_cache(fn, backend, model, embed_cache_size)
        _EMBEDDING_FUNCTION_CACHE[cache_key] = fn
        return fn

    # Default: sentence-transformers
    # Proactively prewarm the model to avoid partial downloads in read-only/home-restricted envs
    try:
        try:
            hf_root = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("TRANSFORMERS_CACHE") or str(Path(get_appdata_base_dir()) / "hf_cache" / "sentence-transformers")
        except Exception:
            hf_root = None
        if hf_root:
            Path(hf_root).mkdir(parents=True, exist_ok=True)
        # Prewarm via sentence-transformers directly (ensures 1_Pooling/config.json etc.)
        from sentence_transformers import SentenceTransformer as _ST
        st_model = _ST(model, cache_folder=hf_root if hf_root else None)
        _ = st_model.encode(["warmup"], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
        try:
            del st_model
        except Exception:
            pass
    except Exception as e:
        try:
            print(json.dumps({"where": "embeddings", "action": "prewarm_error", "error": str(e)}))
        except Exception:
            pass

    fn = _embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    if embed_cache_size > 0:
        fn = _wrap_embedding_with_lru_cache(fn, backend, model, embed_cache_size)
    _EMBEDDING_FUNCTION_CACHE[cache_key] = fn
    return fn


def _wrap_embedding_with_lru_cache(base_fn: Any, backend: str, model: str, capacity: int) -> Any:
    """Wrap a chromadb embedding function with a bounded LRU cache.

    The underlying embedding function is expected to be callable with a list[str]
    and return a list[list[float]]. We cache per-text embeddings keyed by a
    short, stable hash of (model + "|" + text).
    """
    cache_key = (backend, model)
    if cache_key in _EMBEDDING_WRAPPER_CACHE:
        return _EMBEDDING_WRAPPER_CACHE[cache_key]

    class LruEmbeddingWrapper:
        def __init__(self, inner: Any, model_name: str, max_size: int) -> None:
            self._inner = inner
            self._model_name = model_name
            self._max_size = max_size
            self._store: "OrderedDict[str, list[float]]" = OrderedDict()
            self._hits: int = 0
            self._misses: int = 0

        def name(self) -> str:
            # Expose a stable name for Chroma's embedding function interface
            try:
                inner_name = getattr(self._inner, "name", None)
                if callable(inner_name):
                    return inner_name()
            except Exception:
                pass
            return f"lru::{self._model_name}"

        def _hash_key(self, text: str) -> str:
            # Short, stable hash for memory efficiency
            digest = hashlib.sha1((self._model_name + "|" + (text or "")).encode("utf-8")).hexdigest()
            return digest[:16]

        def __call__(self, input: List[str]) -> List[List[float]]:
            # Fast-path for empty
            if not input:
                return []

            # Determine cache membership
            keys = [self._hash_key(t) for t in input]
            missing_indices: List[int] = []
            for idx, key in enumerate(keys):
                if key not in self._store:
                    missing_indices.append(idx)

            # Miss path: compute embeddings for uncached texts with batching and retry
            if missing_indices:
                # Batching: do not accumulate all intermediates in memory
                try:
                    batch_size = max(1, int(str(os.getenv("EMBED_BATCH_SIZE", "64")).strip()))
                except Exception:
                    batch_size = 64
                to_compute = [input[i] for i in missing_indices]
                total = len(to_compute)
                total_batches = (total + batch_size - 1) // batch_size
                start = time.time()
                for batch_index in range(total_batches):
                    b_start = batch_index * batch_size
                    b_end = min(total, b_start + batch_size)
                    batch = to_compute[b_start:b_end]
                    t0 = time.time()
                    # Retry once with jitter on failure
                    for attempt in range(2):
                        try:
                            computed = self._inner(batch)
                            break
                        except Exception as e:
                            if attempt == 0:
                                time.sleep(0.05 + random.random() * 0.1)
                                continue
                            try:
                                print(json.dumps({
                                    "where": "embeddings",
                                    "action": "batch_error",
                                    "error": str(e),
                                    "batch_index": batch_index,
                                    "skipped": len(batch)
                                }))
                            except Exception:
                                pass
                            computed = []
                            break
                    # Insert successful items into cache
                    for j, emb in enumerate(computed):
                        # Locate original index in 'texts'
                        orig_i = missing_indices[b_start + j]
                        k = keys[orig_i]
                        self._store[k] = emb
                        self._store.move_to_end(k)
                        if len(self._store) > self._max_size:
                            self._store.popitem(last=False)
                    elapsed_ms = int((time.time() - t0) * 1000)
                    try:
                        print(json.dumps({
                            "where": "embeddings",
                            "action": "batch",
                            "batch_size": len(batch),
                            "batch_index": batch_index + 1,
                            "total_batches": total_batches,
                            "elapsed_ms": elapsed_ms
                        }))
                    except Exception:
                        pass
                total_ms = int((time.time() - start) * 1000)
                avg_ms = int(total_ms / max(1, total))
                try:
                    print(json.dumps({
                        "where": "embeddings",
                        "action": "job",
                        "total_texts": total,
                        "total_ms": total_ms,
                        "avg_per_text_ms": avg_ms
                    }))
                except Exception:
                    pass
                self._misses += len(missing_indices)
            # Hit/move-to-end for all accessed keys
            for k in keys:
                if k in self._store:
                    self._store.move_to_end(k)
            self._hits += (len(input) - len(missing_indices))

            # Record last job stats for diagnostics
            try:
                _EMBED_LAST_JOB["batches"] = total_batches if missing_indices else 0
                _EMBED_LAST_JOB["batch_size"] = batch_size if missing_indices else 0
                _EMBED_LAST_JOB["cache_hit"] = len(missing_indices) == 0
            except Exception:
                pass

            # Structured log per call
            try:
                print(json.dumps({
                    "where": "embeddings",
                    "action": "cache",
                    "backend": backend,
                    "model": model,
                    "embed_cache_hit": self._hits,
                    "embed_cache_miss": self._misses,
                    "size": len(self._store),
                    "capacity": self._max_size
                }))
            except Exception:
                pass

            # Assemble outputs in original order
            return [self._store[k] for k in keys]

    wrapper = LruEmbeddingWrapper(base_fn, model, capacity)
    _EMBEDDING_WRAPPER_CACHE[cache_key] = wrapper
    return wrapper


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
        return "docs_ibc_v2"
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
    # If caller passed a falsey value, resolve default under appdata
    final_dir = persist_directory or get_default_chroma_dir()
    # Create the directory if it doesn't exist
    os.makedirs(final_dir, exist_ok=True)

    # Reuse client per directory
    if final_dir in _CHROMA_CLIENT_CACHE:
        status = {
            "where": "vector_store",
            "action": "init_or_reuse",
            "target": "client",
            "persist_directory": final_dir,
            "reused": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _LAST_VECTOR_STORE_STATUS = status
        print(json.dumps(status))
        return _CHROMA_CLIENT_CACHE[final_dir]

    client = chromadb.PersistentClient(final_dir)
    _CHROMA_CLIENT_CACHE[final_dir] = client
    status = {
        "where": "vector_store",
        "action": "init_or_reuse",
        "target": "client",
        "persist_directory": final_dir,
        "reused": False,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _LAST_VECTOR_STORE_STATUS = status
    print(json.dumps(status))
    return client


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
    
    # Cache and reuse collection object per (client id, name)
    cache_key = (id(client), resolved_name)
    if cache_key in _COLLECTION_CACHE:
        col_cached = _COLLECTION_CACHE[cache_key]
        status = {
            "where": "vector_store",
            "action": "init_or_reuse",
            "target": "collection",
            "collection_name": resolved_name,
            "reused": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _LAST_VECTOR_STORE_STATUS = status
        print(json.dumps(status))
        return col_cached

    # Try to get the collection, create it if it doesn't exist
    try:
        col = client.get_collection(
            name=resolved_name,
            embedding_function=embedding_func
        )
        _COLLECTION_CACHE[cache_key] = col
        status = {
            "where": "vector_store",
            "action": "init_or_reuse",
            "target": "collection",
            "collection_name": resolved_name,
            "reused": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _LAST_VECTOR_STORE_STATUS = status
        print(json.dumps(status))
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
        col = client.create_collection(
            name=resolved_name,
            embedding_function=embedding_func,
            metadata={
                "hnsw:space": distance_function,
                "embedding_backend": backend,
                "embedding_model": model,
            }
        )
        _COLLECTION_CACHE[cache_key] = col
        status = {
            "where": "vector_store",
            "action": "init_or_reuse",
            "target": "collection",
            "collection_name": resolved_name,
            "reused": False,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _LAST_VECTOR_STORE_STATUS = status
        print(json.dumps(status))
        return col


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
    batch_size: int = 200,
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
    # Prefilter limit: if the total exceeds a threshold, skip client-side scanning entirely
    try:
        prefilter_limit_raw = os.getenv("KEYWORD_PREFILTER_LIMIT", "5000")
        prefilter_limit = max(0, int(str(prefilter_limit_raw).strip()))
    except Exception:
        prefilter_limit = 5000
    if prefilter_limit > 0 and total > prefilter_limit:
        print(json.dumps({
            "where": "vector_store",
            "action": "keyword_prefilter",
            "prefilter_fallback": True,
            "total_candidates": total,
            "limit": prefilter_limit
        }))
        return {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "prefilter_fallback": True,
        }
    # To prevent unbounded scans on large collections, cap the number of rows scanned.
    # Configure via env KEYWORD_SCAN_LIMIT; default to 2000 documents.
    try:
        scan_limit_raw = os.getenv("KEYWORD_SCAN_LIMIT", "2000")
        scan_limit = max(0, int(str(scan_limit_raw).strip()))
    except Exception:
        scan_limit = 2000
    total_to_scan = min(total, scan_limit) if scan_limit > 0 else total
    results_docs: List[str] = []
    results_metas: List[Dict[str, Any]] = []
    results_ids: List[str] = []

    offset = 0
    while offset < total_to_scan and len(results_docs) < max_results:
        res = collection.get(include=["documents", "metadatas"], limit=min(batch_size, total_to_scan - offset), offset=offset)
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

    out = {
        "documents": [results_docs],
        "metadatas": [results_metas],
        "ids": [results_ids],
    }
    if total > total_to_scan:
        # Informative log: scan capped
        print(json.dumps({
            "where": "vector_store",
            "action": "keyword_scan",
            "capped": True,
            "scanned": total_to_scan,
            "total": total
        }))
    return out


# --------------- Health/diagnostics helpers ---------------
def get_process_uptime_seconds() -> int:
    try:
        return int(time.time() - _PROCESS_START_TS)
    except Exception:
        return 0


def get_memory_usage_mb() -> int:
    try:
        proc = psutil.Process(os.getpid())
        rss = getattr(proc, "memory_info")().rss
        return int(rss / (1024 * 1024))
    except Exception:
        return 0


def increment_query_count() -> None:
    global _QUERY_COUNT
    try:
        _QUERY_COUNT += 1
    except Exception:
        pass


def get_query_count() -> int:
    try:
        return int(_QUERY_COUNT)
    except Exception:
        return 0


def get_last_embedding_job_stats() -> Dict[str, Any]:
    # Shallow copy to avoid mutation by callers
    return {
        "batches": int(_EMBED_LAST_JOB.get("batches", 0) or 0),
        "batch_size": int(_EMBED_LAST_JOB.get("batch_size", 0) or 0),
        "cache_hit": bool(_EMBED_LAST_JOB.get("cache_hit", False)),
    }


def get_embedding_cache_stats() -> Dict[str, Any]:
    backend, model = resolve_embedding_backend_and_model()
    wrap_key = (backend, model)
    wrap = _EMBEDDING_WRAPPER_CACHE.get(wrap_key)
    if not wrap:
        return {
            "backend": backend,
            "model": model,
            "size": 0,
            "capacity": int(os.getenv("EMBED_CACHE_SIZE", "2048") or 2048),
            "hits": 0,
            "misses": 0,
            **get_last_embedding_job_stats(),
        }
    # Access wrapper internals safely
    size = len(getattr(wrap, "_store", {}))
    capacity = getattr(wrap, "_max_size", 0)
    hits = getattr(wrap, "_hits", 0)
    misses = getattr(wrap, "_misses", 0)
    return {
        "backend": backend,
        "model": model,
        "size": int(size),
        "capacity": int(capacity),
        "hits": int(hits),
        "misses": int(misses),
        **get_last_embedding_job_stats(),
    }


def build_section_search_terms(query: str) -> List[str]:
    """Generate helpful substrings for section/table lookups.

    Extract sequences like 1507.9.6 and also partials (1507.9, 1507) to improve recall.
    """
    # Normalize inequalities and whitespace
    q_raw = query or ""
    q_norm = normalize_inequalities(q_raw.strip())

    terms: List[str] = []
    terms.append(q_norm)

    # Common prefixes with normalized text
    terms.extend([
        q_norm.replace("section", "", 1).strip(),
        q_norm.replace("Section", "", 1).strip(),
        q_norm.replace("TABLE", "", 1).strip(),
        q_norm.replace("Table", "", 1).strip(),
    ])

    # Extract dotted numeric references (e.g., 1507.9.6)
    section_matches = re.findall(r"\b\d+(?:\.\d+)+\b", q_norm)
    for m in section_matches:
        parts = m.split(".")
        # Always include full section
        terms.append(m)
        # And progressively shorter prefixes
        for i in range(1, len(parts)):
            terms.append(".".join(parts[:i]))

    # Numeric expansion variants (mph, lb/kN, inequalities, etc.)
    num_expanded = expand_numeric_tokens(q_norm)
    terms.extend(num_expanded)

    # Pair unit synonyms within tokens
    terms = pair_unit_synonyms(terms)

    # Deduplicate while preserving order
    seen = set()
    unique_terms: List[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            unique_terms.append(t)
    return unique_terms
