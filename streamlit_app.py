import streamlit as st
import asyncio
import os
import re
from pathlib import Path
import shutil
import time

# Guarded import for model HTTP errors; fallback to generic Exception
try:
    from pydantic_ai.exceptions import ModelHTTPError  # type: ignore
except Exception:  # pragma: no cover - fallback for older/newer versions
    class ModelHTTPError(Exception):
        pass

# Resolve and load .env from writable AppData directory; create scaffold on first run
from utils import resolve_embedding_backend_and_model, get_env_file_path, ensure_appdata_scaffold, resolve_collection_name
from utils import get_process_uptime_seconds, get_memory_usage_mb, get_query_count, get_embedding_cache_stats
from utils import sanitize_and_validate_openai_key, get_key_diagnostics, compute_embedding_fingerprint
from utils import increment_query_count, _LAST_VECTOR_STORE_STATUS
from utils_vectorstore import get_vector_store, resolve_vector_backend

# Force sane defaults: prefer writable /tmp target unless explicitly overridden
try:
    pass
except Exception:
    pass
os.environ.setdefault("RAG_COLLECTION_NAME", "docs_ibc_v2")

ensure_appdata_scaffold()

# Do not load .env in container runtime; rely on App Hosting/Cloud env
if not (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    try:
        from dotenv import load_dotenv  # local dev only
        # Try repo .env first, then fall back to appdata .env
        repo_env = Path(__file__).resolve().parent / ".env"
        if repo_env.exists():
            load_dotenv(dotenv_path=repo_env, override=True)
        else:
            load_dotenv(dotenv_path=get_env_file_path(), override=True)
    except Exception:
        pass

from rag_agent import get_agent, RAGDeps, build_retrieval_context


# Cache a single vector store instance per process to avoid heavy reinitialization on rerenders
@st.cache_resource(show_spinner=False)
def _get_cached_vector_store():
    """Get or create cached vector store instance."""
    # Always force BigQuery backend
    return get_vector_store(backend="bigquery")

# Sanitize and validate key once at startup (lightweight)
sanitize_and_validate_openai_key()
# Defer computing embedding fingerprint (heavy imports) until diagnostics render
_fingerprint: dict = {}

# In container runtimes, fail fast if key invalid and prewarm embeddings before readiness
print("[startup] container mode detected?", bool(os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")))
if (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    kd = get_key_diagnostics()
    print(f"[startup] key diagnostics: {kd}")
    if not kd.get("valid"):
        print("[startup] OPENAI_API_KEY invalid -> exiting early")
        raise SystemExit(1)
    try:
        from utils import create_embedding_function
        print("[startup] warming embeddings...")
        fn = create_embedding_function()
        _ = fn(["warmup"])  # ensure model fully loads
        print("[startup] embedding warmup complete")
    except Exception:
        print("[startup] embedding warmup failed, exiting")
        raise SystemExit(1)

async def get_agent_deps(header_contains: str | None, source_contains: str | None):
    """Create agent dependencies with configured vector store."""
    # Get vector store backend information
    print("[deps] resolving vector backend")
    backend_name, backend_config = resolve_vector_backend()

    # Log backend information
    print(f"[ui] Using vector backend: '{backend_name}'")
    st.sidebar.caption(f"Vector backend: {backend_name}")

    if backend_name == "bigquery":
        st.sidebar.caption(f"Project: {backend_config.get('project', 'N/A')}")
        st.sidebar.caption(f"Dataset: {backend_config.get('dataset', 'N/A')}")

    # Display embeddings info
    emb_backend, emb_model = resolve_embedding_backend_and_model()
    st.sidebar.caption(f"Embeddings: {emb_backend} / {emb_model}")

    collection_name = "docs_ibc_v2"  # BigQuery uses table name instead

    print("[deps] instantiating vector store")
    deps = RAGDeps(
        vector_store=_get_cached_vector_store(),
        embedding_model=emb_model,
        collection_name=collection_name,
        vector_backend=backend_name,
        header_contains=(header_contains or None),
        source_contains=(source_contains or None),
    )
    print("[deps] vector store ready")
    return deps


def _vector_store_diag() -> dict:
    """Return diagnostics for the active vector store (BigQuery)."""
    try:
        vector_store = _get_cached_vector_store()
        
        # Get common info from vector store
        info = vector_store.get_info()
        doc_count = vector_store.count_documents()

        result = {
            "backend": "bigquery",
            "document_count": doc_count,
        }

        # BigQuery-specific diagnostics
        from pathlib import Path
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        creds_status = "not_set"
        if creds_path:
            creds_file = Path(creds_path)
            creds_status = "exists" if creds_file.exists() else "missing_file"

        result.update({
            "project": info.get("project", "N/A"),
            "dataset": info.get("dataset", "N/A"),
            "table": info.get("table", "N/A"),
            "table_size_bytes": info.get("table_size_bytes", 0),
            "indexes": info.get("indexes", []),
            "created": info.get("created", "N/A"),
            "modified": info.get("modified", "N/A"),
            "credentials_status": creds_status,
            "credentials_path": creds_path or "using_default_ADC",
        })

        return result

    except Exception as e:
        return {
            "backend": "unknown",
            "error": str(e),
            "document_count": 0,
        }


def _api_key_diag() -> dict:
    d = get_key_diagnostics()
    try:
        env_path = get_env_file_path()
    except Exception:
        env_path = ""
    return {
        "env_file": env_path,
        "present": bool(d.get("present")),
        "masked_prefix": d.get("masked_prefix"),
        "valid": bool(d.get("valid")),
        "error": d.get("error"),
    }


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        content = part.content
        if "User question:" in content:
            content = content.split("User question:", 1)[1].strip()
        with st.chat_message("user"):
            st.markdown(content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)
    elif part.part_kind == 'tool-return':
        # Enhance display if metadata is present in context (non-breaking; retrieve format unchanged)
        payload = getattr(part, 'content', None)
        if isinstance(payload, dict):
            # Best-effort display of title and section_path if present
            title = payload.get('title')
            section_path = payload.get('section_path')
            source_url = payload.get('source_url')
            if title or section_path or source_url:
                with st.chat_message("assistant"):
                    if title:
                        st.markdown(f"**{title}**")
                    if section_path:
                        st.caption(section_path)
                    if source_url and source_url.startswith('http'):
                        st.markdown(f"[Source]({source_url})")
                    elif source_url and source_url.startswith('file://'):
                        st.caption(source_url.replace('file://', ''))

async def run_agent_with_streaming(user_input):
    try:
        deps: RAGDeps = st.session_state.agent_deps
        context_text, structured_entries = build_retrieval_context(
            user_input,
            deps,
            n_results=5,
            header_contains=getattr(deps, "header_contains", None),
            source_contains=getattr(deps, "source_contains", None),
        )

        if not context_text.strip():
            fallback_prompt = (
                "You do not have any retrieved reference material to cite. Respond to the user's question "
                "using your general building-code expertise. If the question requires specific code references "
                "that you cannot confirm, acknowledge that while still providing helpful guidance.\n\n"
                f"User question: {user_input}"
            )
            async with get_agent().run_stream(
                fallback_prompt,
                deps=deps,
                message_history=st.session_state.agent_messages,
                model_settings={"temperature": 0.4},
            ) as result:
                async for message in result.stream_text(delta=True):
                    yield message
            st.session_state.last_agent_new_messages = result.new_messages()
            return

        augmented_question = (
            "Answer the user using ONLY the context provided. Cite sections in plain text "
            "(e.g., \"IBC 2018 §1507.2\") when possible. If the context does not contain the answer, "
            "reply with \"I don't have that information in the current documentation.\" Use a natural tone.\n\n"
            f"Context:\n{context_text}\n\nUser question: {user_input}"
        )
        async with get_agent().run_stream(
            augmented_question,
            deps=deps,
            message_history=st.session_state.agent_messages,
            model_settings={"temperature": 0.2},
        ) as result:
            async for message in result.stream_text(delta=True):
                yield message
        st.session_state.last_agent_new_messages = result.new_messages()
    except ModelHTTPError as e:
        # Re-raise with a sentinel attribute so UI can display a friendly message
        setattr(e, "_is_auth_error", getattr(e, "status_code", None) == 401 or "status_code: 401" in str(e))
        raise


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    print("[main] entered")
    st.title("CAL AI Agent")

    # Lightweight health probe: return quickly when ?healthz=1
    try:
        params = st.query_params
        healthz = params.get("healthz") if params else None
        if healthz == "1" or (isinstance(healthz, list) and "1" in healthz):
            st.write("ok")
            return
    except Exception:
        pass

    # Gate readiness: fail fast in UI if key invalid; warmup embeddings
    key_diag = get_key_diagnostics()
    print(f"[main] key diagnostics: {key_diag}")
    if not key_diag.get("valid"):
        st.sidebar.error("OPENAI_API_KEY invalid or missing. See Diagnostics.")
        with st.sidebar.expander("Diagnostics", expanded=True):
            st.code(str({"api_key": _api_key_diag()}))
        st.stop()

    # Dependency check: Tesseract
    import shutil
    if not shutil.which("tesseract"):
        st.sidebar.error("⚠️ 'tesseract' dependency is missing! PDF OCR features will fail.")
        with st.sidebar.expander("Installation Help", expanded=True):
            st.markdown("""
            **Tesseract OCR is required.**
            - **Mac:** `brew install tesseract`
            - **Windows:** [Install Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
            - **Linux:** `sudo apt install tesseract-ocr`
            """)

    # Prewarm embeddings on first render (graceful failure - app can continue)
    try:
        # Touch embedding function to ensure model is loaded
        from utils import create_embedding_function
        print("[main] creating embedding function")
        fn = create_embedding_function()
        _ = fn(["warmup"])
        print("[main] embedding warmup finished")
    except Exception as e:
        # Log warning but don't stop - embeddings will be loaded on first query
        st.sidebar.warning("⚠️ Embedding warmup skipped. Models will load on first query.")
        print(f"[embeddings] Warmup failed (non-fatal): {e}")

    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    # Sidebar controls: collection display + optional filters
    # Preserve last values via session_state keys
    if "header_contains" not in st.session_state:
        st.session_state.header_contains = ""
    if "source_contains" not in st.session_state:
        st.session_state.source_contains = ""

    # Filters (collection is fixed to docs_ibc_v2)
    st.sidebar.markdown("### Retrieval Filters")
    st.sidebar.text_input("Header contains", key="header_contains", placeholder="e.g., Section 1507")
    st.sidebar.text_input("Source contains", key="source_contains", placeholder="e.g., pydantic.dev")

    # Recreate deps each render so filters are applied
    print("[main] building agent deps")
    st.session_state.agent_deps = await get_agent_deps(
        st.session_state.header_contains.strip() or None,
        st.session_state.source_contains.strip() or None,
    )
    print("[main] agent deps ready")

    # Show active collection and filters summary
    st.sidebar.markdown(f"**Collection:** {st.session_state.agent_deps.collection_name}")
    if st.session_state.header_contains or st.session_state.source_contains:
        st.sidebar.caption(
            f"Filters: header='{st.session_state.header_contains or ''}', source='{st.session_state.source_contains or ''}'"
        )

    # Display all messages from the conversation so far
    for item in st.session_state.chat_history:
        role = item.get("role", "assistant")
        content = item.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""

            # Properly consume the async generator with async for
            generator = run_agent_with_streaming(user_input)
            try:
                async for message in generator:
                    full_response += message
                    message_placeholder.markdown(full_response + "▌")
                # Final response without the cursor
                message_placeholder.markdown(full_response)
            except ModelHTTPError as e:
                diag = _api_key_diag()
                if getattr(e, "_is_auth_error", False):
                    message_placeholder.markdown(
                        "Authentication failed with the model provider (401). "
                        "Please update your OPENAI_API_KEY and restart the app."
                    )
                    with st.sidebar:
                        st.error("Invalid OPENAI_API_KEY (401).")
                        st.code(str({
                            "env_file": diag.get("env_file"),
                            "present": diag.get("present"),
                            "length": diag.get("length"),
                            "looks_prefixed": diag.get("looks_prefixed"),
                            "model": os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
                        }))
                else:
                    message_placeholder.markdown("An error occurred while contacting the model.")
                    st.warning(str(e))
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        new_agent_messages = st.session_state.pop("last_agent_new_messages", [])
        if new_agent_messages:
            st.session_state.agent_messages.extend(new_agent_messages)
        # Count served query
        increment_query_count()

    # Diagnostics panel
    with st.sidebar.expander("Diagnostics", expanded=False):
        # Quick health surface
        try:
            st.caption("Health")
            st.code(str({
                "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
            }))
        except Exception:
            pass
        st.caption("Initialization & cache status")
        vs_status = _LAST_VECTOR_STORE_STATUS or {}
        emb_stats = get_embedding_cache_stats()
        backend, model = resolve_embedding_backend_and_model()
        try:
            fp = compute_embedding_fingerprint()
        except Exception:
            fp = _fingerprint or {}
        # Get vector store diagnostics (backend-aware)
        vs_diag = _vector_store_diag()

        st.code(str({
            "uptime_sec": get_process_uptime_seconds(),
            "memory_mb": get_memory_usage_mb(),
            "queries_served": get_query_count(),
            "vector_store_init": {
                "reused": vs_status.get("reused"),
                "ts": vs_status.get("ts"),
                "target": vs_status.get("target"),
                "collection": vs_status.get("collection_name"),
            },
            "vector_store_backend": vs_diag,
            "embedding_cache": emb_stats,
            "api_key": _api_key_diag(),
            "model_choice": os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            "embedding_backend": backend,
            "embedding_model": model,
            "embedding_fingerprint": fp,
            "model_home": (os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")),
        }))


if __name__ == "__main__":
    asyncio.run(main())
