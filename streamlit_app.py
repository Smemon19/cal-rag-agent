import streamlit as st
import asyncio
import os
import json
import re
from pathlib import Path
import shutil
import time
import bcrypt
from google.cloud import secretmanager

# Guarded import for model HTTP errors; fallback to generic Exception
try:
    from pydantic_ai.exceptions import ModelHTTPError  # type: ignore
except Exception:  # pragma: no cover - fallback for older/newer versions
    class ModelHTTPError(Exception):
        pass

# Resolve and load .env from writable AppData directory; create scaffold on first run
from utils import resolve_embedding_backend_and_model, get_env_file_path, ensure_appdata_scaffold, resolve_collection_name
from utils import get_process_uptime_seconds, get_memory_usage_mb, get_query_count, get_embedding_cache_stats
from utils import compute_embedding_fingerprint
from utils import increment_query_count, _LAST_VECTOR_STORE_STATUS
from utils_vectorstore import get_vector_store, resolve_vector_backend

# Force sane defaults: prefer writable /tmp target unless explicitly overridden
try:
    pass
except Exception:
    pass
os.environ.setdefault("RAG_COLLECTION_NAME", "default")

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
from policy_badger_pro import load_policy_pro_context, ask_policy_pro

CHATBOT_CONFIG = {
    "policy_badger": {
        "label": "Policy Badger",
        "description": "Ask questions about company policies",
        "collection": "policy_badger",
    },
    "test_badger": {
        "label": "Test Badger",
        "description": "Test chatbot",
        "collection": "documents",
    },
    "policy_badger_pro": {
        "label": "Policy Badger Pro",
        "description": "In-memory policy retrieval with focused document excerpts",
        "collection": "policy_badger_pro",
    },
}


# Cache a single vector store instance per process to avoid heavy reinitialization on rerenders
@st.cache_resource(show_spinner=False)
def _get_cached_vector_store(table_name: str):
    """Get or create cached vector store instance."""
    # Always force BigQuery backend for chatbot-specific table.
    print(f"[vectorstore] cache lookup table={table_name}")
    return get_vector_store(backend="bigquery", table=table_name)

_fingerprint: dict = {}
_RUNTIME_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run a coroutine on a persistent process event loop."""
    global _RUNTIME_LOOP
    if _RUNTIME_LOOP is None or _RUNTIME_LOOP.is_closed():
        _RUNTIME_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_RUNTIME_LOOP)
    return _RUNTIME_LOOP.run_until_complete(coro)

# In container runtimes, prewarm embeddings before readiness
_is_container = bool(os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING"))
_startup_warmup_enabled = os.getenv("ENABLE_STARTUP_EMBED_WARMUP", "0").strip() == "1"
print("[startup] container mode detected?", _is_container)
if _is_container and _startup_warmup_enabled:
    try:
        from utils import create_embedding_function
        print("[startup] warming embeddings...")
        fn = create_embedding_function()
        _ = fn(["warmup"])
        print("[startup] embedding warmup complete")
    except Exception as e:
        print(f"[startup] embedding warmup failed: {e}")
else:
    print("[startup] embedding warmup skipped")


def _get_auth_project_id() -> str:
    return (
        os.getenv("AUTH_PROJECT_ID")
        or os.getenv("VERTEX_PROJECT_ID")
        or os.getenv("BQ_PROJECT")
        or "badgers-487618"
    )


def _get_auth_secret_name() -> str:
    return os.getenv("AUTH_USERS_SECRET", "cal-rag-users")


@st.cache_data(show_spinner=False, ttl=60)
def _load_users_from_secret() -> dict:
    """Load username->bcrypt hash mapping from Secret Manager."""
    project_id = _get_auth_project_id()
    secret_name = _get_auth_secret_name()
    client = secretmanager.SecretManagerServiceClient()
    version_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": version_path})
        payload = response.payload.data.decode("utf-8")
        data = json.loads(payload) if payload.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[auth] failed to load users from secret manager: {e}")
        return {}


def _verify_login(username: str, password: str) -> bool:
    users = _load_users_from_secret()
    stored_hash = users.get(username)
    if not stored_hash or not isinstance(stored_hash, str):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


def _render_login_gate() -> bool:
    """Render login form and block app access until authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = ""

    if st.session_state.authenticated:
        return True

    st.title("CAL AI Agent")
    st.subheader("Sign in")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if _verify_login(username.strip(), password):
            st.session_state.authenticated = True
            st.session_state.auth_user = username.strip()
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()
    return False


def _clear_chat_state() -> None:
    st.session_state.chat_history = []
    st.session_state.agent_messages = []
    st.session_state.header_contains = ""
    st.session_state.source_contains = ""


def _switch_bot(bot_key: str) -> None:
    current = st.session_state.get("current_page", "home")
    if current != bot_key:
        _clear_chat_state()
    st.session_state.current_page = bot_key
    st.session_state.selected_bot = bot_key
    st.rerun()


def _render_home_page() -> None:
    st.title("CAL AI Agent")
    st.subheader(f"Welcome, {st.session_state.get('auth_user', '')}")
    st.write("Choose a chatbot to continue:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Policy Badger")
        st.caption(CHATBOT_CONFIG["policy_badger"]["description"])
        if st.button("Open Policy Badger", use_container_width=True):
            _switch_bot("policy_badger")
    with col2:
        st.markdown("### Test Badger")
        st.caption(CHATBOT_CONFIG["test_badger"]["description"])
        if st.button("Open Test Badger", use_container_width=True):
            _switch_bot("test_badger")
    with col3:
        st.markdown("### Policy Badger Pro")
        st.caption(CHATBOT_CONFIG["policy_badger_pro"]["description"])
        if st.button("Open Policy Badger Pro", use_container_width=True):
            _switch_bot("policy_badger_pro")

    st.sidebar.caption(f"Signed in as: {st.session_state.get('auth_user', '')}")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_user = ""
        st.session_state.current_page = "home"
        st.session_state.selected_bot = ""
        _clear_chat_state()
        st.rerun()

async def get_agent_deps(header_contains: str | None, source_contains: str | None):
    """Create agent dependencies with configured vector store."""
    bot_key = st.session_state.get("current_page", "test_badger")
    bot = CHATBOT_CONFIG.get(bot_key, CHATBOT_CONFIG["test_badger"])
    table_name = bot["collection"]
    st.session_state.active_table = table_name
    print(
        f"[deps] bot={bot_key} table={table_name} "
        f"(explicit table routing; not env-dependent)"
    )

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

    collection_name = table_name  # BigQuery uses table name instead

    print("[deps] instantiating vector store")
    deps = RAGDeps(
        vector_store=_get_cached_vector_store(table_name),
        embedding_model=emb_model,
        collection_name=collection_name,
        vector_backend=backend_name,
        header_contains=(header_contains or None),
        source_contains=(source_contains or None),
    )
    print("[deps] vector store ready")
    return deps


def _vector_store_diag(table_name: str) -> dict:
    """Return diagnostics for the active vector store (BigQuery)."""
    try:
        vector_store = _get_cached_vector_store(table_name)
        
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


from tiered_agent import TieredRagAgent

async def run_agent_with_streaming(user_input):
    try:
        deps: RAGDeps = st.session_state.agent_deps
        print(
            "[stream] using deps "
            f"id={id(deps)} collection={getattr(deps, 'collection_name', '')} "
            f"vector_table={getattr(getattr(deps, 'vector_store', None), 'table', '')}"
        )
        
        # Instantiate the tiered agent (could be cached, but lightweight enough to init here)
        tiered_agent = TieredRagAgent()
        
        # Run the full pipeline
        # The pipeline handles retrieval, reasoning, and polishing internally
        # and yields progress updates and final text chunks.
        async for message in tiered_agent.run_pipeline(user_input, deps):
            yield message
            
    except ModelHTTPError as e:
        # Re-raise with a sentinel attribute so UI can display a friendly message
        setattr(e, "_is_auth_error", getattr(e, "status_code", None) == 401 or "status_code: 401" in str(e))
        raise


async def _stream_response(user_input: str, message_placeholder):
    """Stream model output and update placeholder progressively."""
    full_response = ""
    generator = run_agent_with_streaming(user_input)
    try:
        async for message in generator:
            chunk = _normalize_stream_message(message)
            if not chunk:
                continue
            # Handle both cumulative snapshots and token deltas.
            if chunk.startswith(full_response):
                full_response = chunk
            else:
                full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        return full_response
    except ModelHTTPError as e:
        if getattr(e, "_is_auth_error", False):
            full_response = (
                "Authentication failed with Vertex AI (401). "
                "Check your service account permissions and ADC setup."
            )
            message_placeholder.markdown(full_response)
        else:
            full_response = "An error occurred while contacting the model."
            message_placeholder.markdown(full_response)
            st.warning(str(e))
        return full_response


def _normalize_stream_message(message) -> str:
    """Normalize stream chunks across providers and SDK versions."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        for key in ("text", "content", "delta"):
            value = message.get(key)
            if isinstance(value, str):
                return value
    for attr in ("text", "content", "delta"):
        value = getattr(message, attr, None)
        if isinstance(value, str):
            return value
    return str(message)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    print("[main] entered")

    # Lightweight health probe: return quickly when ?healthz=1
    try:
        params = st.query_params
        healthz = params.get("healthz") if params else None
        if healthz == "1" or (isinstance(healthz, list) and "1" in healthz):
            st.write("ok")
            return
    except Exception:
        pass

    if not _render_login_gate():
        return

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "selected_bot" not in st.session_state:
        st.session_state.selected_bot = ""

    current_page = st.session_state.get("current_page", "home")
    if current_page == "home":
        _render_home_page()
        return

    active_bot = CHATBOT_CONFIG.get(current_page, CHATBOT_CONFIG["test_badger"])
    active_collection = active_bot["collection"]

    st.title("CAL AI Agent")
    st.subheader(active_bot["label"])
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.session_state.selected_bot = ""
        st.rerun()

    if current_page == "policy_badger_pro":
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "policy_pro_last_debug" not in st.session_state:
            st.session_state.policy_pro_last_debug = {}

        st.sidebar.caption(f"Signed in as: {st.session_state.get('auth_user', '')}")
        if st.sidebar.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_user = ""
            st.session_state.current_page = "home"
            st.session_state.selected_bot = ""
            _clear_chat_state()
            st.rerun()

        context = None
        context_debug = {}
        try:
            context = load_policy_pro_context()
            context_debug = (context or {}).get("stats", {})
        except Exception as e:
            st.sidebar.warning("Policy index failed to load.")
            st.sidebar.caption(str(e))

        last_debug = st.session_state.get("policy_pro_last_debug", {})
        status_documents = int(context_debug.get("documents_loaded", 0))
        status_chunks = int(context_debug.get("chunks_indexed", 0))
        status_used = int(last_debug.get("context_chunks_used", 0))
        with st.sidebar.expander("Policy Pro Status", expanded=True):
            st.caption(f"Documents Loaded: {status_documents}")
            st.caption(f"Chunks Indexed: {status_chunks}")
            st.caption(f"Context Chunks Used: {status_used}")

        for item in st.session_state.chat_history:
            role = item.get("role", "assistant")
            content = item.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

        user_input = st.chat_input("What do you want to know?")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                st.info("Retrieving the most relevant policy excerpts...")
                try:
                    if not context:
                        raise RuntimeError("Policy context is unavailable.")
                    result = ask_policy_pro(user_input, context)
                    answer_text = str(result.get("answer_text", "")).strip()
                    sources = result.get("sources", [])
                    debug = result.get("debug", {})
                    st.session_state.policy_pro_last_debug = debug

                    source_lines = "\n".join(f"- {src}" for src in sources) if sources else "- None"
                    full_response = (
                        f"{answer_text}\n\n"
                        "Sources:\n"
                        f"{source_lines}"
                    ).strip()
                except Exception as e:
                    full_response = "An error occurred while searching policy documents."
                    st.warning(str(e))
                st.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            increment_query_count()
        return

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

    # Filters for selected chatbot collection
    st.sidebar.markdown("### Retrieval Filters")
    st.sidebar.text_input("Header contains", key="header_contains", placeholder="e.g., Section 1507")
    st.sidebar.text_input("Source contains", key="source_contains", placeholder="e.g., pydantic.dev")

    # Recreate deps each render so filters are applied
    print("[main] building agent deps")
    st.session_state.agent_deps = _run_async(
        get_agent_deps(
            st.session_state.header_contains.strip() or None,
            st.session_state.source_contains.strip() or None,
        )
    )
    print("[main] agent deps ready")

    # Show active collection and filters summary
    st.sidebar.markdown(f"**Collection:** {st.session_state.agent_deps.collection_name}")
    st.sidebar.caption(f"Signed in as: {st.session_state.get('auth_user', '')}")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_user = ""
        st.session_state.current_page = "home"
        st.session_state.selected_bot = ""
        _clear_chat_state()
        st.rerun()
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
        print(
            f"[query] bot={current_page} table={active_collection} "
            f"(explicit table routing via deps/vector_store)"
        )
        st.sidebar.caption(f"Debug table: {active_collection}")
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            st.info(f"Searching table: {active_collection}")
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = _run_async(_stream_response(user_input, message_placeholder))
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        new_agent_messages = st.session_state.pop("last_agent_new_messages", [])
        if new_agent_messages:
            st.session_state.agent_messages.extend(new_agent_messages)
        # Count served query
        increment_query_count()

    # Diagnostics panel
    with st.sidebar.expander("Diagnostics", expanded=False):
        try:
            backend, model = resolve_embedding_backend_and_model()
        except Exception:
            backend, model = "unknown", "unknown"

        cal_model = os.getenv("CAL_MODEL_NAME") or os.getenv("MODEL_CHOICE") or "gemini-1.5-pro"
        st.write(f"**LLM Model:** `{cal_model}`")
        st.write(f"**Embeddings:** `{backend}` / `{model}`")

        # Show Vertex-specific or OpenAI-specific info
        if backend == "vertex":
            st.write(f"**Vertex Project:** `{os.getenv('VERTEX_PROJECT_ID', 'auto')}`")
            st.write(f"**Vertex Location:** `{os.getenv('VERTEX_LOCATION', 'us-central1')}`")

        st.write("---")
        st.caption("Initialization & cache status")
        vs_status = _LAST_VECTOR_STORE_STATUS or {}
        emb_stats = get_embedding_cache_stats()
        try:
            fp = compute_embedding_fingerprint()
        except Exception:
            fp = _fingerprint or {}
        vs_diag = _vector_store_diag(active_collection)

        st.code(str({
            "uptime_sec": get_process_uptime_seconds(),
            "memory_mb": get_memory_usage_mb(),
            "queries_served": get_query_count(),
            "llm_model": cal_model,
            "vector_store_backend": vs_diag,
            "embedding_cache": emb_stats,
            "embedding_backend": backend,
            "embedding_model": model,
            "embedding_fingerprint": fp,
        }))


if __name__ == "__main__":
    main()
