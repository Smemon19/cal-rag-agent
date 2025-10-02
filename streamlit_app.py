import streamlit as st
import asyncio
import os
from pathlib import Path

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai.exceptions import ModelHTTPError

# Resolve and load .env from writable AppData directory; create scaffold on first run
from utils import get_chroma_client, resolve_embedding_backend_and_model, get_env_file_path, ensure_appdata_scaffold, get_default_chroma_dir, resolve_collection_name
from utils import _LAST_VECTOR_STORE_STATUS
from utils import get_process_uptime_seconds, get_memory_usage_mb, get_query_count, get_embedding_cache_stats
from utils import sanitize_and_validate_openai_key, get_key_diagnostics, compute_embedding_fingerprint
from utils import increment_query_count

ensure_appdata_scaffold()
# Do not load .env in container runtime; rely on App Hosting/Cloud env
if not (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    try:
        from dotenv import load_dotenv  # local dev only
        load_dotenv(dotenv_path=get_env_file_path(), override=False)
    except Exception:
        pass

from rag_agent import get_agent, RAGDeps


# Cache a single Chroma client per process to avoid heavy reinitialization on rerenders
@st.cache_resource(show_spinner=False)
def _get_cached_chroma_client():
    return get_chroma_client(get_default_chroma_dir())

# Sanitize and validate key once at startup
sanitize_and_validate_openai_key()
_fingerprint = compute_embedding_fingerprint()

# In container runtimes, fail fast if key invalid and prewarm embeddings before readiness
if (os.getenv("K_SERVICE") or os.getenv("GOOGLE_CLOUD_RUN") or os.getenv("FIREBASE_APP_HOSTING")):
    kd = get_key_diagnostics()
    if not kd.get("valid"):
        raise SystemExit(1)
    try:
        from utils import create_embedding_function
        fn = create_embedding_function()
        _ = fn(["warmup"])  # ensure model fully loads
    except Exception:
        raise SystemExit(1)

async def get_agent_deps(header_contains: str | None, source_contains: str | None):
    resolved_collection = resolve_collection_name(None)
    # Log once on startup via Streamlit status text and server log
    print(f"[ui] Using ChromaDB collection: '{resolved_collection}'")
    st.sidebar.caption(f"Active collection: {resolved_collection}")
    backend, model = resolve_embedding_backend_and_model()
    # Also display embeddings info once
    st.sidebar.caption(f"Embeddings: {backend} / {model}")
    return RAGDeps(
        chroma_client=_get_cached_chroma_client(),
        collection_name=resolved_collection,
        embedding_model="all-MiniLM-L6-v2",
        header_contains=(header_contains or None),
        source_contains=(source_contains or None),
    )


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
        with st.chat_message("user"):
            st.markdown(part.content)
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
        async with get_agent().run_stream(
            user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
        ) as result:
            async for message in result.stream_text(delta=True):
                yield message
        # Add the new messages to the chat history (including tool calls and responses)
        st.session_state.messages.extend(result.new_messages())
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
    st.title("CAL AI Agent")

    # Gate readiness: fail fast in UI if key invalid; warmup embeddings
    key_diag = get_key_diagnostics()
    if not key_diag.get("valid"):
        st.sidebar.error("OPENAI_API_KEY invalid or missing. See Diagnostics.")
        with st.sidebar.expander("Diagnostics", expanded=True):
            st.code(str({"api_key": _api_key_diag()}))
        st.stop()
    # Prewarm embeddings on first render
    try:
        # Touch embedding function to ensure model is loaded
        from utils import create_embedding_function
        fn = create_embedding_function()
        _ = fn(["warmup"])
    except Exception as e:
        st.sidebar.error("Embedding warmup failed; see logs.")
        st.stop()

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
    st.session_state.agent_deps = await get_agent_deps(
        st.session_state.header_contains.strip() or None,
        st.session_state.source_contains.strip() or None,
    )

    # Show active collection and filters summary
    st.sidebar.markdown(f"**Collection:** {st.session_state.agent_deps.collection_name}")
    if st.session_state.header_contains or st.session_state.source_contains:
        st.sidebar.caption(
            f"Filters: header='{st.session_state.header_contains or ''}', source='{st.session_state.source_contains or ''}'"
        )

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

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
                            "model": os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
                        }))
                else:
                    message_placeholder.markdown("An error occurred while contacting the model.")
                    st.warning(str(e))
        # Count served query
        increment_query_count()

    # Diagnostics panel
    with st.sidebar.expander("Diagnostics", expanded=False):
        st.caption("Initialization & cache status")
        vs_status = _LAST_VECTOR_STORE_STATUS or {}
        emb_stats = get_embedding_cache_stats()
        backend, model = resolve_embedding_backend_and_model()
        fp = _fingerprint
        st.code(str({
            "uptime_sec": get_process_uptime_seconds(),
            "memory_mb": get_memory_usage_mb(),
            "queries_served": get_query_count(),
            "vector_store": {
                "reused": vs_status.get("reused"),
                "ts": vs_status.get("ts"),
                "target": vs_status.get("target"),
                "collection": vs_status.get("collection_name"),
            },
            "embedding_cache": emb_stats,
            "api_key": _api_key_diag(),
            "model_choice": os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
            "embedding_backend": backend,
            "embedding_model": model,
            "embedding_fingerprint": fp,
            "model_home": (os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")),
        }))


if __name__ == "__main__":
    asyncio.run(main())
