import os
import logging
import json
from typing import AsyncGenerator
from pydantic_ai import Agent
from rag_agent import RAGDeps, build_retrieval_context, _build_model, _resolve_model_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EngineeringAgent")

class TieredRagAgent:
    """
    A friendly, conversational Structural Engineer bot.
    Workflow:
    1. Retrieve context (silently).
    2. Generate response using a Master Prompt that enforces the 'Friendly Engineer' persona.
    3. Stream response directly to user.
    """

    def __init__(self):
        model_name = _resolve_model_name()
        model = _build_model(model_name)
        logger.info(f"TieredRagAgent using model: {model_name} (type={type(model).__name__})")

        self.system_prompt = (
            "You are an expert assistant across all topics.\n\n"
            "When retrieval context is provided, treat it as your primary source of truth and answer using it first. "
            "If context is partial or missing, still provide the best helpful answer you can and clearly label what is "
            "from context versus general knowledge. Keep responses clear, practical, and conversational."
        )

        self.agent = Agent(model, deps_type=RAGDeps, system_prompt=self.system_prompt)

    async def run_pipeline(self, user_question: str, deps: RAGDeps) -> AsyncGenerator[str, None]:
        """
        Executes the simplified pipeline: Retrieval -> Inference -> Stream.
        """
        logger.info(f"Pipeline started for query: {user_question[:50]}...")
        logger.info(
            "Pipeline deps: id=%s collection=%s vector_table=%s",
            id(deps),
            getattr(deps, "collection_name", ""),
            getattr(getattr(deps, "vector_store", None), "table", ""),
        )

        # --- STEP 1: SILENT RETRIEVAL ---
        context_str = ""
        try:
            # We do NOT yield any status to the UI. It runs in background.
            context_text, structured_entries, retrieval_debug = build_retrieval_context(
                user_question,
                deps,
                n_results=5,
                header_contains=getattr(deps, "header_contains", None),
                source_contains=getattr(deps, "source_contains", None),
                return_debug=True,
            )
            logger.info(
                "Raw retrieval debug: docs=%s doc_lengths=%s meta_keys=%s",
                retrieval_debug.get("raw_docs_count", 0),
                retrieval_debug.get("raw_doc_lengths", []),
                retrieval_debug.get("raw_meta_keys", []),
            )

            if structured_entries:
                # Format context for the LLM
                formatted_entries = []
                for entry in structured_entries:
                    formatted_entries.append(
                        f"Source: {entry.get('code_label', 'Doc')} {entry.get('section', '')}\n"
                        f"Content: {entry.get('quote', '')}"
                    )
                context_str = "\n\n".join(formatted_entries)
                logger.info(f"Retrieval success: found {len(structured_entries)} chunks.")
            elif context_text.strip():
                context_str = context_text
                logger.info("Retrieval had no structured entries; falling back to raw context.")
            else:
                logger.info("Retrieval returned no chunks.")

        except Exception as e:
            logger.error(f"Retrieval failed (non-blocking): {e}")
            # We proceed without context if retrieval fails
            context_str = ""

        # --- STEP 2: INFERENCE ---
        # Construct the full prompt
        if context_str:
            user_prompt = (
                f"User Question: {user_question}\n\n"
                f"Retrieved Context:\n{context_str}\n\n"
                f"Please answer the user's question using the context above if relevant."
            )
        else:
            user_prompt = user_question
        context_preview = context_str[:200].replace("\n", " ")
        logger.info(
            "Agent prompt context preview (first 200 chars): %r",
            context_preview,
        )

        try:
            # Run the agent with specific system prompt and user content
            # Note: We create a temporary run instance or use the agent to stream
            # pydantic_ai agent.run_stream usage:
            async with self.agent.run_stream(user_prompt, deps=deps) as result:
                async for chunk in result.stream():
                    yield chunk
                    
            logger.info("Pipeline complete.")

        except Exception as e:
            logger.error(f"Inference Error: {e}")
            yield f"**Error**: {str(e)}"
