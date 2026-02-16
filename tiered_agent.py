import os
import logging
import json
from typing import AsyncGenerator
from pydantic_ai import Agent
from rag_agent import RAGDeps, build_retrieval_context

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
        model_name = os.getenv("CAL_MODEL_NAME", "gpt-4o") # Default to gpt-4o if not set
        
        # Master System Prompt
        self.system_prompt = (
            "You are a friendly, helpful Structural Engineer AI assistant. "
            "You specialize in structural calculations, building codes (ASCE 7, ACI 318, AISC SCM, IBC, etc.), "
            "and solving engineering problems.\n\n"
            "Your responses should be:\n"
            "1. **Detailed and Proper**: Explain your steps, cite codes where applicable, and show your work.\n"
            "2. **Conversational and Friendly**: Be helpful and approachable. Engange with the user naturaly.\n"
            "3. **Technically Accurate**: Prioritize correctness in your calculations and code interpretations.\n\n"
            "If retrieval context is provided, use it to ground your answer. "
            "If no context is found, rely on your internal expert knowledge to help the user."
        )
        
        # Initialize Agent with system_prompt
        self.agent = Agent(model_name, deps_type=RAGDeps, system_prompt=self.system_prompt)

    async def run_pipeline(self, user_question: str, deps: RAGDeps) -> AsyncGenerator[str, None]:
        """
        Executes the simplified pipeline: Retrieval -> Inference -> Stream.
        """
        logger.info(f"Pipeline started for query: {user_question[:50]}...")

        # --- STEP 1: SILENT RETRIEVAL ---
        context_str = ""
        try:
            # We do NOT yield any status to the UI. It runs in background.
            context_text, structured_entries = build_retrieval_context(
                user_question,
                deps,
                n_results=5,
                header_contains=getattr(deps, "header_contains", None),
                source_contains=getattr(deps, "source_contains", None),
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
