import asyncio
import os
import dotenv
from tiered_agent import TieredRagAgent
from rag_agent import RAGDeps
from utils_vectorstore import get_vector_store
from utils import resolve_embedding_backend_and_model

# Load environment
dotenv.load_dotenv(override=True)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return result

async def test_question(agent, deps, question, label):
    print(f"\n[{label}] Testing: {question}\n")
    print("-" * 50)
    async for chunk in agent.run_pipeline(question, deps):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 50)

async def main():
    print("Initializing dependencies...")
    vector_store = get_vector_store(backend="bigquery")
    _, emb_model = resolve_embedding_backend_and_model()
    
    deps = RAGDeps(
        vector_store=vector_store,
        embedding_model=emb_model,
        collection_name="docs_ibc_v2",
        vector_backend="bigquery"
    )
    
    print("Creating TieredRagAgent...")
    agent = TieredRagAgent()
    
    # Test 1: Fallback (General Knowledge)
    await test_question(
        agent, deps, 
        "What is 7 + 7?", 
        "FALLBACK TEST"
    )
    
    # Test 2: Clear Refusal (Unanswerable/Nonsense but might be fallback if no RAG hits)
    await test_question(
        agent, deps, 
        "What is the airspeed velocity of an unladen swallow according to the IBC?", 
        "REFUSAL TEST"
    )

    # Test 3: Math (Should be handled by LLM now, not tool)
    await test_question(
        agent, deps,
        "Calculate the deflection limit for a 30ft span with L/180.",
        "MATH TEST (LLM)"
    )


    print("\nDone.")

if __name__ == "__main__":
    _run_async(main())
