import asyncio
import os
from tiered_agent import TieredRagAgent
from unittest.mock import MagicMock
import rag_agent

# Mock retrieval to avoid API calls
rag_agent.build_retrieval_context = MagicMock(return_value=("Mock Context", [{"quote": "Info", "code_label": "TestDoc"}]))


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return result

# Simple mock for RAGDeps
class MockDeps:
    class MockVectorStore:
        def vector_search(self, *args, **kwargs): return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        def keyword_search(self, *args, **kwargs): return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        def get_info(self): return {}
    
    def __init__(self):
        self.vector_store = self.MockVectorStore()
        self.supabase = None
        self.openai = None

async def main():
    print("Initializing Agent...")
    try:
        agent = TieredRagAgent()
        print("Agent Initialized.")
        
        # Test 1: Simple run
        print("\n--- Test Run ---")
        deps = MockDeps()
        # We need to monkeypatch build_retrieval_context if we don't want to run real retrieval
        # But wait, tiered_agent imports it from rag_agent. 
        # For this simple test, we can let it fail gracefully (it has try/except) or patch it.
        # Let's rely on the try/except in tiered_agent.py
        
        async for chunk in agent.run_pipeline("Hello, who are you?", deps):
            print(f"|{chunk}|", end="\n", flush=True)
        print("\n--- End Test ---")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    _run_async(main())
