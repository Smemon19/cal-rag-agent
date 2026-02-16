import asyncio
import os
from pydantic_ai import Agent

# Simple agent without RAG deps, just to test stream behavior
agent = Agent("gpt-5-mini")

async def main():
    print("Testing stream...")
    # Using a simple prompt
    async with agent.run_stream("Count to 5") as result:
        async for chunk in result.stream():
            print(f"|{chunk}|")

if __name__ == "__main__":
    asyncio.run(main())
