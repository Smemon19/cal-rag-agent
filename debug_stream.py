import asyncio
import os
from pydantic_ai import Agent

# Simple agent without RAG deps, just to test stream behavior
agent = Agent("gpt-5-mini")


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return result

async def main():
    print("Testing stream...")
    # Using a simple prompt
    async with agent.run_stream("Count to 5") as result:
        async for chunk in result.stream():
            print(f"|{chunk}|")

if __name__ == "__main__":
    _run_async(main())
