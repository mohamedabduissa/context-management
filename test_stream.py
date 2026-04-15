import asyncio
from openai import AsyncOpenAI
import traceback

async def test():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="test")
    try:
        response = await client.chat.completions.create(
            model="cyankiwi/Qwen3.5-27B-AWQ-4bit",
            messages=[{"role": "user", "content": "hello who are you"}],
            stream=True
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            print(f"Content: {repr(delta.content)} | Reasoning: {repr(getattr(delta, 'reasoning_content', None))} | Tool calls: {repr(delta.tool_calls)}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
