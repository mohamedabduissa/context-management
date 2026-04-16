import asyncio
import httpx
import time
import json

NUM_USERS = 20
PROXY_URL = "http://localhost:9000/v1/chat/completions"

# 1. We generate the massive static 100k string ONCE
massive_string = "This is a dummy log line meant to consume tokens rapidly and test the memory limit. " * 5000

# Store results for logging
test_results = {}

async def simulate_static_user(user_id: int, client: httpx.AsyncClient):
    payload = {
        "model": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
        "messages": [
            {"role": "system", "content": "You are a shared AI."},
            {"role": "user", "content": f"Shared Data Payload:\n\n{massive_string}\n\nVerdict?"}
        ],
        "max_tokens": 32000,
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "shared_tool", "description": "test"}}]
    }

    print(f"[User {user_id}] Initiating connection...")
    start = time.time()
    try:
        async with client.stream("POST", PROXY_URL, json=payload, timeout=300.0) as response:
            if response.status_code != 200:
                print(f"[User {user_id}] ❌ Failed: {response.status_code}")
                test_results[f"User_{user_id}"] = {"status": "failed", "error_code": response.status_code}
                return
            
            first_token = True
            async for line in response.aiter_lines():
                if line and first_token:
                    ttft = time.time() - start
                    print(f"[User {user_id}] ⚡ First token streamed! Time: {round(ttft, 2)}s")
                    test_results[f"User_{user_id}"] = {"status": "success", "ttft_seconds": round(ttft, 2)}
                    first_token = False
                    break 
                    
    except Exception as e:
        print(f"[User {user_id}] ❌ Error: {e}")
        test_results[f"User_{user_id}"] = {"status": "error", "error_msg": str(e)}

async def main():
    print(f"🚀 Engaging STATIC Cache Load Test: {NUM_USERS} concurrent users.")
    
    # 100 connections available
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=100)) as client:
        tasks = [simulate_static_user(i, client) for i in range(NUM_USERS)]
        await asyncio.gather(*tasks)
        
    print("\n💾 Saving benchmark results...")
    with open("benchmark_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("✅ Results saved to 'benchmark_results.json'")

if __name__ == "__main__":
    asyncio.run(main())
