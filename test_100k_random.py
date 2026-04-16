import asyncio
import httpx
import time
import json
import os

NUM_USERS = 20
PROXY_URL = "http://localhost:9000/v1/chat/completions"

# Store results for logging
test_results = {}

async def simulate_random_user(user_id: int, client: httpx.AsyncClient):
    # Generate unique random context per user to bypass prefix caching
    # os.urandom produces random bytes, we convert to hex to fill token space
    # ~150kb of hex string is roughly 100k+ tokens
    random_context = os.urandom(75000).hex()
    
    payload = {
        "model": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
        "messages": [
            {"role": "system", "content": f"You are AI assistant for User {user_id}."},
            {"role": "user", "content": f"Random Unique Payload for user {user_id}:\n\n{random_context}\n\nVerdict?"}
        ],
        "max_tokens": 32000,
        "stream": True,
        "tools": [{"type": "function", "function": {"name": f"random_tool_{user_id}", "description": "test"}}]
    }

    print(f"[User {user_id}] Initiating connection with unique random context...")
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
    print(f"🚀 Engaging RANDOM Context Load Test: {NUM_USERS} concurrent users.")
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=100)) as client:
        tasks = [simulate_random_user(i, client) for i in range(NUM_USERS)]
        await asyncio.gather(*tasks)
        
    print("\n💾 Saving random benchmark results...")
    with open("benchmark_results_random.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("✅ Results saved to 'benchmark_results_random.json'")

if __name__ == "__main__":
    asyncio.run(main())
