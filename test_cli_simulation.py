import asyncio
import httpx
import time
import json
import os
import random

NUM_USERS = 15 # 15 simulated developers
STEPS = 10     # Incremental growth: 5k -> 10k -> ... -> 100k
PROXY_URL = "http://localhost:8000/v1/chat/completions"

# --- REAL-WORLD ASSETS (Shared to trigger Prefix Caching) ---
SHARED_SYSTEM_PROMPT = "You are a senior full-stack engineer. You have access to a massive codebase and advanced CLI tools."
SHARED_TOOLS_SCHEMA = "INTERFACE_DEFINITION: " + ("tool_logic_metadata " * 4000)

test_results = {}

async def simulate_micro_step_developer(user_id: int, client: httpx.AsyncClient):
    user_key = f"Dev_{user_id}"
    test_results[user_key] = {"steps": []}
    
    # JITTERED START (Spread over 60s for real-world traffic distribution)
    delay = random.uniform(0, 30) 
    print(f"[{user_key}] ⏳ Waiting {round(delay, 2)}s to simulate organic arrival...")
    await asyncio.sleep(delay)
    
    messages = [{"role": "system", "content": SHARED_SYSTEM_PROMPT}]
    
    for step in range(1, STEPS + 1):
        target_kb = step * 10
        # ~3,750 hex chars is roughly 5k tokens
        new_block = os.urandom(7500).hex() 
        
        messages.append({"role": "user", "content": f"New code block added (Total {target_kb}k):\n{new_block}\n\nPlease check for syntax errors."})
        
        print(f"[{user_key}] 🧑‍💻 Requesting Step {step} (Total: {target_kb}k tokens)...")
        start_time = time.time()
        assistant_response = ""
        
        try:
            async with client.stream("POST", PROXY_URL, json={
                "model": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
                "messages": messages,
                "max_tokens": 256, # Fast response
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "validator", "description": SHARED_TOOLS_SCHEMA}}]
            }, timeout=300.0) as resp:
                if resp.status_code != 200:
                    print(f"[{user_key}] ❌ Failed: {resp.status_code}")
                    break

                first_token = True
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token:
                            ttft = time.time() - start_time
                            print(f"[{user_key}] ⚡ Step {step} TTFT: {round(ttft, 2)}s")
                            test_results[user_key]["steps"].append({"kb": target_kb, "ttft": round(ttft, 2)})
                            first_token = False
                        
                        data = json.loads(line[6:])
                        assistant_response += data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            
            messages.append({"role": "assistant", "content": assistant_response})
            
            # HUMAN THINK PAUSE (Shortened to keep test moving)
            await asyncio.sleep(random.uniform(3, 7))

        except Exception as e:
            print(f"[{user_key}] ❌ Error: {e}")
            break

async def main():
    print(f"🏗️  Starting MICRO-INCREMENTAL Simulation: {NUM_USERS} devs | {STEPS} steps | 5k increments.")
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=150)) as client:
        await asyncio.gather(*[simulate_micro_step_developer(i, client) for i in range(NUM_USERS)])
        
    with open("benchmark_micro_steps.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("✅ Micro-step benchmark complete.")

if __name__ == "__main__":
    asyncio.run(main())
