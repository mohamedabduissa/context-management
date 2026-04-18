import json
import logging
import time
import httpx
import tiktoken
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Global HTTP Async Client for Connection Pooling
http_limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
async_http_client = httpx.AsyncClient(limits=http_limits, timeout=120.0)

# Global Tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI()

# Utility for authentic token counting
def get_token_count(text: str) -> int:
    if not text:
        return 0
    if tokenizer:
        return len(tokenizer.encode(str(text)))
    return len(str(text)) // 4

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-max",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    }

# OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
async def openai_chat_endpoint(req: Request):
    data = await req.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    if not messages:
        return JSONResponse({"error": "messages required"}, status_code=400)
    
    # Force model and set budgets
    data["model"] = "qwen-max"
    budget = 100000
    current_tokens = 0
    compressed_messages = []
    
    # 1.a Intercept max_tokens
    data["max_tokens"] = 8192
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 8192
        
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    
    for sm in sys_msgs:
        current_tokens += get_token_count(sm.get("content", ""))
        
    if other_msgs:
        last_msg = other_msgs.pop(-1)
        compressed_messages.insert(0, last_msg)
        current_tokens += get_token_count(str(last_msg.get("content", "")))
    
    for msg in reversed(other_msgs):
        raw_text = str(msg.get("content", ""))
        if msg.get("tool_calls"):
            raw_text += str(msg.get("tool_calls", ""))
            
        content_len = get_token_count(raw_text)
        if current_tokens + content_len < budget:
            compressed_messages.insert(0, msg)
            current_tokens += content_len
        else:
            break
            
    final_messages = sys_msgs + compressed_messages
    
    original_token_count = sum(
        [
            get_token_count(str(m.get("content", "")) + str(m.get("tool_calls", "")))
            for m in messages
        ]
    )
    
    print(f"Before logger - Compressed tokens: {current_tokens}")
    logger.info(
        f"[CONTEXT DEBUG] Incoming tokens: ~{original_token_count} | Compressed tokens: ~{current_tokens} | Budget: {budget}"
    )
    print(f"After logger - Compressed tokens: {current_tokens}")
    
    # Force system reasoning instruction
    custom_sys = "CRITICAL INSTRUCTION: ALWAYS explain your reasoning out loud in the 'content' field BEFORE you use a tool."
    if final_messages and final_messages[0].get("role") == "system":
        final_messages[0]["content"] = str(final_messages[0].get("content", "")) + f"\n\n{custom_sys}"
    else:
        final_messages.insert(0, {"role": "system", "content": custom_sys})
        
    data["messages"] = final_messages
    
    # 2. Transparent Streaming Proxy
    if stream:
        return StreamingResponse(stream_forward(data), media_type="text/event-stream")
    else:
        try:
            resp = await async_http_client.post("http://localhost:23333/v1/chat/completions", json=data)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

async def stream_forward(data: dict):
    # Stateful tracking for reasoning
    started_thinking = False
    
    try:
        async with async_http_client.stream("POST", "http://localhost:23333/v1/chat/completions", json=data) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    decoded = line.strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            chunk = json.loads(decoded[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            
                            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                            
                            if reasoning:
                                # We are in reasoning phase
                                if not started_thinking:
                                    delta["content"] = f"<thought>\n{reasoning}"
                                    started_thinking = True
                                else:
                                    delta["content"] = reasoning
                                
                                delta["reasoning_content"] = reasoning
                                if "reasoning" in delta: delta.pop("reasoning")
                                decoded = "data: " + json.dumps(chunk)
                            elif started_thinking:
                                # Transition from reasoning to content
                                content = delta.get("content") or ""
                                delta["content"] = f"\n</thought>\n\n" + content
                                started_thinking = False
                                decoded = "data: " + json.dumps(chunk)
                                
                        except Exception:
                            pass
                    yield decoded + "\n\n"
    except Exception as e:
        logger.error(f"Proxy Error: {e}")
        err_chunk = {"id": "error", "object": "chat.completion.chunk", "created": int(time.time()), "model": "proxy", "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"\nProxy Error: {e}"}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
