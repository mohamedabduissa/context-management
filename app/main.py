import json
import logging
import time
import os
import requests
import httpx
import tiktoken
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, Request, Body
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
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Models & Configuration
# ---------------------------------------------------------


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class SessionContext(BaseModel):
    user_intent: str = ""
    current_task: str = ""
    current_focus: str = ""
    relevant_files: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)


class BudgetConfig:
    MAX_TOTAL_TOKENS = 16000
    SYSTEM_PROMPT_BUDGET = 2000
    SESSION_CONTEXT_BUDGET = 800
    QUERY_BUDGET = 2000
    TOOL_RESULTS_BUDGET = 4000
    HISTORY_BUDGET = 7200


# Utility for authentic token counting
def get_token_count(text: str) -> int:
    if not text:
        return 0
    if tokenizer:
        return len(tokenizer.encode(str(text)))
    return len(str(text)) // 4


# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI()


# OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
async def openai_chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    data = await req.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    if not messages:
        return JSONResponse({"error": "messages required"}, status_code=400)

    # 1. Base Payloads & System Truncation
    # We restrict actual message memory strictly to 16,000 to allow 30+ concurrent vLLM users!
    budget = 16000
    current_tokens = 0
    compressed_messages = []

    # 1.a Intercept dangerously high max_tokens
    data["max_tokens"] = 8192
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 8192

    # 1.b Opencode's tool schema takes ~22,000 tokens passively.
    # We DO NOT add this to current_tokens. Validating tools against the 16k budget
    # would make current_tokens = 22k instantly, dropping the entire chat history on turn 1!
    tool_tokens = 0
    if "tools" in data:
        tool_tokens = get_token_count(json.dumps(data["tools"]))
        # current_tokens += tool_tokens  <-- REMOVED so msg history can survive

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    for sm in sys_msgs:
        current_tokens += get_token_count(sm.get("content", ""))

    first_user_msg = None
    if other_msgs and other_msgs[0].get("role") == "user":
        first_user_msg = other_msgs.pop(0)
        current_tokens += get_token_count(str(first_user_msg.get("content", "")))

    # Guaranteed inclusion of the absolute latest message to prevent Chat Template crashes
    if other_msgs:
        last_msg = other_msgs.pop(-1)
        compressed_messages.insert(0, last_msg)
        current_tokens += get_token_count(str(last_msg.get("content", "")))

    # Retrieve top most recent messages to stay strictly within limit
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

    if first_user_msg:
        compressed_messages.insert(0, first_user_msg)

    # HF Chat Templates (like Qwen) enforce that the first non-system message MUST be a 'user' role.
    # If our budget slice accidentally started on an 'assistant' or 'tool' message, the engine crashes.
    if not compressed_messages:
        compressed_messages.insert(0, {"role": "user", "content": "Continue."})
    elif compressed_messages[0].get("role") != "user":
        compressed_messages.insert(0, {"role": "user", "content": "[System: Earlier context was truncated due to budget.]"})

    final_messages = sys_msgs + compressed_messages

    # Force system reasoning
    custom_sys = "CRITICAL INSTRUCTION: ALWAYS explain your reasoning out loud in the 'content' field BEFORE you use a tool."
    if final_messages and final_messages[0].get("role") == "system":
        final_messages[0]["content"] = (
            str(final_messages[0].get("content", "")) + f"\n\n{custom_sys}"
        )
    else:
        final_messages.insert(0, {"role": "system", "content": custom_sys})

    original_token_count = sum(
        [
            get_token_count(str(m.get("content", "")) + str(m.get("tool_calls", "")))
            for m in messages
        ]
    )
    data["messages"] = final_messages
    logger.info(
        f"[CONTEXT DEBUG] Incoming tokens: ~{original_token_count} | Compressed tokens sent to vLLM: ~{current_tokens} | Budget Limit: {budget}"
    )

    # 2. Transparent Streaming Proxy
    if stream:

        async def async_generator():
            try:
                async with async_http_client.stream(
                    "POST", "http://localhost:8000/v1/chat/completions", json=data
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            decoded = line.strip()
                            if (
                                decoded.startswith("data: ")
                                and decoded != "data: [DONE]"
                            ):
                                try:
                                    chunk = json.loads(decoded[6:])
                                    delta = chunk.get("choices", [{}])[0].get(
                                        "delta", {}
                                    )

                                    # Intercept and route vLLM's internal "reasoning" to official OpenAI "reasoning_content"
                                    if "reasoning" in delta:
                                        r_text = delta.pop("reasoning")
                                        if r_text is not None:
                                            delta["reasoning_content"] = r_text
                                            decoded = "data: " + json.dumps(chunk)

                                except Exception as e:
                                    pass
                            yield decoded + "\n\n"
            except Exception as e:
                logger.error(f"Proxy Error: {e}")
                err_chunk = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "proxy",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"\nProxy Error: {e}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(async_generator(), media_type="text/event-stream")
    else:
        try:
            resp = await async_http_client.post(
                "http://localhost:8000/v1/chat/completions", json=data
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
