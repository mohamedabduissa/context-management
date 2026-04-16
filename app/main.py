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

# Models & Configuration
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


# Global State for Rolling Memory (Structured Context)
ACTIVE_SESSIONS: Dict[str, SessionContext] = {}

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

# Persistence Configuration
SESSIONS_DIR = os.path.join(os.getcwd(), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Summarization Service Configuration (Can be local or external API)
SUMMARY_ENDPOINT = os.getenv("SUMMARY_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "meta-llama/llama-3.2-3b-instruct:free")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "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") # Leave empty for local vLLM
#pip install fastapi uvicorn httpx tiktoken pydantic requests
# ---------------------------------------------------------
# Budget Configuration
# ---------------------------------------------------------


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


def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session_data(session_id: str) -> Dict[str, Any]:
    path = get_session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
    return {"context": SessionContext().model_dump(), "full_history": []}


def save_session_data(session_id: str, data: Dict[str, Any]):
    path = get_session_path(session_id)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")


def retrieve_relevant_context(query: str, full_history: List[Dict[str, Any]], top_k: int = 3) -> str:
    """Simple keyword matching to retrieve relevant past messages."""
    if not full_history:
        return ""
    
    keywords = set(query.lower().split())
    scored_msgs = []
    
    for msg in full_history:
        content = str(msg.get("content", "")).lower()
        score = sum(1 for kw in keywords if kw in content)
        if score > 0:
            scored_msgs.append((score, msg))
    
    # Sort by score and take top K
    scored_msgs.sort(key=lambda x: x[0], reverse=True)
    results = [m[1] for m in scored_msgs[:top_k]]
    
    if not results:
        return ""
        
    formatted = "\n---\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in results])
    return f"\n\n### RELEVANT PAST CONTEXT (Retrieved from Archive):\n{formatted}"


# OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
async def openai_chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    data = await req.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    session_id = data.get("session_id", "default_session")

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
    dropped_msgs = []
    idx = len(other_msgs) - 1
    while idx >= 0:
        msg = other_msgs[idx]
        raw_text = str(msg.get("content", ""))
        if msg.get("tool_calls"):
            raw_text += str(msg.get("tool_calls", ""))

        content_len = get_token_count(raw_text)
        if current_tokens + content_len < budget:
            compressed_messages.insert(0, msg)
            current_tokens += content_len
        else:
            # Capture all messages that didn't fit (from index 0 to current idx)
            dropped_msgs = other_msgs[: idx + 1]
            break
        idx -= 1

    if first_user_msg:
        compressed_messages.insert(0, first_user_msg)

    # HF Chat Templates (like Qwen) enforce that the first non-system message MUST be a 'user' role.
    # If our budget slice accidentally started on an 'assistant' or 'tool' message, the engine crashes.
    if not compressed_messages:
        compressed_messages.insert(0, {"role": "user", "content": "Continue."})
    elif compressed_messages[0].get("role") != "user":
        compressed_messages.insert(0, {"role": "user", "content": "[System: Earlier context was truncated due to budget.]"})

    # Memory Injection (Structured Dashboard + Long-term Retrieval)
    session_data = load_session_data(session_id)
    session_context = SessionContext(**session_data["context"])
    full_history = session_data["full_history"]

    # 1. Inject Dashboard State
    context_str = (
        f"### ACTIVE SESSION STATE ###\n"
        f"USER_INTENT: {session_context.user_intent}\n"
        f"CURRENT_TASK: {session_context.current_task}\n"
        f"FILES_IN_SCOPE: {', '.join(session_context.relevant_files)}\n"
        f"KEY_DECISIONS: {'; '.join(session_context.key_decisions)}\n"
        f"UNRESOLVED: {'; '.join(session_context.unresolved_issues)}\n"
        f"RECENT_ACTIONS: {'; '.join(session_context.recent_actions)}\n"
        f"#############################"
    )

    # 2. Semantic Retrieval (Keyword-based) from full history
    last_user_query = messages[-1].get("content", "") if messages else ""
    archived_context = retrieve_relevant_context(last_user_query, full_history)

    final_sys_content = f"{context_str}{archived_context}"
    
    if sys_msgs:
        sys_msgs[0]["content"] = str(sys_msgs[0].get("content", "")) + f"\n\n{final_sys_content}"
    else:
        sys_msgs.insert(0, {"role": "system", "content": final_sys_content})

    final_messages = sys_msgs + compressed_messages

    # Force system reasoning
    custom_sys = "CRITICAL INSTRUCTION: ALWAYS explain your reasoning out loud in the 'content' field BEFORE you use a tool."
    if final_messages and final_messages[0].get("role") == "system":
        final_messages[0]["content"] = (
            str(final_messages[0].get("content", "")) + f"\n\n{custom_sys}"
        )
    else:
        final_messages.insert(0, {"role": "system", "content": custom_sys})

    # Background Summarization
    if dropped_msgs:
        background_tasks.add_task(summarize_background_memory, session_id, dropped_msgs)

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


async def summarize_background_memory(session_id: str, dropped_msgs: List[Dict[str, Any]]):
    """
    Background task to perform technical state extraction from dropped messages.
    Also handles long-term persistence of history.
    """
    try:
        session_data = load_session_data(session_id)
        ctx = SessionContext(**session_data["context"])
        
        # Update full history on disk
        session_data["full_history"].extend(dropped_msgs)
        save_session_data(session_id, session_data)

        history_str = json.dumps(dropped_msgs, indent=2)

        extraction_prompt = (
            "You are a State Extraction Engine. Analyze the following conversation fragment and the existing session state. "
            "Update the session state with new details. You MUST return ONLY a valid JSON object matching the schema.\n\n"
            f"### EXISTING STATE (JSON):\n{ctx.model_dump_json()}\n\n"
            f"### NEW MESSAGES TO ANALYZE:\n{history_str}\n\n"
            "SCHEMA:\n"
            "{\n"
            "  \"user_intent\": \"string\",\n"
            "  \"current_task\": \"string\",\n"
            "  \"current_focus\": \"string\",\n"
            "  \"relevant_files\": [\"list of strings\"],\n"
            "  \"key_decisions\": [\"list of strings\"],\n"
            "  \"unresolved_issues\": [\"list of strings\"],\n"
            "  \"recent_actions\": [\"list of strings\"]\n"
            "}"
        )

        payload = {
            "model": SUMMARY_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a JSON-only state extraction server. No conversational filler."},
                {"role": "user", "content": extraction_prompt},
            ],
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        # Setup custom headers (for optional API Key)
        headers = {"Content-Type": "application/json"}
        if SUMMARY_API_KEY:
            headers["Authorization"] = f"Bearer {SUMMARY_API_KEY}"

        # Call the configured summary endpoint
        resp = await async_http_client.post(
            SUMMARY_ENDPOINT, 
            json=payload,
            headers=headers
        )
        if resp.status_code == 200:
            res_data = resp.json()
            raw_content = res_data["choices"][0]["message"]["content"]
            
            # Extract JSON from potential markdown blocks
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()

            new_ctx_dict = json.loads(raw_content)
            
            # Persist the new context back to disk
            session_data["context"] = new_ctx_dict
            save_session_data(session_id, session_data)
            
            logger.info(f"[MEMORY] Successfully updated state and archived history for: {session_id}")
        else:
            logger.error(f"[MEMORY ERROR] Status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"[MEMORY CRITICAL] Failed state extraction: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
