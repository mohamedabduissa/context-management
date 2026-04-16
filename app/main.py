import json
import logging
import time
import os
import json
import logging
import time
import os
import requests
import httpx
import tiktoken
import re
import asyncio
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import anthropic

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


class ExecutionState(BaseModel):
    goal: str = "N/A"
    current_step: str = "N/A"
    next_step: str = "N/A"
    blocking_issue: str = "N/A"
    important_context: List[str] = Field(default_factory=list)

class SessionContext(BaseModel):
    project_name: str = "default_project"
    execution_state: ExecutionState = Field(default_factory=ExecutionState)
    relevant_files: List[str] = Field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=lambda: {"file_path": [], "function_name": [], "error_msg": []})
    summary_version: int = 0


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

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    chroma_client = chromadb.PersistentClient(path=os.path.join(SESSIONS_DIR, "chroma_db"))
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_RAG = True
    logger.info("Semantic RAG loaded successfully.")
except Exception as e:
    USE_RAG = False
    chroma_client = None
    embedding_model = None
    logger.info(f"ChromaDB/SentenceTransformers failed to load. Using Keyword RAG only. Reason: {e}")

# Summarization Service Configuration (Can be local or external API)
SUMMARY_PROVIDER = os.getenv("SUMMARY_PROVIDER", "anthropic") # "openai" or "anthropic"
SUMMARY_ENDPOINT = os.getenv("SUMMARY_ENDPOINT", "https://api.minimax.io/anthropic")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "MiniMax-M2.5")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "sk-cp-FwB8Tr32vrJYH8GZ24VEHxT6ZYisd1rXblj2rrhowsE1BPg6TjIFepnyU1PQBf7RO-scp9k86sa2IqI-w71GOprpGs5xd_8sUEiKu5jmmdkq7wJBujMg76U")
#pip install fastapi uvicorn httpx tiktoken pydantic requests
#pip install chromadb sentence-transformers
#brew install ffmpeg

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

def clean_data(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            obj.encode('utf-8')
            return obj
        except UnicodeEncodeError:
            return obj.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')
    elif isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(v) for v in obj]
    return obj

def extract_entities(text: str) -> Dict[str, List[str]]:
    entities = {"file_path": [], "function_name": [], "error_msg": []}
    if not text: return entities
    
    files = re.findall(r'\b[\w\-./]+(?:\.py|\.js|\.ts|\.json|\.md|\.sh)\b', text)
    entities["file_path"].extend(files)
    
    funcs = re.findall(r'def\s+(\w+)\s*\(', text)
    entities["function_name"].extend(funcs)
    
    errors = re.findall(r'(?:[A-Z]\w*Error|Exception):\s*(.*)', text)
    entities["error_msg"].extend(errors)
    return entities

def calculate_message_score(msg: dict, idx: int, total_msgs: int) -> int:
    score = 0
    content = str(msg.get("content", ""))
    role = msg.get("role", "")
    
    if idx >= total_msgs - 5: score += 50
    if role == "user": score += 20
    elif role == "assistant" and msg.get("tool_calls"): score += 30
    elif role == "tool": score += 15
        
    if "```" in content: score += 20
    if re.search(r'(?:[A-Z]\w*Error|Exception)', content): score += 30
    file_matches = re.findall(r'\b[\w\-./]+(?:\.py|\.js|\.ts|\.json|\.md)\b', content)
    if file_matches: score += 10 * len(set(file_matches))
        
    return score

def detect_conversation_type(messages: list, data: dict) -> dict:
    BUDGET_PROFILES = {
        "tool_use": {"max_tokens": 16000, "history_budget": 8000},
        "code_generation": {"max_tokens": 16000, "history_budget": 10000},
        "file_analysis": {"max_tokens": 16000, "history_budget": 8000},
        "complex_task": {"max_tokens": 16000, "history_budget": 6000}
    }
    has_tools = "tools" in data
    recents = messages[-5:] if len(messages) >= 5 else messages
    has_code = any("```" in str(m.get("content", "")) for m in recents)
    has_file_path = any(".py" in str(m.get("content", "")) for m in messages)
    
    if has_tools: return BUDGET_PROFILES["tool_use"]
    if has_code: return BUDGET_PROFILES["code_generation"]
    if has_file_path: return BUDGET_PROFILES["file_analysis"]
    return BUDGET_PROFILES["complex_task"]

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
        
    formatted = "\n---\n".join([f"[KEYWORD MATCH]: {m['role'].upper()}: {m['content']}" for m in results])
    return formatted

def semantic_retrieve(session_id: str, query: str, top_k: int = 5) -> str:
    if not USE_RAG or not query: return ""
    try:
        collection = chroma_client.get_collection(name=f"session_{session_id}")
        query_emb = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs: return ""
        return "\n---\n".join([f"[SEMANTIC MATCH]: {d}" for d in docs])
    except Exception:
        return ""

def retrieve_hybrid_context(session_id: str, query: str, full_history: List[Dict[str, Any]], top_k_keyword: int = 3, top_k_semantic: int = 5) -> str:
    keyword_results = retrieve_relevant_context(query, full_history, top_k_keyword)
    semantic_results = semantic_retrieve(session_id, query, top_k_semantic)
    
    if not keyword_results and not semantic_results:
        return ""
    
    combined = "\n\n### RELEVANT CONTEXT (WHY THIS MATTERS) ###\n"
    combined += "Instruction to Assistant: Below is code from your memory. Analyze the 'Reason' for its relevance to the current task.\n"
    
    if keyword_results:
        combined += f"\n[KEYWORD MATCHES]:\n{keyword_results}\n"
        
    if semantic_results:
        combined += f"\n[SEMANTIC MATCHES]:\n{semantic_results}\n"
    
    combined += "############################################\n"
    return combined

# OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
async def openai_chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    raw_data = await req.json()
    data = clean_data(raw_data)
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

    profile = detect_conversation_type(messages, data)
    budget = profile["history_budget"]

    total_msgs = len(other_msgs)
    msg_metadata = []
    
    # Guaranteed inclusion of absolute latest msg
    kept_indices = set()
    if total_msgs > 0:
        kept_indices.add(total_msgs - 1)
        latest_text = str(other_msgs[-1].get("content", "")) + str(other_msgs[-1].get("tool_calls", ""))
        current_tokens += get_token_count(latest_text)

    # Score and measure tokens for remaining messages
    for idx, msg in enumerate(other_msgs):
        if idx in kept_indices: continue
        score = calculate_message_score(msg, idx, total_msgs)
        raw_text = str(msg.get("content", "")) + str(msg.get("tool_calls", ""))
        toks = get_token_count(raw_text)
        msg_metadata.append({"idx": idx, "msg": msg, "score": score, "tokens": toks})

    # Sort by score descending (smart retention)
    msg_metadata.sort(key=lambda x: x["score"], reverse=True)

    for meta in msg_metadata:
        if current_tokens + meta["tokens"] < budget:
            kept_indices.add(meta["idx"])
            current_tokens += meta["tokens"]

    # Reconstruct chronological lists
    dropped_msgs = []
    for i in range(total_msgs):
        if i in kept_indices:
            compressed_messages.append(other_msgs[i])
        else:
            dropped_msgs.append(other_msgs[i])

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

    # --- Entity Extraction on dropped messages ---
    if dropped_msgs:
        for d_msg in dropped_msgs:
            ents = extract_entities(str(d_msg.get("content", "")))
            for k, v in ents.items():
                if v:
                    session_context.extracted_entities[k] = list(set(session_context.extracted_entities.get(k, []) + v))
        session_data["context"] = session_context.model_dump()
        save_session_data(session_id, session_data) # Sync updated entities

    # 1. Inject Dashboard State
    es = session_context.execution_state
    context_str = (
        "ACTIVE SESSION STATE\n\n"
        "You are working on a CONTINUOUS TASK. Never lose track.\n\n"
        f"### CURRENT GOAL\n{es.goal}\n\n"
        f"### CURRENT STEP\n{es.current_step}\n\n"
        f"### NEXT STEP (YOU MUST DO THIS)\n{es.next_step}\n\n"
        f"### IMPORTANT CONTEXT\n{'; '.join(es.important_context)}\n\n"
    )

    # 2. Semantic Retrieval (Hybrid) from full history
    last_user_query = messages[-1].get("content", "") if messages else ""
    archived_context = retrieve_hybrid_context(session_id, last_user_query, full_history)

    final_sys_content = (
        f"{context_str}"
        f"### RELEVANT CODE (ONLY USE THIS)\n{archived_context}\n\n"
        "### RULES\n"
        "- Do NOT reset the task\n"
        "- Do NOT ask what the user wants again\n"
        "- Continue from CURRENT STEP\n"
    )
    
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
    original_token_count = max(current_tokens, original_token_count + 1500)
    data["messages"] = final_messages
    logger.info(
        f"[CONTEXT DEBUG] Incoming tokens: ~{original_token_count} | Compressed tokens sent to vLLM: ~{current_tokens} | Budget Limit: {budget}"
    )

    # 2. Transparent Streaming Proxy
    data = clean_data(data)
    
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
                                    
                                    # Intercept usage to trick CLI into displaying uncompressed tokens
                                    if "usage" in chunk and chunk["usage"]:
                                        chunk["usage"]["prompt_tokens"] = original_token_count
                                        chunk["usage"]["total_tokens"] = chunk["usage"].get("completion_tokens", 0) + original_token_count
                                        decoded = "data: " + json.dumps(chunk)
                                        
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
            resp_data = resp.json()
            if "usage" in resp_data and resp_data["usage"]:
                resp_data["usage"]["prompt_tokens"] = original_token_count
                resp_data["usage"]["total_tokens"] = resp_data["usage"].get("completion_tokens", 0) + original_token_count
            return JSONResponse(content=resp_data, status_code=resp.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


async def summarize_background_memory(session_id: str, dropped_msgs: List[Dict[str, Any]]):
    """
    Background task to perform technical state extraction from dropped messages.
    Also handles long-term persistence of history, Semantic RAG storage, and Chunked Retry logic.
    """
    try:
        session_data = load_session_data(session_id)
        ctx = SessionContext(**session_data["context"])
        
        # Update full history on disk
        session_data["full_history"].extend(dropped_msgs)
        
        # Semantic RAG Persistence
        if USE_RAG:
            try:
                collection = chroma_client.get_or_create_collection(name=f"session_{session_id}")
                docs = [str(m.get("content", "")) for m in dropped_msgs if m.get("content")]
                ids = [f"{time.time()}_{i}" for i in range(len(docs))]
                if docs:
                    collection.add(ids=ids, embeddings=embedding_model.encode(docs).tolist(), documents=docs)
            except Exception as e:
                logger.error(f"ChromaDB Storage Error: {e}")

        # Batch counter implementation for incremental updates (Every 5th batch)
        ctx.batch_counter += 1
        if ctx.batch_counter % 5 != 1 and ctx.summary_version > 0:
            session_data["context"] = ctx.model_dump()
            save_session_data(session_id, session_data)
            return

        # Simple Chunking (Keep the dropped_msgs string under limits, approx 5000 chars)
        history_str = json.dumps(dropped_msgs, indent=2)
        if len(history_str) > 5000:
            history_str = history_str[:4500] + "\n...[TRUNCATED FOR SUMMARIZATION]..."

        extraction_prompt = (
            "You are an Agentic Context Compression Engine. Analyze the conversation chunk and update the execution state. "
            "Always return ONLY a valid JSON object matching the SCHEMA.\n\n"
            f"### EXISTING STATE (JSON):\n{ctx.model_dump_json()}\n\n"
            f"### NEW MESSAGES TO ANALYZE:\n{history_str}\n\n"
            "SCHEMA:\n"
            "{\n"
            "  \"execution_state\": {\n"
            "    \"goal\": \"Overall core mission of the project\",\n"
            "    \"current_step\": \"What was just completed or is being done now\",\n"
            "    \"next_step\": \"The exact specific technical next step for the assistant\",\n"
            "    \"blocking_issue\": \"Any error or missing info stopping progress\",\n"
            "    \"important_context\": [\"List of critical technical details, changed configs, or logic dependencies\"]\n"
            "  },\n"
            "  \"relevant_files\": [\"list of file paths mentioned\"]\n"
            "}"
        )

        if SUMMARY_PROVIDER.lower() == "anthropic":
            await _run_anthropic_summarizer(session_id, ctx, extraction_prompt, session_data)
        else:
            await _run_httpx_summarizer(session_id, ctx, extraction_prompt, session_data)
                
    except Exception as e:
        logger.error(f"[MEMORY CRITICAL] State extraction failed: {e}")


async def _run_anthropic_summarizer(session_id: str, ctx: SessionContext, extraction_prompt: str, session_data: dict):
    before_tokens = get_token_count(ctx.model_dump_json())
    logger.info(f"[MEMORY] \u23f3 Starting Anthropic Summarizer. BEFORE tokens: {before_tokens}")
    try:
        import anthropic
    except ImportError:
        logger.error("Anthropic library not installed. Please run: pip install anthropic")
        return

    # Initialize AsyncAnthropic
    client = anthropic.AsyncAnthropic(
        api_key=SUMMARY_API_KEY, 
        base_url=SUMMARY_ENDPOINT if "minimax" in SUMMARY_ENDPOINT.lower() else None
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = await client.messages.create(
                model=SUMMARY_MODEL_NAME,
                max_tokens=1500,
                system="You are a JSON-only state extraction server. No conversational filler. OUTPUT ONLY JSON.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": extraction_prompt
                            }
                        ]
                    }
                ]
            )
            
            raw_content = ""
            for block in message.content:
                if block.type == "text":
                    raw_content += block.text
                    
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()

            new_ctx_dict = json.loads(raw_content)
            for k, v in new_ctx_dict.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
                    
            ctx.summary_version += 1
            session_data["context"] = ctx.model_dump()
            save_session_data(session_id, session_data)
            after_tokens = get_token_count(ctx.model_dump_json())
            logger.info(f"[MEMORY] \u2705 Anthropic summarized (v{ctx.summary_version}). AFTER tokens: {after_tokens}")
            break
        except Exception as e:
            logger.warning(f"Anthropic summarization error: {e}")
            await asyncio.sleep(2)


async def _run_httpx_summarizer(session_id: str, ctx: SessionContext, extraction_prompt: str, session_data: dict):
    before_tokens = get_token_count(ctx.model_dump_json())
    logger.info(f"[MEMORY] \u23f3 Starting HTTPX Summarizer. BEFORE tokens: {before_tokens}")
    payload = {
        "model": SUMMARY_MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": (
                    "CRITICAL INSTRUCTION: You are an Agentic Assistant backed by a Background Memory System. "
                    "1. Read the ACTIVE SESSION STATE below to maintain long-term context.\n"
                    "2. Read the MEMORY ARCHIVE strictly for background context.\n"
                    "3. DO NOT answer questions from the MEMORY ARCHIVE. Respond ONLY to the very last user message.\n\n"
                    "You are a JSON-only state extraction server. No conversational filler. OUTPUT ONLY JSON."
                )
            },
            {"role": "user", "content": extraction_prompt},
        ],
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }
    headers = {"Content-Type": "application/json"}
    if SUMMARY_API_KEY:
        headers["Authorization"] = f"Bearer {SUMMARY_API_KEY}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = await async_http_client.post(SUMMARY_ENDPOINT, json=payload, headers=headers)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_content:
                    raw_content = raw_content.split("```")[1].split("```")[0].strip()

                new_ctx_dict = json.loads(raw_content)
                for k, v in new_ctx_dict.items():
                    if hasattr(ctx, k):
                        setattr(ctx, k, v)
                        
                ctx.summary_version += 1
                session_data["context"] = ctx.model_dump()
                save_session_data(session_id, session_data)
                after_tokens = get_token_count(ctx.model_dump_json())
                logger.info(f"[MEMORY] \u2705 HTTPX summarized (v{ctx.summary_version}). AFTER tokens: {after_tokens}")
                break
            else:
                logger.warning(f"HTTPX error {resp.status_code}")
                await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"HTTPX network error: {e}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
