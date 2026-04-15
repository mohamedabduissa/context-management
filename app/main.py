import json
import logging
import time
import os
import requests
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
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

# Utility to mock token counting
def mock_count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())

# ---------------------------------------------------------
# Local Storage / Memory 
# ---------------------------------------------------------
# Using in-memory store for demonstration instead of Redis

SESSIONS_HISTORY: Dict[str, List[Message]] = {}
SESSIONS_EVENTS: Dict[str, List[Any]] = {}
SESSIONS_CONTEXT: Dict[str, SessionContext] = {}

class MemoryManager:
    def save_event(self, session_id: str, event: Any):
        if session_id not in SESSIONS_EVENTS:
            SESSIONS_EVENTS[session_id] = []
        SESSIONS_EVENTS[session_id].append(event)
    
    def save_message(self, session_id: str, msg: Message):
        if session_id not in SESSIONS_HISTORY:
            SESSIONS_HISTORY[session_id] = []
        SESSIONS_HISTORY[session_id].append(msg)

    def get_session_context(self, session_id: str) -> SessionContext:
        if session_id not in SESSIONS_CONTEXT:
            SESSIONS_CONTEXT[session_id] = SessionContext()
        return SESSIONS_CONTEXT[session_id]

    def update_session_context(self, session_id: str, context: SessionContext):
        SESSIONS_CONTEXT[session_id] = context
        
    def get_history(self, session_id: str) -> List[Message]:
        return SESSIONS_HISTORY.get(session_id, [])

# ---------------------------------------------------------
# Clients Interfaces
# ---------------------------------------------------------

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1/chat/completions", model: str = "cyankiwi/Qwen3.5-27B-AWQ-4bit"):
        self.base_url = base_url
        self.model = model

    def chat_stream(self, messages: List[Message], tools: List[Dict] = None):
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content, "name": m.name, "tool_calls": m.tool_calls, "tool_call_id": m.tool_call_id} for m in messages if m.role],
            "temperature": 0.2,
            "stream": True
        }
        
        # Clean null values
        cleaned_messages = []
        for msg in payload["messages"]:
            cleaned_messages.append({k: v for k, v in msg.items() if v is not None})
        payload["messages"] = cleaned_messages
            
        if tools:
            payload["tools"] = tools
            
        logger.info(f"[LLM_CALL] Sending STREAM request to vLLM | model: {self.model} | messages: {len(messages)}")
        
        full_content = ""
        full_reasoning = ""
        tool_calls_accumulator = {}
        
        try:
            resp = requests.post(self.base_url, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        if json_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk["choices"][0]["delta"]
                            
                            if "reasoning" in delta and delta["reasoning"]:
                                r = delta["reasoning"]
                                full_reasoning += r
                                yield {"type": "stream_reasoning", "content": r}
                                
                            if "content" in delta and delta["content"]:
                                c = delta["content"]
                                full_content += c
                                yield {"type": "stream_content", "content": c}
                                
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc["index"]
                                    if idx not in tool_calls_accumulator:
                                        tool_calls_accumulator[idx] = {"id": tc.get("id", ""), "type": "function", "function": {"name": "", "arguments": ""}}
                                        
                                    tc_dict = tool_calls_accumulator[idx]
                                    if tc.get("id"): tc_dict["id"] = tc["id"]
                                    if "function" in tc:
                                        if tc["function"].get("name"):
                                            tc_dict["function"]["name"] += tc["function"]["name"]
                                        if tc["function"].get("arguments"):
                                            tc_dict["function"]["arguments"] += tc["function"]["arguments"]
                                            
                        except Exception as e:
                            pass
                            
            # Construct final message
            final_tools = list(tool_calls_accumulator.values()) if tool_calls_accumulator else None
            yield {"type": "final_message", "message": Message(role="assistant", content=full_content, reasoning=full_reasoning, tool_calls=final_tools)}
            
        except Exception as e:
            logger.error(f"[LLM_CALL] Failed: {e}")
            yield {"type": "final_message", "message": Message(role="assistant", content=f"Service Error: {str(e)}")}

# ---------------------------------------------------------
# Pipeline Components
# ---------------------------------------------------------

class ContextCompressor:
    def __init__(self, llm_client: VLLMClient):
        self.llm_client = llm_client

    def compress(self, session_id: str, recent_history: List[Message], current_context: SessionContext, memory: MemoryManager):
        logger.info(f"[CONTEXT] Running background compression for session: {session_id}")
        prompt = f"""
        Analyze the recent conversation and update the session context.
        Current Context: {current_context.model_dump_json()}
        Recent Messages: {[m.model_dump() for m in recent_history[-5:]]}
        
        Return ONLY valid JSON matching the SessionContext schema.
        """
        # Returning same context to save tokens in demo, 
        # normally you call LLM here to get the JSON back
        memory.update_session_context(session_id, current_context)

class RelevanceScorer:
    def get_top_k_messages(self, history: List[Message], token_budget: int) -> List[Message]:
        selected = []
        current_tokens = 0
        
        for msg in reversed(history):
            msg_tokens = mock_count_tokens(msg.content or "")
            if current_tokens + msg_tokens > token_budget:
                continue
            
            selected.insert(0, msg)
            current_tokens += msg_tokens
            
        return selected

class ContextBudgetManager:
    def __init__(self, scorer: RelevanceScorer, memory: MemoryManager):
        self.scorer = scorer
        self.memory = memory
        self.configs = BudgetConfig()

    def build_payload(self, session_id: str, system_prompt: str, user_query: str, recent_tools: List[Message]) -> tuple[List[Message], dict]:
        payload = []
        
        # Calculate raw tokens (Tokens Before)
        history = self.memory.get_history(session_id)
        raw_tokens = mock_count_tokens(system_prompt) + mock_count_tokens(user_query)
        raw_tokens += sum(mock_count_tokens(m.content or "") for m in history)
        raw_tokens += sum(mock_count_tokens(m.content or "") for m in recent_tools)

        # 1. Merged System Prompt
        ctx = self.memory.get_session_context(session_id)
        ctx_str = f"SESSION CONTEXT:\n{ctx.model_dump_json()}"
        
        combined_system = f"{system_prompt}\n\n{ctx_str}"
        payload.append(Message(role="system", content=combined_system))
        
        # 2. User Query
        query_msg = Message(role="user", content=user_query)
        
        remaining_budget = (
            self.configs.MAX_TOTAL_TOKENS 
            - mock_count_tokens(system_prompt)
            - mock_count_tokens(ctx_str)
            - mock_count_tokens(user_query)
        )
        
        # 4. Tools
        tools_selected = 0
        for tool in recent_tools:
            t_tokens = mock_count_tokens(tool.content or "")
            if remaining_budget - t_tokens > 0:
                payload.append(tool)
                remaining_budget -= t_tokens
                tools_selected += 1
                
        # 5. History
        relevant_history = self.scorer.get_top_k_messages(history, remaining_budget)
        
        final_payload = payload + relevant_history + [query_msg]
        
        # Calculate compressed tokens
        tokens_after = sum(mock_count_tokens(m.content or "") for m in final_payload)
        
        stats = {
            "tokens_before": raw_tokens,
            "tokens_after": tokens_after,
            "compression_ratio": round(tokens_after / max(raw_tokens, 1), 2),
            "messages_selected": len(relevant_history),
            "tool_results_selected": tools_selected
        }
        
        return final_payload, stats

class ToolExecutionManager:
    def __init__(self, tools_registry: Dict[str, callable]):
        self.registry = tools_registry

    def execute(self, tool_call: Dict[str, Any]) -> Message:
        func_obj = tool_call.get("function", {})
        name = func_obj.get("name")
        args_str = func_obj.get("arguments", "{}")
        tool_id = tool_call.get("id")
        
        try:
            args = json.loads(args_str)
        except:
            args = {}
            
        logger.info(f"[TOOL_CALL] Executing {name} with args: {args}")
        
        if name not in self.registry:
            logger.error(f"[TOOL_CALL] Tool {name} not found")
            return Message(role="tool", tool_call_id=tool_id, name=name, content=f"Error: Tool '{name}' not found.")
            
        try:
            result = self.registry[name](**args)
            return Message(role="tool", tool_call_id=tool_id, name=name, content=str(result))
        except Exception as e:
            logger.error(f"[TOOL_CALL] Tool {name} failed: {e}")
            return Message(role="tool", tool_call_id=tool_id, name=name, content=f"Execution Failed: {str(e)}")

# ---------------------------------------------------------
# Tools Definition
# ---------------------------------------------------------

def list_directory(path: str = ".") -> str:
    """Lists files in a given directory."""
    try:
        files = os.listdir(path)
        return json.dumps({"files": files})
    except Exception as e:
        return str(e)

TOOLS_REGISTRY = {
    "list_directory": list_directory
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files in a given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to list"}
                },
                "required": ["path"]
            }
        }
    }
]

# ---------------------------------------------------------
# Orchestrator Loop
# ---------------------------------------------------------

class Orchestrator:
    def __init__(self, budget_manager: ContextBudgetManager, tool_manager: ToolExecutionManager, llm_client: VLLMClient):
        self.budget_manager = budget_manager
        self.tool_manager = tool_manager
        self.llm_client = llm_client
        
    def process_query_stream(self, session_id: str, query: str):
        MAX_ITERATIONS = 25
        system_prompt = (
            "You are a senior AI coding assistant. "
            "CRITICAL: ALWAYS explain your reasoning out loud BEFORE you use a tool. "
            "Write your internal thoughts in the 'content' field before triggering 'tool_calls'. "
            "If a tool call fails or is blocked, DO NOT repeat it identically. Rethink your approach."
        )
        recent_tools = []
        stats = {}
        
        last_tool_signature = ""
        duplicate_count = 0
        
        logger.info(f"[REQUEST] New request | session_id: {session_id} | query: {query}")
        
        for iteration in range(MAX_ITERATIONS):
            # 1. Build Payload
            messages, current_stats = self.budget_manager.build_payload(session_id, system_prompt, query, recent_tools)
            if iteration == 0:
                stats = current_stats
                
            logger.info(f"[CONTEXT DEBUG] tokens_before: {current_stats['tokens_before']}, tokens_after: {current_stats['tokens_after']}, compression_ratio: {current_stats['compression_ratio']}")

            # 2. Call LLM (Streaming)
            response = None
            in_thought_block = False
            
            for chunk in self.llm_client.chat_stream(messages, TOOLS_SCHEMA):
                if chunk["type"] == "stream_reasoning":
                    if not in_thought_block:
                        yield {"type": "content", "content": "\n*💭 Thinking...*\n> "}
                        in_thought_block = True
                    
                    text = chunk["content"]
                    formatted = text.replace("\n", "\n> ")
                    yield {"type": "content", "content": formatted}
                    
                elif chunk["type"] == "stream_content":
                    if in_thought_block:
                        yield {"type": "content", "content": "\n\n"}
                        in_thought_block = False
                        
                    yield {"type": "content", "content": chunk["content"]}
                    
                elif chunk["type"] == "final_message":
                    response = chunk["message"]
                    
            if in_thought_block:
                yield {"type": "content", "content": "\n\n"}
                
            self.budget_manager.memory.save_event(session_id, response.model_dump())
            
            # 3. Tool Loop
            if response.tool_calls:
                recent_tools.append(response)
                
                for t_call in response.tool_calls:
                    func_obj = t_call.get("function", {})
                    func_name = func_obj.get("name", "unknown_tool")
                    args_str = func_obj.get("arguments", "")
                    
                    sig = f"{func_name}:{args_str}"
                    if sig == last_tool_signature:
                        duplicate_count += 1
                        if duplicate_count >= 3:
                            err_msg = Message(role="tool", tool_call_id=t_call.get("id"), name=func_name, content="System: Repeated tool call blocked. Please rethink your approach or stop.")
                            recent_tools.append(err_msg)
                            self.budget_manager.memory.save_event(session_id, err_msg.model_dump())
                            continue
                    else:
                        last_tool_signature = sig
                        duplicate_count = 0
                        
                    result_msg = self.tool_manager.execute(t_call)
                    
                    # Prevent massive command outputs from crashing the model context next loop
                    if len(result_msg.content) > 6000:
                        result_msg.content = result_msg.content[:6000] + "\n...[TRUNCATED ALERTS: Output too large]"
                        
                    recent_tools.append(result_msg)
                    self.budget_manager.memory.save_event(session_id, result_msg.model_dump())
            else:
                logger.info(f"[FINAL_RESPONSE] Generated final response after {iteration + 1} iterations")
                self.budget_manager.memory.save_message(session_id, Message(role="user", content=query))
                self.budget_manager.memory.save_message(session_id, response)
                
                yield {
                    "type": "final",
                    "content": response.content,
                    "tokens_before": stats.get("tokens_before", 0),
                    "tokens_after": stats.get("tokens_after", 0),
                    "compression_ratio": stats.get("compression_ratio", 0.0),
                    "iterations": iteration + 1,
                    "final_payload": [m.model_dump() for m in messages]
                }
                return
                
        yield {
            "type": "final",
            "content": "Error: Reached maximum tool execution iterations.",
            "iterations": MAX_ITERATIONS
        }

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI()

# Initializers
llm_client = VLLMClient(base_url="http://localhost:8000/v1/chat/completions", model="cyankiwi/Qwen3.5-27B-AWQ-4bit")
memory = MemoryManager()
scorer = RelevanceScorer()
budget_manager = ContextBudgetManager(scorer, memory)
tool_manager = ToolExecutionManager(TOOLS_REGISTRY)
compressor = ContextCompressor(llm_client)
orchestrator = Orchestrator(budget_manager, tool_manager, llm_client)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks, debug: bool = True):
    final_result = None
    for event in orchestrator.process_query_stream(req.session_id, req.message):
        if event["type"] == "final":
            final_result = event
            
    if not final_result:
        final_result = {"content": "Error missing payload", "tokens_after": 0, "tokens_before": 0, "compression_ratio": 0.0, "iterations": 0}
    
    # Schedule background compression
    current_ctx = memory.get_session_context(req.session_id)
    history = memory.get_history(req.session_id)
    background_tasks.add_task(compressor.compress, req.session_id, history, current_ctx, memory)
    
    res_data = {
        "response": final_result.get("content", final_result.get("response", "")),
        "tokens_before": final_result.get("tokens_before", 0),
        "tokens_after": final_result.get("tokens_after", 0),
        "compression_ratio": final_result.get("compression_ratio", 0.0),
        "iterations": final_result.get("iterations", 1)
    }
    
    if debug:
        res_data["debug_payload"] = final_result.get("final_payload", [])
        
    return res_data

# OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
async def openai_chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    data = await req.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    if not messages:
        return JSONResponse({"error": "messages required"}, status_code=400)
    
    # 1. Budget Truncation (Prevent OOM)
    budget = 16000
    current_tokens = 0
    compressed_messages = []
    
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    
    # Retrieve top most recent messages to stay strictly within limit
    for msg in reversed(other_msgs):
        content_len = len(str(msg.get("content", ""))) // 4  # Very naive token count
        if current_tokens + content_len < budget:
            compressed_messages.insert(0, msg)
            current_tokens += content_len
        else:
            break
            
    final_messages = sys_msgs + compressed_messages
    
    # Force system reasoning
    custom_sys = "CRITICAL INSTRUCTION: ALWAYS explain your reasoning out loud in the 'content' field BEFORE you use a tool."
    if final_messages and final_messages[0].get("role") == "system":
        final_messages[0]["content"] = str(final_messages[0].get("content", "")) + f"\n\n{custom_sys}"
    else:
        final_messages.insert(0, {"role": "system", "content": custom_sys})
        
    data["messages"] = final_messages
    logger.info(f"[CONTEXT DEBUG] Incoming tokens: ~{len(str(messages)) // 4} | Compressed tokens sent to vLLM: ~{current_tokens} | Budget Limit: {budget}")
    
    # 2. Transparent Streaming Proxy
    if stream:
        def sync_generator():
            try:
                resp = requests.post("http://localhost:8000/v1/chat/completions", json=data, stream=True, timeout=120)
                resp.raise_for_status()
                in_reasoning = False
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode('utf-8').strip()
                        if decoded.startswith("data: ") and decoded != "data: [DONE]":
                            try:
                                chunk = json.loads(decoded[6:])
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                
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
                err_chunk = {"id": "error", "object": "chat.completion.chunk", "created": int(time.time()), "model": "proxy", "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"\nProxy Error: {e}"}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n"
                
        return StreamingResponse(sync_generator(), media_type="text/event-stream")
    else:
        try:
            resp = requests.post("http://localhost:8000/v1/chat/completions", json=data, timeout=120)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
