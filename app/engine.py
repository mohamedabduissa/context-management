import json
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Models & Configuration
# ---------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

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
    HISTORY_BUDGET = 7200 # Remaining for message history

def count_tokens(text: str) -> int:
    """Mock token counter. Use tiktoken in production."""
    return len(text.split())

# ---------------------------------------------------------
# 1. ContextCompressor
# ---------------------------------------------------------

class ContextCompressor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def compress(self, recent_history: List[Message], current_context: SessionContext) -> SessionContext:
        """
        Takes raw history and updates the structured session context via a summarization prompt.
        """
        prompt = f"""
        Analyze the recent conversation and update the session context.
        Current Context: {current_context.model_dump_json()}
        Recent Messages: {[m.model_dump() for m in recent_history[-5:]]}
        
        Return ONLY valid JSON matching the SessionContext schema.
        """
        # In production:
        # response = self.llm_client.generate(prompt, response_format="json")
        # return SessionContext(**json.loads(response))
        return current_context

# ---------------------------------------------------------
# 2. RelevanceScorer
# ---------------------------------------------------------

class RelevanceScorer:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def get_top_k_messages(self, query: str, history: List[Message], token_budget: int) -> List[Message]:
        """
        Scores history items based on recency + semantic similarity to query/active task.
        Returns messages that fit within the token budget.
        """
        selected = []
        current_tokens = 0
        
        for msg in reversed(history):
            msg_tokens = count_tokens(msg.content or "")
            if current_tokens + msg_tokens > token_budget:
                continue
            
            selected.insert(0, msg)
            current_tokens += msg_tokens
            
        return selected

# ---------------------------------------------------------
# 3. MemoryManager
# ---------------------------------------------------------

class MemoryManager:
    def __init__(self, redis_client=None):
        self.redis = redis_client

    def save_event(self, session_id: str, event: Any):
        """Append-only event log (never sent directly to LLM)."""
        # self.redis.rpush(f"session:{session_id}:events", json.dumps(event))
        pass

    def get_session_context(self, session_id: str) -> SessionContext:
        """Load summarized context state."""
        # raw = self.redis.get(f"session:{session_id}:context")
        # return SessionContext(**json.loads(raw)) if raw else SessionContext()
        return SessionContext()

    def update_session_context(self, session_id: str, context: SessionContext):
        # self.redis.set(f"session:{session_id}:context", context.model_dump_json())
        pass
        
    def get_history(self, session_id: str) -> List[Message]:
        # raw = self.redis.lrange(f"session:{session_id}:history", 0, -1)
        # return [Message(**json.loads(r)) for r in raw]
        return []

# ---------------------------------------------------------
# 4. ContextBudgetManager
# ---------------------------------------------------------

class ContextBudgetManager:
    def __init__(self, compressor: ContextCompressor, scorer: RelevanceScorer, memory: MemoryManager):
        self.compressor = compressor
        self.scorer = scorer
        self.memory = memory
        self.configs = BudgetConfig()

    def build_payload(self, session_id: str, system_prompt: str, user_query: str, recent_tools: List[Message]) -> List[Message]:
        """
        Constructs the final list of messages to send to the LLM strictly within budget.
        """
        payload = []
        
        # 1. System Prompt
        payload.append(Message(role="system", content=system_prompt))
        
        # 2. Session Context
        ctx = self.memory.get_session_context(session_id)
        ctx_str = f"SESSION CONTEXT:\n{ctx.model_dump_json()}"
        payload.append(Message(role="system", content=ctx_str))
        
        # 3. User Query
        query_msg = Message(role="user", content=user_query)
        
        history = self.memory.get_history(session_id)
        
        remaining_budget = (
            self.configs.MAX_TOTAL_TOKENS 
            - count_tokens(system_prompt)
            - count_tokens(ctx_str)
            - count_tokens(user_query)
        )
        
        # 4 & 5. Add tools and robust history retrieval
        for tool in recent_tools:
            t_tokens = count_tokens(tool.content)
            if remaining_budget - t_tokens > 0:
                payload.append(tool)
                remaining_budget -= t_tokens
                
        relevant_history = self.scorer.get_top_k_messages(user_query, history, remaining_budget)
        
        return payload + relevant_history + [query_msg]

# ---------------------------------------------------------
# 5. ToolExecutionManager
# ---------------------------------------------------------

class ToolExecutionManager:
    def __init__(self, tools_registry: Dict[str, callable]):
        self.registry = tools_registry

    def execute(self, tool_call: Dict[str, Any]) -> Message:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        if name not in self.registry:
            return Message(role="tool", name=name, content=f"Error: Tool '{name}' not found.")
            
        try:
            result = self.registry[name](**args)
            return Message(role="tool", name=name, content=str(result))
        except Exception as e:
            logging.error(f"Tool {name} failed: {e}")
            return Message(role="tool", name=name, content=f"Execution Failed: {str(e)}")

# ---------------------------------------------------------
# 6. Orchestrator Loop
# ---------------------------------------------------------

class Orchestrator:
    def __init__(self, budget_manager: ContextBudgetManager, tool_manager: ToolExecutionManager, llm_client):
        self.budget_manager = budget_manager
        self.tool_manager = tool_manager
        self.llm_client = llm_client
        
    def process_query(self, session_id: str, query: str):
        MAX_ITERATIONS = 5
        system_prompt = "You are a senior AI coding assistant. Solve the user's task using tools."
        recent_tools = []
        
        for iteration in range(MAX_ITERATIONS):
            # 1. Build context
            messages = self.budget_manager.build_payload(session_id, system_prompt, query, recent_tools)
            
            # 2. Call LLM
            response = self.llm_client.chat(messages) 
            
            # Record events
            self.budget_manager.memory.save_event(session_id, response.model_dump())
            
            # 3. Detect and handle tool calls
            if response.tool_calls:
                for t_call in response.tool_calls:
                    result_msg = self.tool_manager.execute(t_call)
                    recent_tools.append(result_msg)
                    self.budget_manager.memory.save_event(session_id, result_msg.model_dump())
            else:
                # 4. Success path & background context compression
                current_ctx = self.budget_manager.memory.get_session_context(session_id)
                new_ctx = self.budget_manager.compressor.compress(self.budget_manager.memory.get_history(session_id), current_ctx)
                self.budget_manager.memory.update_session_context(session_id, new_ctx)
                
                return response.content
                
        return "Error: Reached maximum tool execution iterations."
