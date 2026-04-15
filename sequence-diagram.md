# LLM Orchestration System - Sequence Diagram

## Main Chat Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Orchestrator
    participant BudgetManager
    participant Memory
    participant Scorer
    participant LLMClient
    participant ToolManager
    participant ExternalTool
    participant vLLM

    Client->>FastAPI: POST /chat (session_id, message)
    activate FastAPI
    
    FastAPI->>Orchestrator: process_query_stream(session_id, query)
    activate Orchestrator
    
    loop Max Iterations (25)
        Orchestrator->>BudgetManager: build_payload(session_id, system_prompt, query, recent_tools)
        activate BudgetManager
        
        BudgetManager->>Memory: get_session_context(session_id)
        Memory-->>BudgetManager: SessionContext
        BudgetManager->>Memory: get_history(session_id)
        Memory-->>BudgetManager: List[Message]
        
        BudgetManager->>Scorer: get_top_k_messages(history, token_budget)
        Scorer-->>BudgetManager: Selected messages (within budget)
        
        BudgetManager-->>Orchestrator: (messages, stats)
        deactivate BudgetManager
        
        Orchestrator->>LLMClient: chat_stream(messages, tools_schema)
        activate LLMClient
        
        LLMClient->>vLLM: POST /v1/chat/completions (stream=true)
        activate vLLM
        
        loop Stream chunks
            vLLM-->>LLMClient: data: {delta: {reasoning, content, tool_calls}}
            LLMClient-->>Orchestrator: yield {type: stream_reasoning/content, content}
        end
        
        Orchestrator->>FastAPI: yield streaming content
        FastAPI-->>Client: SSE stream (reasoning + content)
        
        deactivate vLLM
        deactivate LLMClient
        
        alt Has tool_calls
            Orchestrator->>ToolManager: execute(tool_call)
            activate ToolManager
            
            ToolManager->>ExternalTool: invoke(args)
            ExternalTool-->>ToolManager: result
            ToolManager-->>Orchestrator: Message(role="tool", content=result)
            deactivate ToolManager
            
            Orchestrator->>Memory: save_event(session_id, result)
            Orchestrator->>Orchestrator: Add to recent_tools, continue loop
        else No tool_calls
            Orchestrator->>Memory: save_message(session_id, query + response)
            Orchestrator-->>FastAPI: yield {type: "final", content, stats}
            deactivate Orchestrator
            break
        end
    end
    
    par Background Task
        FastAPI->>Memory: get_session_context + get_history
        FastAPI->>Compressor: compress(session_id, history, context, memory)
        Compressor->>Memory: update_session_context(session_id, updated_context)
    end
    
    FastAPI-->>Client: {response, tokens_before, tokens_after, compression_ratio, iterations}
    deactivate FastAPI
```

## Context Budget Management Flow

```mermaid
sequenceDiagram
    participant BM as BudgetManager
    participant Memory
    participant Scorer
    
    Note over BM,Scorer: Token Budget: 16000 total
    
    BM->>Memory: get_session_context(session_id)
    Memory-->>BM: SessionContext (~800 tokens)
    
    BM->>BM: Calculate remaining budget<br/>MAX - system - context - query
    
    BM->>BM: Add tool results (if within budget)
    
    BM->>Memory: get_history(session_id)
    Memory-->>BM: Full history
    
    BM->>Scorer: get_top_k_messages(history, remaining_budget)
    
    rect rgb(200, 220, 255)
        Note over Scorer: Scoring Algorithm
        Scorer->>Scorer: Reverse through history
        Scorer->>Scorer: Check token budget
        Scorer->>Scorer: Insert messages in order
    end
    
    Scorer-->>BM: Top K relevant messages
    
    BM->>BM: Build final payload:<br/>[system + context + tools + history + query]
    BM->>BM: Calculate compression ratio
    BM-->>BM: (messages, stats)
```

## Tool Execution Flow

```mermaid
sequenceDiagram
    participant Orchestrator
    participant ToolManager
    participant Registry
    participant LLMClient
    
    Orchestrator->>Orchestrator: Detect tool_calls in response
    
    loop Each tool_call
        Orchestrator->>ToolManager: execute(tool_call)
        
        ToolManager->>ToolManager: Parse tool name + arguments
        
        ToolManager->>Registry: lookup(tool_name)
        Registry-->>ToolManager: callable function
        
        alt Tool found
            ToolManager->>ToolManager: invoke(function, **args)
            ToolManager-->>ToolManager: result
        else Tool not found
            ToolManager-->>ToolManager: Error message
        end
        
        ToolManager-->>Orchestrator: Message(role="tool", content=result)
        
        Orchestrator->>Orchestrator: Check for duplicate calls
        alt Duplicate detected (3x)
            Orchestrator->>Orchestrator: Block and send error to LLM
        else New call
            Orchestrator->>Orchestrator: Add to recent_tools
        end
    end
    
    Orchestrator->>Orchestrator: Loop back with tool results
```

## Streaming Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Orchestrator
    participant LLMClient
    participant vLLM
    
    Client->>FastAPI: POST /chat
    
    FastAPI->>Orchestrator: process_query_stream()
    
    loop LLM streaming
        LLMClient->>vLLM: Request stream
        vLLM-->>LLMClient: chunk {reasoning: "..." }
        LLMClient-->>Orchestrator: {type: "stream_reasoning"}
        Orchestrator-->>FastAPI: yield {type: "content", "💭 Thinking..."}
        FastAPI-->>Client: SSE: *💭 Thinking...*\n>
        
        vLLM-->>LLMClient: chunk {reasoning: "more..."}
        LLMClient-->>Orchestrator: {type: "stream_reasoning"}
        Orchestrator-->>FastAPI: yield {type: "content", "\n> more..."}
        FastAPI-->>Client: SSE: \n> more...
        
        vLLM-->>LLMClient: chunk {content: "answer..."}
        LLMClient-->>Orchestrator: {type: "stream_content"}
        Orchestrator-->>FastAPI: yield {type: "content", "answer..."}
        FastAPI-->>Client: SSE: answer...
    end
    
    vLLM-->>LLMClient: chunk {tool_calls: [...]}
    LLMClient-->>Orchestrator: {type: "final_message", message}
    Orchestrator->>Orchestrator: Process tool calls (loop)
    
    alt No more tools
        Orchestrator-->>FastAPI: yield {type: "final", content, stats}
        FastAPI-->>Client: Final JSON response
    end
```

## OpenAI-Compatible Proxy Flow

```mermaid
sequenceDiagram
    participant Client
    participant Proxy
    participant vLLM
    
    Client->>Proxy: POST /v1/chat/completions (stream=true)
    
    Proxy->>Proxy: Budget truncation<br/>Keep messages within 16000 tokens
    
    Proxy->>Proxy: Inject system prompt<br/>(force reasoning output)
    
    Proxy->>vLLM: POST /v1/chat/completions
    
    loop Stream from vLLM
        vLLM-->>Proxy: data: {delta: {reasoning: "..."}}
        Proxy->>Proxy: Transform: reasoning -> reasoning_content
        Proxy-->>Client: data: {delta: {reasoning_content: "..."}}
    end
    
    vLLM-->>Proxy: data: [DONE]
    Proxy-->>Client: data: [DONE]
```

## Key Components

| Component | Responsibility |
|-----------|---------------|
| **Orchestrator** | Main loop: builds context, calls LLM, executes tools |
| **ContextBudgetManager** | Manages token budget, assembles final payload |
| **RelevanceScorer** | Selects most relevant history messages within budget |
| **MemoryManager** | Stores session context, history, and events |
| **ContextCompressor** | Summarizes conversation into structured context (background) |
| **ToolExecutionManager** | Executes tool calls and returns results |
| **VLLMClient** | Communicates with vLLM server via streaming |

## Token Budget Allocation

```
MAX_TOTAL_TOKENS: 16000
├── System Prompt:       2000
├── Session Context:       800
├── User Query:          2000
├── Tool Results:        4000
└── History (dynamic):   7200
```

## Iteration Limits

- **Max tool iterations**: 25
- **Duplicate tool call block**: After 3 identical calls
- **Tool output truncation**: 6000 characters max
