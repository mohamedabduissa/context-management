# Complete Implementation Plan: 100K Effective Context

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID APPROACH                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Active Context (16K)                             │
│    - Smart message scoring (keep high-value messages)       │
│    - Entity extraction (files, functions, errors)           │
│                                                              │
│  Layer 2: Session Memory (Persistent)                       │
│    - Structured state per session                           │
│    - Extracted entities preserved                           │
│    - Project context tracking                               │
│                                                              │
│  Layer 3: Long-term Memory (Archived)                       │
│    - Full history stored to disk                            │
│    - LLM summarization of dropped messages                  │
│    - RAG retrieval with embeddings (optional)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Compression (Already Started)

### 1.1 Smart Message Scoring ✅ (In Progress)
**Goal**: Keep valuable messages, drop low-value ones

**Implementation**:
- [x] `calculate_message_score()` - score by role, tools, code, errors, recency
- [x] Replace FIFO dropping with score-based selection
- [x] Extract entities before dropping messages

**Code Location**: `app/main.py:79-137`

---

### 1.2 Entity Extraction ✅ (In Progress)
**Goal**: Preserve critical info even when messages are dropped

**Implementation**:
- [x] `extract_entities()` - regex patterns for files, functions, classes, errors
- [x] `create_entity_summary()` - compact string representation
- [x] Store entities in `SessionContext.extracted_entities`
- [ ] Inject entity summary into system prompt

**Entities to Track**:
- File paths (`.py`, `.js`, `.ts`, `.json`, `.md`)
- Function/class names
- Error messages
- Variable names (critical ones)

---

### 1.3 Enhanced Session Context
**Goal**: Track project-specific state

**Implementation**:
- [ ] Add `project_name` field to `SessionContext`
- [ ] Add `code_context` dict for function/class signatures
- [ ] Update session state on each request (not just background)
- [ ] Inject full entity summary into system prompt

**Session Context Structure**:
```python
{
    "user_intent": "refactor auth module",
    "current_task": "fix login bug",
    "project_name": "my-api-backend",
    "relevant_files": ["auth.py", "models/user.py"],
    "extracted_entities": {
        "file_path": [...],
        "function_name": [...],
        "error_msg": [...]
    },
    "key_decisions": ["use JWT instead of sessions"],
    "code_context": {
        "login_user": "async def login_user(username, password) -> Token"
    }
}
```

---

## Phase 2: LLM Summarization Pipeline

### 2.1 Background Summarization Enhancement
**Goal**: Compress dropped messages into concise summaries

**Current State**: Basic state extraction exists (`summarize_background_memory`)

**Enhancements Needed**:
- [ ] Add retry logic with `tenacity` library
- [ ] Summarize in chunks (if dropped > 5K tokens)
- [ ] Cache summaries to avoid re-summarizing same messages
- [ ] Fallback: keep original messages if summarization fails

**Summarization Prompt Template**:
```python
SUMMARIZATION_PROMPT = """
You are a context compression assistant. Summarize the following conversation 
into {max_tokens} tokens. Preserve:
- File names and code references
- User decisions and preferences
- Error messages and fixes
- Current task state

DO NOT preserve:
- Greetings, pleasantries
- Already-resolved questions

Messages:
{messages}

Summary:
"""
```

---

### 2.2 Incremental Summary Updates
**Goal**: Update summaries as conversation evolves

**Implementation**:
- [ ] Store `summary_version` in session state
- [ ] When new messages arrive, check if summary needs update
- [ ] Merge new relevant info into existing summary
- [ ] Keep summary < 200 tokens always

**Strategy**:
```
Every 5th dropped message batch:
  - Call LLM to merge new info into existing summary
  - Keep summary under 200 tokens
  - Update summary_version counter
```

---

## Phase 3: RAG Retrieval (Optional but Recommended)

### 3.1 Embedding-Based Retrieval
**Goal**: Semantic search instead of keyword matching

**Dependencies**:
```txt
sentence-transformers==2.3.1  # For embeddings
chromadb==0.4.22              # Lightweight vector DB
```

**Implementation**:
- [ ] Replace `retrieve_relevant_context()` with embedding-based search
- [ ] Create per-session ChromaDB collection
- [ ] Store embeddings of all messages in full_history
- [ ] Retrieve top-5 most similar messages (not just top-3)

**Code Change**:
```python
# Current: Keyword matching (app/main.py:117-140)
archived_context = retrieve_relevant_context(last_user_query, full_history, top_k=3)

# New: Semantic search
archived_context = await semantic_retrieve(session_id, last_user_query, top_k=5)
```

---

### 3.2 Hybrid Retrieval (Best Results)
**Goal**: Combine keyword + semantic search

**Implementation**:
- [ ] Run both keyword and embedding search
- [ ] Merge results (remove duplicates)
- [ ] Return top-7 most relevant messages
- [ ] Inject into system prompt as "RELEVANT PAST CONTEXT"

---

## Phase 4: Adaptive Budget Allocation

### 4.1 Dynamic Budget Profiles
**Goal**: Adjust context budget based on conversation type

**Budget Profiles**:
```python
BUDGET_PROFILES = {
    "tool_use": {
        "max_tokens": 16000,
        "history_budget": 4000,  # Tools need space
        "summary_budget": 1000,
        "tool_budget": 8000
    },
    "code_generation": {
        "max_tokens": 16000,
        "history_budget": 10000,  # Need more context
        "summary_budget": 1500,
        "tool_budget": 2000
    },
    "file_analysis": {
        "max_tokens": 16000,
        "history_budget": 8000,
        "summary_budget": 2000,
        "tool_budget": 4000
    }
}
```

**Detection Logic**:
```python
def detect_conversation_type(messages, data) -> str:
    has_tools = "tools" in data
    has_code = any("```" in str(m.get("content", "")) for m in messages[-5:])
    has_file_path = any(".py" in str(m.get("content", "")) for m in messages)
    
    if has_tools:
        return "tool_use"
    if has_file_path:
        return "file_analysis"
    if has_code:
        return "code_generation"
    return "complex_task"
```

---

## Phase 5: Testing & Validation

### 5.1 Memory Retention Test
**Goal**: Verify no critical info is lost

**Test Script**:
```python
# test_memory_retention.py
async def test_entity_preservation():
    """Test that file names, functions, errors are preserved after 50K tokens"""
    session_id = "test_retention"
    
    # Turn 1-10: Define project structure, create files
    for i in range(10):
        await send_message(session_id, f"Created file: src/module_{i}.py with function: func_{i}")
    
    # Turn 11-20: Add more context (exceed 16K budget)
    for i in range(10, 20):
        await send_message(session_id, f"Error in module_{i-10}: {random_error}")
    
    # Turn 21: Ask about early files
    response = await send_message(session_id, "What files were created in turns 1-5?")
    
    # Verify: response should mention module_0.py through module_4.py
    assert "module_0.py" in response
    assert "module_4.py" in response
```

### 5.2 Concurrent User Test
**Goal**: Verify 20 users don't cause memory loss

**Test Script**:
```python
# test_concurrent_20.py
async def test_20_concurrent_devs():
    """Simulate 20 developers, each working on different projects"""
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(20):
            session_id = f"dev_{i}"
            tasks.append(simulate_developer(client, session_id, project_name=f"project_{i}"))
        
        await asyncio.gather(*tasks)
    
    # Verify: each session has correct project context
    for i in range(20):
        session = load_session_data(f"dev_{i}")
        assert session["context"]["project_name"] == f"project_{i}"
```

---

## Implementation Order & Timeline

```
┌─────────────────────────────────────────────────────────┐
│ DAY 1-2: Core Compression (4h)                         │
├─────────────────────────────────────────────────────────┤
│ [ ] Finish entity extraction integration                │
│ [ ] Update session context with entities                │
│ [ ] Test scoring with real conversations                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DAY 3-4: Summarization (4h)                            │
├─────────────────────────────────────────────────────────┤
│ [ ] Add retry logic to summarization                    │
│ [ ] Implement chunked summarization                     │
│ [ ] Add summary caching                                 │
│ [ ] Test with 50K+ token conversations                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DAY 5-6: RAG Retrieval (4h)                            │
├─────────────────────────────────────────────────────────┤
│ [ ] Install sentence-transformers + chromadb            │
│ [ ] Implement semantic retrieval                        │
│ [ ] Create hybrid retrieval (keyword + embedding)       │
│ [ ] Benchmark retrieval accuracy                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DAY 7: Testing & Optimization (3h)                     │
├─────────────────────────────────────────────────────────┤
│ [ ] Run memory retention tests                          │
│ [ ] Run 20 concurrent user test                         │
│ [ ] Measure VRAM usage on 2x3090                        │
│ [ ] Tune budget profiles for file analysis workloads    │
└─────────────────────────────────────────────────────────┘
```

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Effective Context | 16K | 80K-100K | 5-6x |
| Critical Info Retention | 60% | 95%+ | +35% |
| File Name Retention | Lost after 16K | 100% | +100% |
| Error Message Retention | Lost | 100% | +100% |
| Conversation Length | ~15 turns | ~60 turns | 4x |

---

## Success Criteria

- [ ] 95% of high-value messages retained (files, functions, errors)
- [ ] Summarization reduces 1000 tokens → 100 tokens (10:1 ratio)
- [ ] <5% overhead from summarization calls (<500ms added latency)
- [ ] Zero information loss for file names, decisions, code references
- [ ] 20 concurrent users with <2s TTFT average
- [ ] Graceful fallback when summarization fails

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Summarization fails | Keep original messages, log error |
| Summary loses critical info | Entity extraction runs first, preserves entities separately |
| Added latency | Summarization timeout 30s, async, cached results |
| VRAM overflow on 2x3090 | Limit to 16K per request, use 4-bit quantization |
| RAG retrieval slow | Cache embeddings, use lightweight model (all-MiniLM-L6-v2) |

---

## Next Action Required

**Start with Phase 1 completion** (entity extraction integration). Should I proceed with the remaining implementation?
