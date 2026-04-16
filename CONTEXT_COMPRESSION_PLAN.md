# Context Compression Enhancement Plan

**Current Rating:** 6/10  
**Target Rating:** 9/10  
**Estimated Effort:** 15 hours  
**Expected Gain:** 3-4x effective context window (48k-64k vs 16k)

---

## Phase 1: Smart Message Prioritization (5h)

### 1.1 Message Scoring Algorithm (2h)

**Impact:** 8/10 | **Effort:** 2h

**Objective:** Score messages by value instead of simple recency.

**Scoring Formula:**
```
Message Score = (recency_weight * 0.4) + (role_weight * 0.3) + (content_weight * 0.3)
```

**Scoring Weights:**
- User message: +30 points
- Assistant message: +10 points
- System message: +5 points
- Contains tool_call: +25 points
- Contains code/files: +20 points
- Each turn older (from latest): -5 points
- Contains error/failure: +15 points

**Implementation:**
```python
def calculate_message_score(message: Dict, position_from_end: int) -> float:
    score = 0.0
    
    # Role-based scoring
    role_scores = {"user": 30, "assistant": 10, "system": 5}
    score += role_scores.get(message.get("role", ""), 0)
    
    # Content-based scoring
    content = str(message.get("content", ""))
    if message.get("tool_calls"):
        score += 25
    if any(ext in content for ext in [".py", ".js", ".ts", "```"]):
        score += 20
    if any(word in content.lower() for word in ["error", "fail", "issue"]):
        score += 15
    
    # Recency penalty (older = lower score)
    score -= position_from_end * 5
    
    return max(0, score)
```

---

### 1.2 Key Entity Extraction (3h)

**Impact:** 9/10 | **Effort:** 3h

**Objective:** Preserve critical information even when messages are old.

**Entities to Extract:**
- File names and paths
- Function/class names
- Key decisions (user confirmations)
- Code snippets (critical logic only)
- Error messages

**Implementation:**
```python
import re

ENTITY_PATTERNS = {
    "file_path": r'["\']?([a-zA-Z0-9_./-]+\.py|\.js|\.ts|\.json|\.md)["\']?',
    "function_name": r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    "class_name": r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    "decision": r'(?:decided|confirmed|agreed|yes|no)\s*(?:to)?\s*(.+?)(?:\.|$)',
}

def extract_entities(content: str) -> Dict[str, List[str]]:
    entities = {}
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            entities[entity_type] = list(set(matches))
    return entities

def create_entity_summary(entities: Dict) -> str:
    """Create compact entity preservation string"""
    parts = []
    if "file_path" in entities:
        parts.append(f"Files: {', '.join(entities['file_path'][:5])}")
    if "function_name" in entities:
        parts.append(f"Functions: {', '.join(entities['function_name'][:5])}")
    if "decision" in entities:
        parts.append(f"Decisions: {entities['decision'][0][:100]}")
    return " | ".join(parts) if parts else ""
```

---

## Phase 2: LLM Summarization Pipeline (7h)

### 2.1 Sliding Window Summarization (4h)

**Impact:** 9/10 | **Effort:** 4h

**Objective:** Compress old messages into concise summaries using vLLM.

**Process Flow:**
```
When 16k budget exceeded:
  1. Sort messages by score (lowest first)
  2. Select bottom 20% of messages for summarization
  3. Call vLLM with summarization prompt
  4. Replace selected messages with summary
  5. Free up ~4000 tokens per summarization cycle
```

**Summarization Prompt Template:**
```python
SUMMARIZATION_PROMPT = """
You are a context compression assistant. Summarize the following conversation turns 
into exactly {max_tokens} tokens maximum. Preserve:
- Key decisions and user preferences
- File names and code references
- Important constraints or requirements
- Current task state

DO NOT preserve:
- Greetings or pleasantries
- Minor clarifications
- Already-resolved questions

Conversation to summarize:
{messages}

Summary (max {max_tokens} tokens):
"""
```

**Implementation:**
```python
async def summarize_message_batch(
    messages: List[Dict], 
    max_summary_tokens: int = 100
) -> Optional[str]:
    """Call vLLM to summarize a batch of messages"""
    
    if not messages:
        return None
    
    prompt = SUMMARIZATION_PROMPT.format(
        max_tokens=max_summary_tokens,
        messages=json.dumps(messages)
    )
    
    try:
        resp = await async_http_client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_summary_tokens,
                "temperature": 0.3
            },
            timeout=30.0
        )
        
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
    
    return None
```

---

### 2.2 Incremental Summary Updates (3h)

**Impact:** 8/10 | **Effort:** 3h

**Objective:** Update summaries as conversation evolves.

**Strategy:**
- Store summary version number
- When new context added, check if summary needs update
- Merge new relevant info into existing summary
- Keep summary < 200 tokens always

**Implementation:**
```python
class ContextCompressor:
    def __init__(self, max_tokens: int = 16000):
        self.max_tokens = max_tokens
        self.current_summary: Optional[str] = None
        self.summary_version = 0
        self.compressed_history: List[Dict] = []
    
    async def compress_context(
        self, 
        messages: List[Dict],
        tool_tokens: int = 0
    ) -> List[Dict]:
        """Main compression entry point"""
        
        # Step 1: Score all messages
        scored_messages = [
            (msg, calculate_message_score(msg, i))
            for i, msg in enumerate(reversed(messages))
        ]
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Step 2: Calculate current token usage
        current_tokens = tool_tokens
        kept_messages = []
        
        # Step 3: Keep high-score messages until budget filled
        for msg, score in scored_messages:
            msg_tokens = get_token_count(str(msg.get("content", "")))
            if current_tokens + msg_tokens < self.max_tokens * 0.8:  # 80% threshold
                kept_messages.append(msg)
                current_tokens += msg_tokens
        
        # Step 4: Summarize remaining messages if needed
        discarded = [m for m, _ in scored_messages if m not in kept_messages]
        if discarded and len(discarded) >= 3:
            summary = await summarize_message_batch(discarded)
            if summary:
                self.current_summary = summary
                self.summary_version += 1
        
        # Step 5: Build final message list
        result = []
        if self.current_summary:
            result.append({
                "role": "system",
                "content": f"Conversation Summary (v{self.summary_version}):\n{self.current_summary}"
            })
        result.extend(kept_messages)
        
        return result
```

---

## Phase 3: Adaptive Budget Allocation (3h)

### 3.1 Dynamic Budget Adjustment (2h)

**Impact:** 8/10 | **Effort:** 2h

**Objective:** Adjust context budget based on conversation type.

**Budget Profiles:**
```python
BUDGET_PROFILES = {
    "tool_use": {
        "max_tokens": 16000,
        "history_budget": 4000,      # Tools need space
        "summary_budget": 1000,
        "tool_budget": 8000
    },
    "code_generation": {
        "max_tokens": 16000,
        "history_budget": 10000,     # Need more context
        "summary_budget": 1500,
        "tool_budget": 2000
    },
    "simple_qa": {
        "max_tokens": 8000,
        "history_budget": 6000,
        "summary_budget": 500,
        "tool_budget": 0
    },
    "complex_task": {
        "max_tokens": 16000,
        "history_budget": 8000,
        "summary_budget": 2000,
        "tool_budget": 4000
    }
}
```

**Detection Logic:**
```python
def detect_conversation_type(messages: List[Dict], data: Dict) -> str:
    """Auto-detect conversation type for budget selection"""
    
    has_tools = "tools" in data
    has_code = any("```" in str(m.get("content", "")) for m in messages[-5:])
    message_count = len(messages)
    
    if has_tools:
        return "tool_use"
    if has_code and message_count > 3:
        return "code_generation"
    if message_count <= 2 and not has_tools:
        return "simple_qa"
    return "complex_task"
```

---

### 3.2 Tool-aware Compression (1h)

**Impact:** 7/10 | **Effort:** 1h

**Objective:** Prioritize tool-related messages over chat history.

**Strategy:**
- Tool call messages: +50 score bonus
- Tool response messages: +30 score bonus
- Tool schema messages: Always keep (never compress)
- When tools present, reduce history budget by 40%

**Implementation:**
```python
def is_tool_related(message: Dict) -> bool:
    """Check if message is tool-related"""
    return bool(
        message.get("tool_calls") or 
        message.get("tool_call_id") or
        "tool" in str(message.get("content", "")).lower()
    )

def apply_tool_aware_scoring(messages: List[Dict]) -> List[tuple]:
    """Apply tool-aware scoring to messages"""
    scored = []
    for i, msg in enumerate(reversed(messages)):
        base_score = calculate_message_score(msg, i)
        if is_tool_related(msg):
            base_score += 50 if msg.get("tool_calls") else 30
        scored.append((msg, base_score))
    return scored
```

---

## Implementation Order & Timeline

```
┌─────────────────────────────────────────────────────────┐
│ Week 1: Core Compression (9h)                          │
├─────────────────────────────────────────────────────────┤
│ Day 1-2: Message Scoring Algorithm (2h)                │
│ Day 2-3: Entity Extraction (3h)                        │
│ Day 4-5: LLM Summarization (4h)                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Week 2: Optimization (6h)                              │
├─────────────────────────────────────────────────────────┤
│ Day 1-2: Incremental Summary Updates (3h)              │
│ Day 3: Dynamic Budget Adjustment (2h)                  │
│ Day 3-4: Tool-aware Compression (1h)                   │
└─────────────────────────────────────────────────────────┘
```

---

## New Dependencies

```txt
tenacity==8.2.3              # Retry logic for summarization calls
```

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Effective Context | 16k | 48k-64k | 3-4x |
| Critical Info Retention | 60% | 95% | +35% |
| Token Efficiency | 1.0x | 2.5x | +150% |
| Conversation Length | ~15 turns | ~45 turns | 3x |

---

## Success Criteria

- [ ] 80% of high-value messages retained (vs 60% current)
- [ ] Summarization reduces 1000 tokens → 100 tokens (10:1 ratio)
- [ ] <5% overhead from summarization calls (<500ms added latency)
- [ ] No information loss for file names, decisions, code references
- [ ] Graceful fallback when summarization fails

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Summarization fails | Keep original messages, log error |
| Summary loses critical info | Entity extraction runs first, preserves entities separately |
| Added latency | Summarization timeout 30s, async, cached results |
| Summary quality poor | Low temperature (0.3), explicit prompt constraints |

---

**Document Created:** 2025-04-15  
**Last Updated:** 2025-04-15  
**Status:** Planning Phase
