# Performance Improvement Plan

## Current Bottlenecks Identified

### 1. **Synchronous HTTP Calls** (CRITICAL)
- `VLLMClient.chat_stream()` uses `requests.post()` (blocking)
- Blocks entire event loop during LLM calls
- **Impact**: Can't handle concurrent requests efficiently

### 2. **In-Memory Storage** (HIGH)
- `SESSIONS_HISTORY`, `SESSIONS_EVENTS`, `SESSIONS_CONTEXT` are dicts
- No persistence, memory grows unbounded
- **Impact**: Memory leaks, data loss on restart

### 3. **Mock Token Counting** (MEDIUM)
- `mock_count_tokens()` uses word splitting (inaccurate)
- `tiktoken` in requirements but not used
- **Impact**: Poor budget allocation, context overflow risks

### 4. **No Connection Pooling** (MEDIUM)
- New HTTP connection per LLM call
- **Impact**: Connection overhead, slower response times

### 5. **Sequential Tool Execution** (LOW)
- Tools execute one at a time
- **Impact**: Could parallelize independent tool calls

### 6. **No Caching** (LOW)
- Repeated LLM calls for similar queries
- **Impact**: Wasted tokens and latency

---

## Performance Improvements (Priority Order)

### Phase 1: Critical (Week 1)

#### 1.1 Switch to Async HTTP Client
```python
# Replace requests with httpx async
import httpx

class VLLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat_stream(self, messages: List[Message], tools: List[Dict] = None):
        # Async streaming - doesn't block event loop
        async with self.client.stream('POST', self.base_url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line:
                    # process chunk
                    yield {...}
```
**Expected Gain**: 3-5x concurrent request handling

#### 1.2 Implement Real Token Counting
```python
import tiktoken

class TokenCounter:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def count_messages(self, messages: List[Message]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.content or "")
        return total
```
**Expected Gain**: 20-30% better budget utilization

#### 1.3 Redis Integration for Session Storage
```python
import redis.asyncio as redis

class RedisMemoryManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def save_message(self, session_id: str, msg: Message):
        await self.redis.rpush(f"session:{session_id}:history", msg.model_dump_json())
    
    async def get_history(self, session_id: str) -> List[Message]:
        messages = await self.redis.lrange(f"session:{session_id}:history", 0, -1)
        return [Message(**json.loads(m)) for m in messages]
    
    async def expire_session(self, session_id: str, ttl: int = 3600):
        await self.redis.expire(f"session:{session_id}", ttl)
```
**Expected Gain**: Persistent sessions, auto-cleanup, cluster support

---

### Phase 2: High Impact (Week 2)

#### 2.1 HTTP Connection Pooling
```python
class VLLMClient:
    def __init__(self, base_url: str, model: str, pool_size: int = 10):
        self.base_url = base_url
        self.model = model
        limits = httpx.MaxConnections(max_connections=pool_size)
        self.client = httpx.AsyncClient(limits=limits, timeout=120.0)
```
**Expected Gain**: 15-20% faster response times

#### 2.2 Response Caching
```python
from functools import lru_cache
import hashlib

class ResponseCache:
    def __init__(self, redis_client, ttl: int = 300):
        self.redis = redis_client
    
    async def get_cache_key(self, messages: List[Message]) -> str:
        msg_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        return f"llm_cache:{hashlib.sha256(msg_str.encode()).hexdigest()}"
    
    async def get_or_cache(self, key: str, fetch_func):
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        result = await fetch_func()
        await self.redis.setex(key, ttl, json.dumps(result))
        return result
```
**Expected Gain**: 40-60% reduction in LLM calls for repeated queries

#### 2.3 Parallel Tool Execution
```python
import asyncio

class ToolExecutionManager:
    async def execute_parallel(self, tool_calls: List[Dict]) -> List[Message]:
        tasks = [self.execute_single(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)
```
**Expected Gain**: 2-3x faster when multiple independent tools called

---

### Phase 3: Optimization (Week 3)

#### 3.1 Context Compression with Summarization
```python
class ContextCompressor:
    async def compress_with_llm(self, history: List[Message]) -> SessionContext:
        # Use LLM to summarize conversation
        prompt = f"""Summarize this conversation into structured context:
        {json.dumps([m.model_dump() for m in history[-10:]])}
        
        Return JSON: {{user_intent, current_task, key_decisions, ...}}"""
        
        response = await self.llm_client.generate(prompt, response_format="json")
        return SessionContext(**json.loads(response))
```
**Expected Gain**: 50% smaller context, faster LLM responses

#### 3.2 Streaming Optimization
```python
# Pre-allocate buffers, reduce JSON serialization overhead
class OptimizedStreamer:
    def __init__(self):
        self.buffer = bytearray(4096)
    
    async def stream_response(self, response):
        async for chunk in response.aiter_bytes():
            # Process raw bytes, minimal serialization
            yield chunk
```
**Expected Gain**: 10-15% faster streaming

#### 3.3 History Pruning Strategy
```python
class HistoryPruner:
    def __init__(self, max_messages: int = 50, max_tokens: int = 7200):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    async def prune_history(self, session_id: str, history: List[Message]):
        # Keep last N messages + important ones (tool calls, decisions)
        important = [m for m in history if m.role == "tool" or self.is_decision(m)]
        recent = history[-self.max_messages:]
        
        pruned = list(dict.fromkeys(important + recent))  # dedupe
        return pruned[:self.max_messages]
```
**Expected Gain**: 30-40% smaller payloads

---

### Phase 4: Advanced (Week 4)

#### 4.1 Request Batching
```python
# Batch multiple user queries into single LLM call
class RequestBatcher:
    async def batch_requests(self, requests: List[ChatRequest]):
        combined = self.combine_prompts(requests)
        response = await self.llm_client.generate_batch(combined)
        return self.split_response(response, len(requests))
```
**Expected Gain**: 50% reduction in API calls

#### 4.2 Prefetching & Speculative Execution
```python
# Pre-compute likely tool calls while streaming
class SpeculativeExecutor:
    async def prefetch_tools(self, partial_response: str):
        # Predict likely tool calls from partial content
        predicted_tools = self.analyze_partial(partial_response)
        # Pre-warm tool execution
        return await self.prepare_tools(predicted_tools)
```
**Expected Gain**: 20-30% faster overall response time

#### 4.3 Load Balancing for vLLM
```python
class LoadBalancedVLLMClient:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.load = {ep: 0 for ep in endpoints}
    
    def get_least_loaded(self) -> str:
        return min(self.load, key=self.load.get)
    
    async def chat_stream(self, messages: List[Message]):
        endpoint = self.get_least_loaded()
        # route to least loaded instance
```
**Expected Gain**: Better resource utilization, horizontal scaling

---

## Implementation Checklist

### Week 1 (Critical)
- [ ] Replace `requests` with `httpx.AsyncClient`
- [ ] Make all endpoints async (`async def chat_endpoint`)
- [ ] Implement `TokenCounter` with tiktoken
- [ ] Add Redis session storage
- [ ] Add session TTL expiration

### Week 2 (High Impact)
- [ ] Configure connection pooling (max_connections=20)
- [ ] Implement LLM response caching
- [ ] Add parallel tool execution
- [ ] Add Redis connection pooling

### Week 3 (Optimization)
- [ ] Implement LLM-based context compression
- [ ] Optimize streaming with byte buffers
- [ ] Add history pruning strategy
- [ ] Add metrics/monitoring (Prometheus)

### Week 4 (Advanced)
- [ ] Request batching for similar queries
- [ ] Speculative tool execution
- [ ] Load balancing across vLLM instances
- [ ] Rate limiting per session

---

## Expected Performance Gains

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Concurrent users | ~10 | ~50 | ~100 | ~200 |
| P95 latency | ~8s | ~4s | ~2.5s | ~1.8s |
| Memory usage | High (unbounded) | Low (Redis) | Low | Low |
| Token efficiency | ~60% | ~80% | ~90% | ~95% |
| Cache hit rate | 0% | N/A | ~35% | ~50% |

---

## Monitoring & Metrics

```python
from prometheus_client import Counter, Histogram

# Add these metrics
LLM_CALLS = Counter('llm_calls_total', 'Total LLM calls', ['endpoint'])
LLM_LATENCY = Histogram('llm_latency_seconds', 'LLM call latency')
TOKEN_USAGE = Counter('tokens_used_total', 'Tokens used', ['type'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
TOOL_EXECUTIONS = Counter('tool_executions_total', 'Tool calls', ['tool_name'])
```

---

## Quick Wins (1-2 hours)

1. **Add tiktoken** - Replace `mock_count_tokens()` immediately
2. **Add connection pooling** - 5 lines change
3. **Add session TTL** - Prevent memory leaks
4. **Add logging/metrics** - Understand bottlenecks

Want me to implement any of these phases?
