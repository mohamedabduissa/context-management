# LLM Orchestration System - ASCII Sequence Diagram

## Main Request Flow

```
Client      FastAPI     Orchestrator    BudgetManager    Memory    vLLM     ToolManager
   |             |              |               |          |        |           |
   |--POST /chat->|              |               |          |        |           |
   |             |-------------->|               |          |        |           |
   |             |              |--------------->|          |        |           |
   |             |              |                |<-------->|        |           |
   |             |              |                |context    |        |           |
   |             |              |                |<-------->|        |           |
   |             |              |                |history    |        |           |
   |             |              |                |           |        |           |
   |             |              |--------------->|           |        |           |
   |             |              |  build_payload |           |        |           |
   |             |              |                |           |        |           |
   |             |              |----------------|>          |        |           |
   |             |              |                |           |        |           |
   |             |              |                |<----------+        |           |
   |             |              |  messages      |           |        |           |
   |             |              |----------------|>          |        |           |
   |             |              |                |           |        |           |
   |             |              |--------------------------->|        |           |
   |             |              |                |           |        |           |
   |             |              |                |           |  stream |           |
   |             |              |                |           |<-------+           |
   |             |              |                |           |  chunks |           |
   |             |              |                |           |-------->|           |
   |             |              |                |           |         |           |
   |             |  stream      |                |           |         |           |
   |<------------|<-------------|                |           |         |           |
   |  content    |              |                |           |         |           |
   |             |              |                |           |         |           |
   |             |              |  tool_calls?   |           |         |           |
   |             |              |----------------|>          |         |           |
   |             |              |                |           |         |           |
   |             |              |--------------------------->|         |           |
   |             |              |                            |  result |           |
   |             |              |                |           |<--------+           |
   |             |              |                |           |         |           |
   |             |              |  loop back     |           |         |           |
   |             |              |<---------------|           |         |           |
   |             |              |                |           |         |           |
   |             |  final       |                |           |         |           |
   |<------------|<-------------|                |           |         |           |
   |  response   |              |                |           |         |           |
   |             |              |                |           |         |           |
   |             |              |  background:   |           |         |           |
   |             |              |  compress      |           |         |           |
   |             |              |--------------------------->|         |           |
   |             |              |                |           |         |           |
```

## Context Budget Building

```
+------------------+     +----------------+     +----------+     +--------+
|   System Prompt  |     | Session Context|     |  Tools   |     |   User |
|    (2000 tokens) |---->|  (800 tokens)  |---->| (4000)   |---->|  Query |
+------------------+     +----------------+     +----------+     +--------+
                                                                   |
                                                                   v
+------------------+     +----------------+     +----------+     +--------+
|                 |     |                |     |          |     |        |
|  MAX: 16000     |---->|  History       |---->|  Scorer  |---->|  Select|
|  tokens total   |     |  (7200 max)    |     |          |     | top K  |
+------------------+     +----------------+     +----------+     +--------+
                                                                   |
                                                                   v
                                                              +---------+
                                                              | Final   |
                                                              | Payload |
                                                              +---------+
```

## Tool Execution Loop

```
         +------------------+
         |  LLM Response    |
         |  with tool_calls |
         +--------+---------+
                  |
                  v
         +--------+--------+
         |  ToolManager    |
         |  execute()      |
         +--------+--------+
                  |
          +-------+-------+
          |               |
          v               v
   +------+------+   +------+------+
   |  Tool Found  |   |  Not Found  |
   |  Execute     |   |  Return Err |
   +------+------+   +------+------+
          |               |
          v               |
   +------+------+         |
   |  External   |         |
   |  Tool Call  |         |
   +------+------+         |
          |               |
          +-------+-------+
                  |
                  v
          +-------+-------+
          |  Check Dupes  |
          +-------+-------+
                  |
        +---------+---------+
        |                   |
        v                   v
  +----------+        +----------+
  | Duplicate|        |  New Call|
  |  Blocked |        |  Add to  |
  +----------+        |  queue   |
                      +----+-----+
                           |
                           +--------+
                                    |
                                    v
                          +---------+---------+
                          |  Loop back to LLM |
                          +-------------------+
```

## Streaming Response Format

```
Client receives SSE stream:

> data: {"type": "content", "content": "\n*💭 Thinking...*\n> "}
> data: {"type": "content", "content": "I need to list the directory\n> "}
> data: {"type": "content", "content": "before answering.\n> "}
> data: {"type": "content", "content": "\n\n"}
> data: {"type": "content", "content": "Calling list_directory..."}
> data: {"type": "content", "content": "\n\n"}
> data: {"type": "final", "content": "Here are the files: [...]",
>                 "tokens_before": 15000,
>                 "tokens_after": 8500,
>                 "compression_ratio": 0.57,
>                 "iterations": 2}
```

## OpenAI Proxy Flow

```
+--------+     +--------+     +--------+     +--------+
| Client |     | Proxy  |     | Budget |     |  vLLM  |
+---+----+     +----+---+     +----+---+     +----+---+
    |              |              |            |
    |  POST /chat  |              |            |
    +------------->|              |            |
    |              |              |            |
    |              |  Truncate    |            |
    |              +------------->|            |
    |              |  to 16000    |            |
    |              |              |            |
    |              |  Inject sys  |            |
    |              +------------->|            |
    |              |  prompt      |            |
    |              |              |            |
    |              |  Forward     |            |
    |              +------------->|            |
    |              |              |            |
    |              |  Stream      |<-----------+
    |              |  chunks      |            |
    |              |              |            |
    |              |  Transform   |            |
    |              |  reasoning-> |            |
    |              |  reasoning_  |            |
    |              |  content      |            |
    |              |              |            |
    |  SSE stream  |              |            |
    <--------------+              |            |
    |              |              |            |
```

## Component Architecture

```
+------------------------------------------------------------------+
|                        FastAPI Application                        |
+------------------------------------------------------------------+
|  +----------+    +----------+    +----------+    +----------+    |
|  |  /chat   |    | /v1/chat |    |  Memory  |    |  Tools   |    |
|  | Endpoint |    | Endpoint |    | Manager  |    | Registry |    |
|  +----+-----+    +----+-----+    +-----+----+    +----+-----+    |
|       |               |                |              |           |
|       +-------+-------+                |              |           |
|               |                        |              |           |
|  +------------v------------+           |              |           |
|  |      Orchestrator       |<----------+              |           |
|  +------------+------------+                    +-----v-----+     |
|               |                                      |           |
|  +------------v------------+           +------------v-----+     |
|  |    BudgetManager        |           |   ToolManager    |     |
|  +------------+------------+           +------------------+     |
|               |                                                  |
|  +------------v------------+           +------------------+     |
|  |      Scorer             |           |    vLLM Client   |     |
|  +-------------------------+           +------------------+     |
|                                                                  |
+------------------------------------------------------------------+
                             |
                             v
                    +--------+--------+
                    |      vLLM       |
                    |  (External API) |
                    +-----------------+
```
