# DevPilot

DevPilot is a learning-focused AI developer agent for understanding a local code repository. It is built with LangChain, LangGraph, OpenAI, and Chroma.

## Capabilities

### V1 — Tool-using agent

- LangGraph state and dynamic Agent Loop.
- `list_files()` safely lists `workspace/test_repo/`.
- `read_file(path)` safely reads a UTF-8 file inside that repository.
- Tool observations are appended to the LangGraph message state.

### V2 — Agentic RAG

- Repository files are loaded, chunked, embedded, and stored in local Chroma.
- `search_repository(query)` performs semantic repository search.
- The LLM decides when retrieval is useful; RAG is not mandatory preprocessing.

### V3 — Agent Memory

- Short-term memory: LangGraph `messages` state for one Agent run.
- Long-term memory: a separate persistent Chroma collection.
- The LLM can selectively call `save_memory(content, memory_type)` for durable user preferences, project facts, and architecture decisions.
- Relevant long-term memory is retrieved before the Agent node runs.

## Architecture

```text
User question
    ↓
retrieve_memory node
    ↓ relevant long-term memory context
agent node (LLM)
    ↓ conditional edge
    ├── no tool calls ───────────────────────────────→ END
    └── tool calls → ToolNode → agent node → ...
```

| Tool | Purpose |
|---|---|
| `list_files()` | Discover repository structure. |
| `read_file(path)` | Read exact source from a known file. |
| `search_repository(query)` | Semantically search indexed repository chunks. |
| `save_memory(content, memory_type)` | Persist concise, durable long-term memory. |

## RAG vs Memory

| Repository RAG | Long-term Memory |
|---|---|
| Searches repository code and documentation. | Retrieves durable conversation-derived facts. |
| Stored in `workspace/vectorstore/`. | Stored in `workspace/memory_store/`. |
| Rebuilt explicitly from `workspace/test_repo/`. | Saved selectively by the Agent. |
| Example: “Where is the entry point?” | Example: “The user prefers concise Chinese answers.” |

The stores use separate directories and Chroma collections. Repository code is not stored as long-term memory, and user preferences are not mixed into RAG.

## Project structure

```text
app/
├── agent/
│   ├── state.py           # messages and memory_context
│   ├── nodes.py           # memory retrieval and LLM nodes
│   └── graph.py           # LangGraph workflow and Agent Loop
├── memory/
│   └── store.py           # persistent long-term memory operations
├── rag/
│   ├── ingest.py          # repository loading, chunking, indexing
│   └── retriever.py       # Chroma repository retrieval
├── tools/
│   └── repository.py      # safe local repository tools
└── main.py                # CLI and debug trace output

workspace/
├── test_repo/             # repository DevPilot analyzes
├── vectorstore/           # generated V2 Chroma index (ignored)
└── memory_store/          # generated V3 memory store (ignored)

tests/
└── test_memory.py         # V3 memory-store tests
```

## Setup

Prerequisites: Python 3.11+ and an OpenAI API key.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit `.env`, `workspace/vectorstore/`, or `workspace/memory_store/`.

## Running DevPilot

```powershell
python -X utf8 -m app.main "What files are in this project?"
python -X utf8 -m app.main --debug "How is the greeting implemented?"
```

Debug output exposes the actual execution path:

```text
[0] Long-term memory retrieved
[1] User
[2] LLM tool decision
[3] Tool observation
[4] LLM final response generated
```

## Indexing and standalone RAG test

Build or rebuild the repository index before semantic search:

```powershell
python -X utf8 -m app.rag.ingest
python -X utf8 -m app.rag.retriever "Where is the greeting created?"
```

The ingestion command should print:

```text
Indexed 4 chunks in workspace/vectorstore.
```

For the sample repository, the retrieval result should include `src/greetings.py`.

## Test cases

### V1 — safe repository tools

```powershell
python -X utf8 -c "from app.tools.repository import list_files; print(list_files())"
python -X utf8 -c "from app.tools.repository import read_file; print(read_file('main.py'))"
python -X utf8 -c "from app.tools.repository import read_file; print(read_file('../../.env'))"
```

| Case | Expected result |
|---|---|
| List files | A readable tree rooted at `test_repo/`. |
| Read `main.py` | Returns the file content. |
| Path traversal | Returns `Access denied`; it never exposes `.env`. |

### V2 — Agentic RAG and tool routing

After indexing, use debug mode:

```powershell
python -X utf8 -m app.main --debug "Hello"
python -X utf8 -m app.main --debug "What files are in this project?"
python -X utf8 -m app.main --debug "Explain this specific file: main.py"
python -X utf8 -m app.main --debug "How is the greeting implemented? Search the repository semantically first."
```

| Query | Expected tool behavior |
|---|---|
| `Hello` | No repository tool. |
| `What files are in this project?` | `list_files`. |
| `Explain this specific file: main.py` | `read_file`. |
| Greeting implementation question | `search_repository`; it may additionally use `read_file`. |

### V3 — selective long-term memory

Run these as two separate CLI executions:

```powershell
python -X utf8 -m app.main --debug "Remember that I prefer concise Chinese answers."
python -X utf8 -m app.main --debug "Explain the agent loop."
```

| Case | Expected result |
|---|---|
| Explicit durable preference | Calls `save_memory` with `user_preference`. |
| New interaction | Debug step `[0]` shows the saved preference. |
| Casual `Hello` | Does not call `save_memory`. |
| Invalid memory type/content | Returns a clear `not saved` message. |

Run the V3 memory tests:

```powershell
pytest
```

`tests/test_memory.py` covers valid memory types, empty or oversized-memory rejection, persistence, global preference retrieval, semantic matching, deduplication, and memory-context formatting.

> **Important:** the current memory tests use `OpenAIEmbeddings`. They require `OPENAI_API_KEY`, may call the embedding API, and may incur API usage. Chroma data is written only to pytest temporary directories, not the project memory store.

## How V3 memory works

1. A new question enters graph state as a `HumanMessage`.
2. `retrieve_memory` loads global preferences and semantically relevant durable memories from the separate Memory Chroma collection.
3. The node writes a formatted context block to `AgentState.memory_context`.
4. The Agent node sends the system prompt, memory context, and short-term `messages` to the LLM.
5. The LLM may answer directly, use repository tools, or selectively call `save_memory`.
6. Tool results become `ToolMessage` observations, and the Agent Loop continues until the LLM makes no more tool calls.

## Interview summary

> DevPilot uses LangGraph state as short-term memory inside a single Agent run. For long-term memory, it stores only selected durable facts in a separate Chroma collection. Before every new question, a retrieval node injects only relevant memory as context. Repository RAG retrieves code evidence; Memory retrieves user preferences and project decisions. The LLM decides when a new fact is important enough to save.
