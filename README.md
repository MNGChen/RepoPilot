# DevPilot

DevPilot is a learning-focused AI developer agent for understanding a local code repository. It is built with LangChain, LangGraph, OpenAI, and Chroma.

## V5 — GitHub MCP (Steps 1–2: discovery and read-only calls)

DevPilot will act as an MCP client for GitHub's official MCP Server. The first
step connects to the server locally through Docker and discovers the tools it
advertises; LangGraph is not changed yet.

The initial server configuration is deliberately small and safe:

- GitHub's official `ghcr.io/github/github-mcp-server` image.
- `repos` toolset only.
- `GITHUB_READ_ONLY=1`, so write tools are not exposed.
- A fine-grained GitHub personal access token loaded from the local `.env`.

Create a fine-grained PAT with the minimum read-only access needed for the
repositories you want DevPilot to analyze, then add it to `.env`:

```text
GITHUB_PERSONAL_ACCESS_TOKEN=github_pat_...
```

Never commit the token. Docker Desktop must be running before the command
below. The first run downloads GitHub's public MCP image.

Discover the server's actual tool names, descriptions, and JSON input schemas:

```powershell
python -X utf8 -m app.mcp.github_client
```

This command makes DevPilot the MCP client. It launches the GitHub MCP server
as a temporary child container over standard input/output; there is no separate
server terminal to keep open for this stdio integration.

After discovery succeeds, test a real read-only MCP tool call. This reads a
public file from GitHub through the discovered `get_file_contents` tool:

```powershell
python -X utf8 -m app.mcp.github_client --read-file octocat Hello-World README
```

Use a repository that your fine-grained token is allowed to access when testing
private repositories. The command first verifies that the server advertises
`get_file_contents`, then sends its `owner`, `repo`, and `path` arguments over
MCP. It prints the result returned by the server and makes no GitHub changes.

For the more convenient URL form, paste a repository or file link directly:

```powershell
python -X utf8 -m app.mcp.github_client --read-url https://github.com/octocat/Hello-World/blob/master/README
```

V5 then adds the same MCP capability to the existing LangGraph ToolNode as
`github_read_file_url(url)`. Keep Docker Desktop running and give DevPilot a
GitHub URL in its question:

```powershell
python -X utf8 -m app.main --debug "Explain this file: https://github.com/octocat/Hello-World/blob/master/README"
```

DevPilot discovers the GitHub server's tools at startup and explicitly allows
only its read-only repository-analysis capabilities. The Agent-facing URL
adapter keeps the LLM interface concise while GitHub MCP receives its native
`owner`, `repo`, `path`, and optional `ref` arguments.

For large remote repositories, V5 also explicitly allows two more read-only
GitHub MCP tools: `get_repository_tree` and `search_code`. DevPilot exposes
them as URL-first tools, so it can locate evidence before reading files:

```powershell
python -X utf8 -m app.mcp.github_client --list-url https://github.com/octocat/Hello-World
python -X utf8 -m app.mcp.github_client --search-url https://github.com/octocat/Hello-World "README"
python -X utf8 -m app.main --debug "这个大型项目如何组织？https://github.com/owner/repository"
```

The intended remote workflow is: list the top-level tree, search code in that
repository when the server advertises `search_code`, then read only the few
relevant files. If a GitHub MCP Server does not advertise
`get_repository_tree`, DevPilot uses `get_file_contents(path="")` to list the
root instead. DevPilot does not clone, download, or build a Chroma index for
the remote repository in V5.

## V5 architecture summary

```text
User supplies a GitHub URL
    ↓
DevPilot Agent (LangGraph controls the Agent → Tool → Agent loop)
    ↓
GitHub URL adapter LangChain tool
    ↓
DevPilot MCP Client discovers/calls approved tools over stdio
    ↓
GitHub official MCP Server in Docker (read-only)
    ↓
GitHub API result
    ↓
Tool result managed by V4 context budgets
    ↓
Agent final answer
```

Run with `--debug` to see the discovered GitHub tools, MCP tool decisions, and
MCP observations separately from DevPilot's local tools.

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

### V4 — Context Engineering

- A Context Manager constructs the information sent to the LLM at every Agent
  step; the complete LangGraph message state is still retained separately.
- Context priority is: system prompt, relevant long-term memory, current user
  request, then recent agent-loop history.
- History keeps six recent messages while preserving tool-call/result pairs.
- Memory context is capped at 2,000 characters. The latest tool observation is
  capped at 6,000 characters and older observations at 1,500 characters.
- Repository search returns at most four chunks, each capped at 1,500
  characters. RAG chunks are removed from the LLM view when their source file
  is later read in full.

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
│   ├── state.py           # messages, memory context, and selected model context
│   ├── context.py         # V4 context selection and budgets
│   ├── context_demo.py    # visible V4 manual demonstration
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

V4 adds a compact section before that trace for every LLM call. It shows the
selected message types and character counts, so you can see exactly which
context sources entered the model input without printing entire source files.

Run the standalone V4 demonstration without pytest:

```powershell
python -m app.agent.context_demo
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
