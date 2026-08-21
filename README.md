# DevPilot

> An AI agent for understanding local repositories and read-only GitHub projects.

DevPilot helps developers get oriented in unfamiliar codebases. Ask questions in natural language—such as where a feature is implemented, how a project is organized, or what a file is responsible for—and the agent retrieves relevant repository evidence before responding.

It supports local repository analysis with semantic search, durable memory for useful project context, and optional read-only GitHub analysis through GitHub's official MCP Server.

## What it can do

- Explore a local repository and read specific files safely.
- Find relevant code with semantic search when the file path is unknown.
- Explain code using retrieved repository evidence.
- Save and retrieve durable user preferences and project decisions separately from code search data.
- Inspect GitHub repository and file URLs without cloning the remote repository.
- List remote directory trees and search remote code when those capabilities are available from GitHub MCP.

## Key design choices

- **Agent-driven retrieval**: the model decides whether to answer directly, read a file, or search the repository.
- **Evidence-first answers**: repository questions are grounded in retrieved source material whenever tools are needed.
- **Separated storage**: Chroma collections for code retrieval and long-term memory are kept apart.
- **Bounded context**: memory, conversation history, and remote tool output are limited before they reach the model.
- **Read-only remote access**: GitHub MCP is configured to expose only repository-analysis capabilities.

## Quick start

### Prerequisites

- Python 3.11 or later
- An OpenAI API key

### Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit `.env` or generated data under `workspace/`.

### Run a local analysis

Copy or clone the repository you want to analyze into `workspace/test_repo/`. Its contents are intentionally ignored by Git, so your local source code will not be included when you push DevPilot.

```text
workspace/
└── test_repo/             # place the repository files to analyze here
```

Build the search index after adding the files:

```powershell
python -X utf8 -m app.rag.ingest
python -X utf8 -m app.main --debug "How is the greeting implemented?"
```

Use `--debug` to inspect the agent's tool decisions, retrieved memory, and the context selected for each model call.

## Usage examples

```powershell
# Inspect the local repository structure
python -X utf8 -m app.main "What files are in this project?"

# Explain a known file
python -X utf8 -m app.main --debug "Explain this specific file: main.py"

# Search for an implementation by intent
python -X utf8 -m app.main --debug "How is the greeting implemented? Search the repository first."

# Persist a durable preference
python -X utf8 -m app.main --debug "Remember that I prefer concise answers."
```

## GitHub MCP (optional)

DevPilot can inspect public or authorized private GitHub repositories through the official `ghcr.io/github/github-mcp-server` Docker image. The integration uses standard input/output and creates a temporary child container; no separate server terminal is required.

Create a fine-grained GitHub personal access token with the minimum read-only repository permissions needed, then add it to `.env`:

```text
GITHUB_PERSONAL_ACCESS_TOKEN=github_pat_...
```

Docker Desktop must be running. The first call downloads the public GitHub MCP image.

```powershell
# Discover the MCP tools available from the server
python -X utf8 -m app.mcp.github_client

# Read a remote file
python -X utf8 -m app.mcp.github_client --read-url https://github.com/octocat/Hello-World/blob/master/README

# List a repository tree or search remote code
python -X utf8 -m app.mcp.github_client --list-url https://github.com/octocat/Hello-World
python -X utf8 -m app.mcp.github_client --search-url https://github.com/octocat/Hello-World "README"

# Ask the agent to analyze a remote file
python -X utf8 -m app.main --debug "Explain this file: https://github.com/octocat/Hello-World/blob/master/README"
```

DevPilot permits only the read-only repository tools it recognizes. Remote results are bounded before entering agent state: 3,000 characters for a tree, 4,500 for code search, and 6,000 for file content. The normal model-context budget is applied afterward.

## Architecture

```text
User question
    ↓
Relevant long-term memory retrieval
    ↓
Agent node (LLM)
    ↓
    ├── no tool calls ──→ final answer
    └── tool calls ─────→ local tools or GitHub MCP tools ──→ Agent node
```

| Tool | Purpose |
|---|---|
| `list_files()` | Discover the local repository structure. |
| `read_file(path)` | Read a UTF-8 file inside the configured local repository. |
| `search_repository(query)` | Search indexed local repository chunks semantically. |
| `save_memory(content, memory_type)` | Persist a concise, durable memory. |
| `github_read_file_url(url)` | Read a GitHub file URL through the read-only MCP adapter. |
| `github_list_repository_url(url)` | List a GitHub repository tree when supported. |
| `github_search_repository_url(url, query)` | Search a GitHub repository when supported. |

## Project structure

```text
app/
├── agent/
│   ├── state.py           # graph state and selected model context
│   ├── context.py         # context selection and budgets
│   ├── context_demo.py    # standalone context demonstration
│   ├── nodes.py           # memory retrieval and LLM nodes
│   └── graph.py           # LangGraph workflow and agent loop
├── mcp/
│   ├── github_client.py   # GitHub MCP stdio client and CLI utilities
│   └── langchain_tools.py # approved GitHub URL-based LangChain tools
├── memory/
│   └── store.py           # persistent long-term memory operations
├── rag/
│   ├── ingest.py          # repository loading, chunking, and indexing
│   └── retriever.py       # Chroma repository retrieval
├── tools/
│   └── repository.py      # safe local repository tools
└── main.py                # CLI entry point and debug output

workspace/
└── test_repo/             # place the local repository to analyze here

```

Generated Chroma data is stored under `workspace/vectorstore/` and `workspace/memory_store/`; both are ignored by Git.

## Development notes

This project was built as a hands-on exploration of agent workflows, RAG, durable memory, context engineering, and MCP integration.
