# DevPilot

DevPilot is a small, learning-focused AI developer agent. It will help developers inspect and understand a local code repository using an LLM, tools, and a LangGraph workflow.

## Step 1: Project setup

This first step creates the project foundation only. The agent, LangGraph workflow, and repository tools are deliberately not implemented yet.

```
devpilot/
├── app/
│   ├── agent/              # LangGraph state, nodes, and graph (later)
│   └── tools/              # Repository tools (later)
├── workspace/
│   └── test_repo/          # A harmless local repository DevPilot will inspect
├── .env.example            # API-key template; never commit the real .env file
├── .gitignore
├── requirements.txt
└── README.md
```

The current workspace directory is the `devpilot/` project root. `workspace/test_repo/` is intentionally a separate, small Python project: later, DevPilot will be restricted to reading only this directory.

## Prerequisites

- Python 3.11 or later
- An OpenAI API key (needed only when we connect the LLM in a later step)

## Setup and verification

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -c "import langchain, langgraph, dotenv; print('DevPilot environment is ready.')"
```

Add your API key to `.env` after copying it. Do not commit `.env`.

To verify the example repository independently:

```powershell
python workspace/test_repo/main.py
```

Expected output:

```text
Hello, DevPilot!
```

## Why the example repository exists

DevPilot needs a predictable local repository to inspect while we learn. It has a simple entry point and one reusable module, so later questions such as “Where is the entry point?” or “Explain this file” have concrete answers.

## Step 2: Repository tools

`app/tools/repository.py` contains two small, read-only Python functions:

- `list_files()` returns a tree of `workspace/test_repo/`.
- `read_file(path)` reads one UTF-8 text file from that repository.

The functions are deliberately not connected to an LLM yet. Keeping the tool
logic separate makes it easy to test and lets a later LangGraph node use the
same functions.

### Security boundary

The target repository is fixed to `workspace/test_repo/`. `read_file` rejects
empty and absolute paths, resolves relative paths, and verifies that the final
path remains inside the target directory. This blocks `../../.env` and also a
symlink that points outside the repository. Files larger than 100 KB are not
read, which prevents oversized tool responses from consuming LLM context.

### Test the tools

After activating your virtual environment, run:

```powershell
python -X utf8 -c "from app.tools.repository import list_files; print(list_files())"
python -X utf8 -c "from app.tools.repository import read_file; print(read_file('main.py'))"
python -X utf8 -c "from app.tools.repository import read_file; print(read_file('../../.env'))"
```

The third command should return an `Access denied` message, not the contents
of `.env`.

## Next step

## Step 3: LangGraph agent loop

The graph is now deliberately small and dynamic:

```text
START → agent → conditional edge ── no tool calls ──→ END
                       │
                       └── tool calls → tools → agent
```

- `AgentState` in `app/agent/state.py` stores `messages`. Its `add_messages`
  reducer appends messages instead of replacing the history.
- The `agent` node invokes `ChatOpenAI` with DevPilot's system prompt and the
  complete message history. The model has the two repository functions bound
  as tool schemas.
- The conditional edge examines the latest `AIMessage`. Tool calls route to
  `ToolNode`; otherwise the AI message is the final answer and the graph ends.
- `ToolNode` executes every requested tool and appends `ToolMessage`
  observations to state. The agent node then receives those observations on
  its next turn, so it can request more information or answer.

This is an **agent loop** because the graph can repeat `agent → tools → agent`
as often as the model needs. It is not a hard-coded, one-tool workflow.

### Run DevPilot

Put a valid `OPENAI_API_KEY` in `.env`, then run:

```powershell
python -X utf8 -m app.main "Analyze the structure of this project."
python -X utf8 -m app.main "Where is the entry point?"
python -X utf8 -m app.main "Explain src/greetings.py"
```

Expected behavior: the model selects `list_files` and/or `read_file`, receives
the results as observations, then gives an evidence-based final answer. The
exact wording and number of calls are model decisions.

## Next step

After you have run these commands successfully, we can improve the command
line experience or inspect the graph execution in detail.
