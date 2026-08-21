"""The LLM node and tools available to the DevPilot graph."""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.agent.context import build_model_context
from app.agent.state import AgentState
from app.memory.store import format_memory_context, retrieve_relevant_memories, save_memory
from app.mcp.langchain_tools import load_github_mcp_tools
from app.rag.retriever import search_repository
from app.tools.repository import list_files, read_file


SYSTEM_PROMPT = """You are DevPilot, an AI agent that helps developers understand
and analyze code repositories.

You have access to local tools: list_files, read_file, search_repository, and
save_memory. When GitHub MCP is configured, you may also have
github_read_file_url, github_list_repository_url, and optionally
github_search_repository_url.

Use list_files when the user asks about project structure or you need to
discover files. Use read_file when the user names a specific file or you need
its complete, exact contents. Use search_repository for conceptual questions
about how an implementation works or when semantic search can find relevant
files efficiently. You can combine tools: use search results to identify files,
then read_file when you need complete source context.

Use github_read_file_url when the user gives a GitHub repository or file URL
and asks about its contents. Do not use it for a local repository path; use
the local repository tools for those requests.

For a large or unfamiliar GitHub repository, use github_list_repository_url
first for its top-level structure. Use github_search_repository_url for a
conceptual question to find likely files, then github_read_file_url for the
exact file evidence. Do not attempt to read an entire remote repository.

Use save_memory only for concise, durable information that will help future
conversations: explicit user preferences, confirmed project facts, or
architecture decisions. Do not save ordinary questions, temporary details,
raw repository content, raw tool results, secrets, API keys, or every turn.
Do not mention that a memory was saved unless the user asks.

Do not assume or invent repository contents. Do not use repository tools for
casual conversation. If the available information is insufficient, use another
tool. Base final answers only on the conversation and actual tool results."""

LOCAL_TOOLS = [tool(list_files), tool(read_file), tool(search_repository), tool(save_memory)]
TOOLS = [*LOCAL_TOOLS, *load_github_mcp_tools()]


def _build_model() -> ChatOpenAI:
    """Create the OpenAI chat model and give it DevPilot's tool schemas."""
    load_dotenv()
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    model = ChatOpenAI(model=model_name, temperature=0)
    return model.bind_tools(TOOLS)


MODEL = _build_model()


def agent_node(state: AgentState) -> dict[str, list]:
    """Ask the model for either a tool call or a final response."""
    model_context = build_model_context(
        system_prompt=SYSTEM_PROMPT,
        memory_context=state.get("memory_context", ""),
        messages=state["messages"],
    )
    response = MODEL.invoke(model_context)
    return {"messages": [response], "model_context": model_context}


def retrieve_memory_node(state: AgentState) -> dict[str, str]:
    """Find long-term memories relevant to the newest user question."""
    latest_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if latest_user_message is None:
        return {"memory_context": ""}

    memories = retrieve_relevant_memories(str(latest_user_message.content))
    return {"memory_context": format_memory_context(memories)}
