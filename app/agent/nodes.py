"""The LLM node and tools available to the DevPilot graph."""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.agent.state import AgentState
from app.memory.store import format_memory_context, retrieve_relevant_memories, save_memory
from app.rag.retriever import search_repository
from app.tools.repository import list_files, read_file


SYSTEM_PROMPT = """You are DevPilot, an AI agent that helps developers understand
and analyze code repositories.

You have access to four tools: list_files, read_file, search_repository, and
save_memory.

Use list_files when the user asks about project structure or you need to
discover files. Use read_file when the user names a specific file or you need
its complete, exact contents. Use search_repository for conceptual questions
about how an implementation works or when semantic search can find relevant
files efficiently. You can combine tools: use search results to identify files,
then read_file when you need complete source context.

Use save_memory only for concise, durable information that will help future
conversations: explicit user preferences, confirmed project facts, or
architecture decisions. Do not save ordinary questions, temporary details,
raw repository content, raw tool results, secrets, API keys, or every turn.
Do not mention that a memory was saved unless the user asks.

Do not assume or invent repository contents. Do not use repository tools for
casual conversation. If the available information is insufficient, use another
tool. Base final answers only on the conversation and actual tool results."""

TOOLS = [tool(list_files), tool(read_file), tool(search_repository), tool(save_memory)]


def _build_model() -> ChatOpenAI:
    """Create the OpenAI chat model and give it DevPilot's tool schemas."""
    load_dotenv()
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    model = ChatOpenAI(model=model_name, temperature=0)
    return model.bind_tools(TOOLS)


MODEL = _build_model()


def agent_node(state: AgentState) -> dict[str, list]:
    """Ask the model for either a tool call or a final response."""
    system_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if state.get("memory_context"):
        system_messages.append(SystemMessage(content=state["memory_context"]))

    response = MODEL.invoke([*system_messages, *state["messages"]])
    return {"messages": [response]}


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
