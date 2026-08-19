"""The LLM node and tools available to the DevPilot graph."""

import os

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.agent.state import AgentState
from app.tools.repository import list_files, read_file


SYSTEM_PROMPT = """You are DevPilot, an AI agent that helps developers understand
and analyze code repositories.

You have access to two tools: list_files and read_file. Use a tool whenever
repository information is needed. Do not assume or invent repository contents.
If the available information is insufficient, use another tool. Base your final
answer only on the conversation and actual tool results."""

TOOLS = [tool(list_files), tool(read_file)]


def _build_model() -> ChatOpenAI:
    """Create the OpenAI chat model and give it DevPilot's tool schemas."""
    load_dotenv()
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    model = ChatOpenAI(model=model_name, temperature=0)
    return model.bind_tools(TOOLS)


MODEL = _build_model()


def agent_node(state: AgentState) -> dict[str, list]:
    """Ask the model for either a tool call or a final response."""
    response = MODEL.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}
