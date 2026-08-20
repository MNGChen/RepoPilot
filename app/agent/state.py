"""The shared data that moves through the DevPilot graph."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation history, including user, model, and tool messages."""

    messages: Annotated[list[AnyMessage], add_messages]
    memory_context: str
    model_context: list[AnyMessage]
