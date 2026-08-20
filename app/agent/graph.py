"""LangGraph workflow for DevPilot's tool-using agent loop."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.nodes import TOOLS, agent_node, retrieve_memory_node
from app.agent.state import AgentState


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to tools only when the latest model response requests them."""
    latest_message = state["messages"][-1]
    if getattr(latest_message, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    """Build and compile the dynamic agent → tools → agent workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")

    return builder.compile()


graph = build_graph()
