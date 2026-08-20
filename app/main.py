"""Command-line entry point for DevPilot."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.graph import graph


MAX_DEBUG_CONTENT_LENGTH = 1_500


def main() -> None:
    """Run one repository question through the DevPilot graph."""
    load_dotenv()
    arguments = sys.argv[1:]
    debug_enabled = "--debug" in arguments
    question = " ".join(argument for argument in arguments if argument != "--debug").strip()

    if not question:
        print('Usage: python -m app.main [--debug] "Analyze the structure of this project."')
        return

    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    if debug_enabled:
        _print_debug_trace(result["messages"], result.get("memory_context", ""))

    final_message = result["messages"][-1]
    print("\nFinal answer:\n")
    print(final_message.content)


def _print_debug_trace(messages: list[BaseMessage], memory_context: str) -> None:
    """Display the message history that shows DevPilot's agent loop."""
    print("\n=== DevPilot debug trace ===")

    if memory_context:
        print(f"\n[0] Long-term memory retrieved\n{memory_context}")
    else:
        print("\n[0] Long-term memory retrieved\nNo relevant memory.")

    for step, message in enumerate(messages, start=1):
        if isinstance(message, HumanMessage):
            print(f"\n[{step}] User\n{message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            print(f"\n[{step}] LLM tool decision")
            for tool_call in message.tool_calls:
                print(f"  → {tool_call['name']}({tool_call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"\n[{step}] Tool observation: {message.name}")
            print(_shorten_debug_content(message.content))
        elif isinstance(message, AIMessage):
            print(f"\n[{step}] LLM final response generated")

    print("\n=== End debug trace ===")


def _shorten_debug_content(content: object) -> str:
    """Keep verbose tool output readable while preserving the beginning."""
    text = str(content)
    if len(text) <= MAX_DEBUG_CONTENT_LENGTH:
        return text

    return f"{text[:MAX_DEBUG_CONTENT_LENGTH]}\n... [debug output truncated]"


if __name__ == "__main__":
    main()
