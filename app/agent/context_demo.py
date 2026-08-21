"""A visible demonstration of DevPilot context selection."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.context import build_model_context


def main() -> None:
    """Print the context selected for a representative tool-using agent loop."""
    original_search_output = """--- app/auth.py (chunk 0) ---
Authentication search result.

--- app/config.py (chunk 1) ---
Configuration search result."""
    original_file_output = "--- app/auth.py ---\n" + "authentication source line\n" * 500
    messages = [
        HumanMessage(content="How is authentication implemented?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_repository", "args": {"query": "authentication"}, "id": "1"}
            ],
        ),
        ToolMessage(
            content=original_search_output,
            tool_call_id="1",
            name="search_repository",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "app/auth.py"}, "id": "2"}
            ],
        ),
        ToolMessage(content=original_file_output, tool_call_id="2", name="read_file"),
    ]
    selected_context = build_model_context(
        system_prompt="You are DevPilot.",
        memory_context="Relevant long-term memory:\n- User prefers concise answers.\n" * 100,
        messages=messages,
    )

    print("Original RAG output:", len(original_search_output), "characters")
    print("Original file output:", len(original_file_output), "characters")
    print("\nSelected model context:")
    for index, message in enumerate(selected_context):
        content = str(message.content)
        preview = content[:100].replace("\n", " ")
        print(f"[{index}] {type(message).__name__}: {len(content)} characters")
        print(f"    {preview}{'...' if len(content) > 100 else ''}")


if __name__ == "__main__":
    main()
