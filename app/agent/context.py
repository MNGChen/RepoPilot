"""Small, deterministic context selection for DevPilot V4.

The graph keeps its complete message state for tool execution and debugging.
This module creates the smaller message list that is actually sent to the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


# Character limits are deliberately used in this learning project instead of a
# tokenizer dependency. They make the context trade-offs visible and predictable.
MAX_HISTORY_MESSAGES = 6
MAX_MEMORY_CONTEXT_CHARS = 2_000
MAX_LATEST_TOOL_OBSERVATION_CHARS = 6_000
MAX_OLDER_TOOL_OBSERVATION_CHARS = 1_500
TRUNCATION_MARKER = "\n... [truncated by Context Manager]"
RETRIEVAL_SECTION_PATTERN = re.compile(
    r"(?ms)^--- (?P<source>.+?) \(chunk [^)]+\) ---\n.*?(?=^--- .+? \(chunk [^)]+\) ---\n|\Z)"
)


def build_model_context(
    *,
    system_prompt: str,
    messages: Sequence[BaseMessage],
    memory_context: str = "",
) -> list[BaseMessage]:
    """Select a bounded, priority-ordered context for one model invocation.

    Priority is fixed: system instructions, relevant long-term memory, the
    current user request, then recent agent-loop history. Full graph state is
    never changed here.
    """
    selected_messages = _select_recent_messages(messages)
    model_context: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    if memory_context:
        model_context.append(
            SystemMessage(content=_truncate(memory_context, MAX_MEMORY_CONTEXT_CHARS))
        )

    model_context.extend(_trim_tool_observations(selected_messages))
    return model_context


def _select_recent_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Keep recent history while preserving valid tool-call/result pairs."""
    start_index = max(0, len(messages) - MAX_HISTORY_MESSAGES)
    start_index = _expand_to_tool_call_boundary(messages, start_index)
    recent_messages = list(messages[start_index:])
    latest_user_message = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )

    if latest_user_message is not None and latest_user_message not in recent_messages:
        # The latest request can precede several tool calls, so it may no longer
        # fit in the recent-message window. It still outranks old observations.
        recent_messages.insert(0, latest_user_message)

    return recent_messages


def _expand_to_tool_call_boundary(
    messages: Sequence[BaseMessage], start_index: int
) -> int:
    """Avoid beginning selected history with an orphaned tool result."""
    if start_index == 0 or not isinstance(messages[start_index], ToolMessage):
        return start_index

    required_tool_call_id = messages[start_index].tool_call_id
    for index in range(start_index - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage) and any(
            tool_call.get("id") == required_tool_call_id
            for tool_call in message.tool_calls
        ):
            # Preserve the entire result batch that follows this AI tool call.
            return index

    return start_index


def _trim_tool_observations(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Prioritize recent evidence and remove superseded RAG source chunks."""
    latest_tool_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, ToolMessage)),
        default=-1,
    )
    processed_contents: dict[int, str] = {}
    later_read_sources: set[str] = set()

    # Work backwards so each search result can be compared with files read
    # later in the agent loop. The ToolMessages themselves remain present,
    # preserving the protocol required by the tool-calling API.
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ToolMessage):
            continue

        content = str(message.content)
        if message.name == "read_file":
            source = _read_file_source(content)
            if source:
                later_read_sources.add(source)
        elif message.name == "search_repository":
            content = _remove_superseded_retrieval_sections(content, later_read_sources)

        maximum_characters = (
            MAX_LATEST_TOOL_OBSERVATION_CHARS
            if index == latest_tool_index
            else MAX_OLDER_TOOL_OBSERVATION_CHARS
        )
        processed_contents[index] = _truncate(content, maximum_characters)

    trimmed_messages: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            trimmed_messages.append(
                ToolMessage(
                    content=processed_contents[index],
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    additional_kwargs=message.additional_kwargs,
                    response_metadata=message.response_metadata,
                )
            )
        else:
            trimmed_messages.append(message)
    return trimmed_messages


def _read_file_source(content: str) -> str | None:
    """Extract the path from DevPilot's ``read_file`` result format."""
    first_line, _, _ = content.partition("\n")
    if first_line.startswith("--- ") and first_line.endswith(" ---"):
        return first_line.removeprefix("--- ").removesuffix(" ---")
    return None


def _remove_superseded_retrieval_sections(content: str, read_sources: set[str]) -> str:
    """Drop RAG chunks when a later ``read_file`` supplied that file in full."""
    if not read_sources:
        return content

    sections = list(RETRIEVAL_SECTION_PATTERN.finditer(content))
    if not sections:
        return content

    retained_sections = [
        match.group(0).rstrip()
        for match in sections
        if match.group("source") not in read_sources
    ]
    if not retained_sections:
        return "[Repository search results omitted: their source files were read in full later.]"

    return "\n\n".join(retained_sections)


def _truncate(content: str, maximum_characters: int) -> str:
    """Return a clearly marked prefix when context must be shortened."""
    if len(content) <= maximum_characters:
        return content

    prefix_length = maximum_characters - len(TRUNCATION_MARKER)
    return f"{content[:prefix_length]}{TRUNCATION_MARKER}"
