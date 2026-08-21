"""Adapt discovered GitHub MCP tools for DevPilot's existing LangGraph loop."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool, tool

from app.mcp.github_client import GitHubMCPClient, parse_github_url


LOGGER = logging.getLogger(__name__)
GITHUB_FILE_TOOL_NAME = "get_file_contents"
GITHUB_TREE_TOOL_NAME = "get_repository_tree"
GITHUB_SEARCH_TOOL_NAME = "search_code"
ALLOWED_GITHUB_MCP_TOOLS = {
    GITHUB_FILE_TOOL_NAME,
    GITHUB_TREE_TOOL_NAME,
    GITHUB_SEARCH_TOOL_NAME,
}


def load_github_mcp_tools() -> list[BaseTool]:
    """Discover the safe GitHub capability and return its LangChain adapter.

    Discovery is deliberately an allow-list boundary: DevPilot learns what the
    external server offers but exposes only the one read-only capability needed
    for V5. If Docker, the token, or GitHub are unavailable, existing V1-V4
    tools continue to work without the GitHub addition.
    """
    try:
        discovered_names = {
            discovered_tool.name
            for discovered_tool in asyncio.run(GitHubMCPClient().discover_tools())
        }
    except Exception as error:  # The optional remote integration must not break V1-V4.
        LOGGER.warning("GitHub MCP tools are unavailable: %s", error)
        return []

    if GITHUB_FILE_TOOL_NAME not in discovered_names:
        LOGGER.warning(
            "GitHub MCP did not advertise %s; skipping GitHub integration.",
            GITHUB_FILE_TOOL_NAME,
        )
        return []

    unavailable_optional_tools = (
        ALLOWED_GITHUB_MCP_TOOLS - discovered_names - {GITHUB_FILE_TOOL_NAME}
    )
    if unavailable_optional_tools:
        LOGGER.info(
            "GitHub MCP optional V5 tools unavailable: %s.",
            ", ".join(sorted(unavailable_optional_tools)),
        )

    @tool("github_read_file_url")
    def github_read_file_url(url: str) -> str:
        """Read a GitHub repository file from a pasted github.com URL.

        Use this only when the user provides a repository URL or a file URL,
        such as https://github.com/owner/repository/blob/main/README.md. This
        tool is read-only and retrieves source through GitHub's MCP server.
        """
        try:
            location = parse_github_url(url)
        except ValueError as error:
            return f"Invalid GitHub URL: {error}"

        tool_arguments = {
            "owner": location.owner,
            "repo": location.repository,
            "path": location.path,
        }
        if location.ref:
            tool_arguments["ref"] = location.ref

        try:
            result = asyncio.run(
                GitHubMCPClient().call_tool(GITHUB_FILE_TOOL_NAME, tool_arguments)
            )
        except Exception as error:
            return f"GitHub MCP connection failed: {error}"

        if result.is_error:
            return f"GitHub MCP tool error: {result.text}"
        return result.text

    @tool("github_list_repository_url")
    def github_list_repository_url(url: str) -> str:
        """List a GitHub repository's top-level contents from its pasted URL.

        Use this before reading remote files when the user asks about an
        unfamiliar or large GitHub repository's structure. This tool is
        read-only and does not download or clone the repository.
        """
        try:
            location = parse_github_url(url)
        except ValueError as error:
            return f"Invalid GitHub URL: {error}"

        if GITHUB_TREE_TOOL_NAME in discovered_names:
            tool_name = GITHUB_TREE_TOOL_NAME
            tool_arguments = {
                "owner": location.owner,
                "repo": location.repository,
                "recursive": False,
            }
        else:
            # Older GitHub MCP servers may omit get_repository_tree. Their
            # file-content tool still lists a directory when given an empty
            # path, which is enough for this V5 top-level overview.
            tool_name = GITHUB_FILE_TOOL_NAME
            tool_arguments = {
                "owner": location.owner,
                "repo": location.repository,
                "path": "",
            }

        try:
            result = asyncio.run(GitHubMCPClient().call_tool(tool_name, tool_arguments))
        except Exception as error:
            return f"GitHub MCP connection failed: {error}"

        if result.is_error:
            return f"GitHub MCP tool error: {result.text}"
        return result.text

    tools: list[BaseTool] = [github_read_file_url, github_list_repository_url]

    if GITHUB_SEARCH_TOOL_NAME in discovered_names:

        @tool("github_search_repository_url")
        def github_search_repository_url(url: str, query: str) -> str:
            """Search code in one GitHub repository from its pasted URL.

            Use this for conceptual questions about a large remote repository.
            Provide a concise code or architecture query. The search is scoped
            to the repository in the URL and is read-only.
            """
            try:
                location = parse_github_url(url)
            except ValueError as error:
                return f"Invalid GitHub URL: {error}"

            try:
                result = asyncio.run(
                    GitHubMCPClient().call_tool(
                        GITHUB_SEARCH_TOOL_NAME,
                        {
                            "query": f"{query} repo:{location.owner}/{location.repository}",
                            "perPage": 10,
                        },
                    )
                )
            except Exception as error:
                return f"GitHub MCP connection failed: {error}"

            if result.is_error:
                return f"GitHub MCP tool error: {result.text}"
            return result.text

        tools.append(github_search_repository_url)

    return tools
