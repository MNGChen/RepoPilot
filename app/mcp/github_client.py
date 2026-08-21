"""Minimal MCP client for GitHub's official read-only MCP server.

This module owns the MCP client lifecycle: configure a server process,
discover its tools, and call a tool by its discovered name. A separate adapter
exposes the approved capability to DevPilot's LangGraph workflow.
"""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


GITHUB_MCP_IMAGE = "ghcr.io/github/github-mcp-server"
GITHUB_TOOLSETS = "repos"


@dataclass(frozen=True)
class DiscoveredMCPTool:
    """The model-relevant definition returned by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPToolResult:
    """A tool result in a form that DevPilot can later give to LangGraph."""

    text: str
    is_error: bool


@dataclass(frozen=True)
class GitHubFileLocation:
    """MCP arguments derived from a standard GitHub repository/file URL."""

    owner: str
    repository: str
    path: str
    ref: str | None = None


def parse_github_url(url: str) -> GitHubFileLocation:
    """Convert a GitHub repository or simple ``blob`` URL into MCP arguments.

    Supported examples:
    ``https://github.com/owner/repository``
    ``https://github.com/owner/repository/blob/main/README.md``

    GitHub URLs whose branch name itself contains ``/`` are intentionally out
    of scope for this first ergonomic adapter; callers can use the explicit
    owner/repository/path command for those rare cases.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("Provide an https://github.com/owner/repository URL.")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include both an owner and repository name.")

    owner, repository = parts[:2]
    if repository.endswith(".git"):
        repository = repository.removesuffix(".git")
    if len(parts) == 2:
        return GitHubFileLocation(owner=owner, repository=repository, path="")

    if len(parts) < 5 or parts[2] != "blob":
        raise ValueError(
            "Use a repository URL or a file URL containing /blob/<branch>/<path>."
        )

    return GitHubFileLocation(
        owner=owner,
        repository=repository,
        ref=parts[3],
        path="/".join(parts[4:]),
    )


def build_github_server_parameters() -> StdioServerParameters:
    """Configure the official GitHub MCP server as a local Docker subprocess.

    The server receives a fine-grained PAT only through the subprocess
    environment. ``GITHUB_READ_ONLY`` prevents GitHub write tools from being
    exposed even if the token has broader permissions.
    """
    load_dotenv()
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    if not token or token == "your_github_personal_access_token_here":
        raise RuntimeError(
            "GITHUB_PERSONAL_ACCESS_TOKEN is not configured. Add a real "
            "fine-grained, read-only token to .env before using GitHub MCP."
        )

    process_environment = os.environ.copy()
    process_environment.update(
        {
            "GITHUB_PERSONAL_ACCESS_TOKEN": token,
            "GITHUB_TOOLSETS": GITHUB_TOOLSETS,
            "GITHUB_READ_ONLY": "1",
        }
    )

    return StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e",
            "GITHUB_TOOLSETS",
            "-e",
            "GITHUB_READ_ONLY",
            GITHUB_MCP_IMAGE,
        ],
        env=process_environment,
    )


class GitHubMCPClient:
    """Connect to GitHub MCP over stdio and discover the permitted tools."""

    # load server parameters
    def __init__(self, server_parameters: StdioServerParameters | None = None) -> None:
        self._server_parameters = server_parameters or build_github_server_parameters()

    async def discover_tools(self) -> list[DiscoveredMCPTool]:
        """Start the server, negotiate MCP, and return its tool definitions."""
        # The installed v2 SDK expects a transport context manager. GitHub's
        # current MCP server closes stdio when it receives v2's initial
        # ``server/discover`` probe, so force the compatible classic handshake
        # instead of relying on automatic protocol negotiation.
        async with Client(
            stdio_client(self._server_parameters), mode="legacy"
        ) as client:
            result = await client.list_tools()

        return [
            DiscoveredMCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Call one advertised GitHub MCP tool with JSON-compatible arguments."""
        async with Client(
            stdio_client(self._server_parameters), mode="legacy"
        ) as client:
            result = await client.call_tool(name, arguments)

        text_blocks = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        fallback_content = json.dumps(
            [block.model_dump(mode="json") for block in result.content],
            ensure_ascii=False,
        )
        return MCPToolResult(
            text="\n".join(text_blocks) if text_blocks else fallback_content,
            is_error=result.is_error,
        )




# Command-line demonstration of the GitHub MCP client.
# used for local testing
async def _discover_and_print() -> None:
    """Run the smallest visible V5 client demonstration."""
    tools = await GitHubMCPClient().discover_tools()
    print("GitHub MCP tools discovered (read-only repos toolset):")
    for tool in tools:
        print(f"\n- {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Input schema: {json.dumps(tool.input_schema, ensure_ascii=False)}")


async def _read_file_and_print(owner: str, repository: str, path: str) -> None:
    """Show discovery followed by one safe, read-only GitHub MCP tool call."""
    client = GitHubMCPClient()
    discovered_names = {tool.name for tool in await client.discover_tools()}
    if "get_file_contents" not in discovered_names:
        raise RuntimeError(
            "GitHub MCP did not advertise get_file_contents. "
            "Check the configured repos toolset and read-only server logs."
        )

    result = await client.call_tool(
        "get_file_contents",
        {"owner": owner, "repo": repository, "path": path},
    )
    status = "error" if result.is_error else "success"
    print(f"MCP call get_file_contents: {status}\n")
    print(result.text)


async def _read_url_and_print(url: str) -> None:
    """Read a GitHub URL so a user need not split it into three arguments."""
    location = parse_github_url(url)
    client = GitHubMCPClient()
    discovered_names = {tool.name for tool in await client.discover_tools()}
    if "get_file_contents" not in discovered_names:
        raise RuntimeError(
            "GitHub MCP did not advertise get_file_contents. "
            "Check the configured repos toolset and read-only server logs."
        )

    arguments: dict[str, str] = {
        "owner": location.owner,
        "repo": location.repository,
        "path": location.path,
    }
    if location.ref:
        arguments["ref"] = location.ref
    result = await client.call_tool("get_file_contents", arguments)
    status = "error" if result.is_error else "success"
    print(f"MCP call get_file_contents: {status}\n")
    print(result.text)


async def _list_repository_and_print(url: str) -> None:
    """List a remote repository tree through the discovered GitHub MCP tool."""
    location = parse_github_url(url)
    client = GitHubMCPClient()
    discovered_names = {tool.name for tool in await client.discover_tools()}
    if "get_repository_tree" not in discovered_names:
        raise RuntimeError("GitHub MCP did not advertise get_repository_tree.")

    result = await client.call_tool(
        "get_repository_tree",
        {"owner": location.owner, "repo": location.repository, "recursive": False},
    )
    status = "error" if result.is_error else "success"
    print(f"MCP call get_repository_tree: {status}\n")
    print(result.text)


async def _search_url_and_print(url: str, query: str) -> None:
    """Search code in one remote repository through GitHub MCP."""
    location = parse_github_url(url)
    client = GitHubMCPClient()
    discovered_names = {tool.name for tool in await client.discover_tools()}
    if "search_code" not in discovered_names:
        raise RuntimeError("GitHub MCP did not advertise search_code.")

    result = await client.call_tool(
        "search_code",
        {"query": f"{query} repo:{location.owner}/{location.repository}", "perPage": 10},
    )
    status = "error" if result.is_error else "success"
    print(f"MCP call search_code: {status}\n")
    print(result.text)


def main() -> None:
    """Run MCP discovery without involving DevPilot's LangGraph agent."""
    parser = argparse.ArgumentParser(description="Test DevPilot's GitHub MCP client.")
    parser.add_argument(
        "--read-file",
        nargs=3,
        metavar=("OWNER", "REPOSITORY", "PATH"),
        help="Discover tools, then read one GitHub repository file through MCP.",
    )
    parser.add_argument(
        "--read-url",
        metavar="GITHUB_URL",
        help="Discover tools, then read a GitHub repository/file URL through MCP.",
    )
    parser.add_argument(
        "--list-url",
        metavar="GITHUB_URL",
        help="Discover tools, then list the top-level remote repository tree through MCP.",
    )
    parser.add_argument(
        "--search-url",
        nargs=2,
        metavar=("GITHUB_URL", "QUERY"),
        help="Discover tools, then search code in one GitHub repository through MCP.",
    )
    arguments = parser.parse_args()

    try:
        if arguments.read_file:
            asyncio.run(_read_file_and_print(*arguments.read_file))
        elif arguments.read_url:
            asyncio.run(_read_url_and_print(arguments.read_url))
        elif arguments.list_url:
            asyncio.run(_list_repository_and_print(arguments.list_url))
        elif arguments.search_url:
            asyncio.run(_search_url_and_print(*arguments.search_url))
        else:
            asyncio.run(_discover_and_print())
    except RuntimeError as error:
        print(f"GitHub MCP configuration error: {error}")


if __name__ == "__main__":
    main()
