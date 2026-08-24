"""Browser interface for asking DevPilot about the configured repository."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from app.agent.graph import graph
from app.agent.nodes import TOOLS
from app.mcp.github_client import parse_github_url


BASE_DIR = Path(__file__).resolve().parent
GITHUB_URL_PATTERN = re.compile(r"https://github\.com/[^\s<>()]+", re.IGNORECASE)
MAX_SHORT_TERM_MESSAGES = 12
SESSION_IDLE_SECONDS = 60 * 60 * 2


@dataclass
class ConversationSession:
    """In-memory, browser-session-scoped conversation state."""

    messages: list[BaseMessage] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


SESSIONS: dict[str, ConversationSession] = {}
SESSIONS_LOCK = asyncio.Lock()


async def home(_: Request) -> HTMLResponse:
    """Serve the single-page DevPilot chat interface."""
    return HTMLResponse((BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


async def analyze(request: Request) -> JSONResponse:
    """Run a repository question without blocking Starlette's event loop."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request format."}, status_code=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Enter a question about the repository."}, status_code=400)
    if len(question) > 4_000:
        return JSONResponse({"error": "Questions cannot exceed 4,000 characters."}, status_code=400)

    session_id = _validated_session_id(payload.get("sessionId"))
    session = await _get_session(session_id)

    github_url = _find_github_url(question)
    if github_url:
        try:
            parse_github_url(github_url)
        except ValueError as error:
            return JSONResponse({"error": f"Unsupported GitHub URL: {error}"}, status_code=400)

        if not _github_tools_available():
            return JSONResponse(
                {
                    "error": (
                        "GitHub analysis is not ready. Configure "
                        "GITHUB_PERSONAL_ACCESS_TOKEN in .env, start Docker Desktop, "
                        "then restart the web server."
                    )
                },
                status_code=503,
            )

        if question == github_url:
            question = (
                "Analyze this GitHub repository or file. Explain its purpose, "
                f"structure, and important implementation details: {github_url}"
            )

    load_dotenv()
    try:
        async with session.lock:
            turn_messages = [*session.messages, HumanMessage(content=question)]
            result = await asyncio.to_thread(graph.invoke, {"messages": turn_messages})
            session.messages = _short_term_messages(result["messages"])
            session.last_active = time.monotonic()
            answer = str(result["messages"][-1].content)
    except Exception as error:
        return JSONResponse(
            {"error": f"Analysis could not be completed: {error}"}, status_code=500
        )

    response: dict[str, object] = {"answer": answer, "sessionId": session_id}
    if bool(payload.get("debug", False)):
        response["behaviorTree"] = _build_behavior_tree(result)
    return JSONResponse(response)


async def capabilities(_: Request) -> JSONResponse:
    """Expose optional integration availability for the browser interface."""
    return JSONResponse({"githubAnalysisAvailable": _github_tools_available()})


def _find_github_url(text: str) -> str | None:
    """Return the first GitHub URL in a question, excluding trailing punctuation."""
    match = GITHUB_URL_PATTERN.search(text)
    return match.group(0).rstrip(".,!?;:") if match else None


def _github_tools_available() -> bool:
    """Check whether GitHub MCP discovery succeeded during application startup."""
    return any(tool.name.startswith("github_") for tool in TOOLS)


def _validated_session_id(candidate: object) -> str:
    """Accept a browser-generated ID or create a new safe session ID."""
    session_id = str(candidate or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{16,80}", session_id):
        return session_id
    return uuid4().hex


async def _get_session(session_id: str) -> ConversationSession:
    """Retrieve a session and remove inactive conversation state."""
    async with SESSIONS_LOCK:
        now = time.monotonic()
        stale_session_ids = [
            key for key, value in SESSIONS.items() if now - value.last_active > SESSION_IDLE_SECONDS
        ]
        for key in stale_session_ids:
            del SESSIONS[key]

        return SESSIONS.setdefault(session_id, ConversationSession())


def _short_term_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep recent user/final-answer pairs, excluding internal tool traffic."""
    conversation = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls and message.content)
    ]
    return conversation[-MAX_SHORT_TERM_MESSAGES:]


def _build_behavior_tree(result: dict) -> dict[str, object]:
    """Create a safe, compact execution tree from a completed graph run."""
    children: list[dict[str, object]] = [
        {
            "label": "Retrieve memory",
            "detail": (
                "Relevant long-term context found."
                if result.get("memory_context")
                else "No relevant long-term context found."
            ),
            "status": "success",
        }
    ]
    tool_calls: dict[str, dict[str, object]] = {}
    decision_number = 0

    messages = result["messages"]
    latest_user_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=0,
    )
    for message in messages[latest_user_index:]:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                decision_number += 1
                requested_tools = []
                for call in message.tool_calls:
                    tool_node = {
                        "label": call["name"],
                        "detail": _format_tool_arguments(call.get("args", {})),
                        "status": "running",
                    }
                    tool_calls[call["id"]] = tool_node
                    requested_tools.append(tool_node)
                children.append(
                    {
                        "label": f"Agent decision {decision_number}",
                        "detail": "Requested repository evidence.",
                        "status": "success",
                        "children": requested_tools,
                    }
                )
            elif message.content:
                children.append(
                    {
                        "label": "Agent response",
                        "detail": "Generated the final answer from the available evidence.",
                        "status": "success",
                    }
                )
        elif isinstance(message, ToolMessage):
            tool_node = tool_calls.get(message.tool_call_id)
            if tool_node is not None:
                tool_node["status"] = (
                    "error" if _tool_result_is_error(str(message.content)) else "success"
                )
                tool_node["result"] = _summarize_tool_result(str(message.content))

    return {"label": "DevPilot behavior tree", "status": "success", "children": children}


def _format_tool_arguments(arguments: object) -> str:
    """Display tool arguments compactly without dumping long user input."""
    text = json.dumps(arguments, ensure_ascii=False, default=str)
    return text if len(text) <= 180 else f"{text[:177]}..."


def _summarize_tool_result(content: str) -> str:
    """Report result metadata only, keeping debug output safe and readable."""
    return f"Returned {len(content):,} characters."


def _tool_result_is_error(content: str) -> bool:
    return content.lower().startswith(("error", "github mcp tool error", "github mcp connection failed"))


app = Starlette(
    debug=False,
    routes=[
        Route("/", home),
        Route("/api/analyze", analyze, methods=["POST"]),
        Route("/api/capabilities", capabilities),
    ],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
