"""Browser interface for asking DevPilot about the configured repository."""

from __future__ import annotations

import asyncio
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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
        result = await asyncio.to_thread(
            graph.invoke, {"messages": [HumanMessage(content=question)]}
        )
        answer = str(result["messages"][-1].content)
    except Exception as error:
        return JSONResponse(
            {"error": f"Analysis could not be completed: {error}"}, status_code=500
        )

    return JSONResponse({"answer": answer})


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


app = Starlette(
    debug=False,
    routes=[
        Route("/", home),
        Route("/api/analyze", analyze, methods=["POST"]),
        Route("/api/capabilities", capabilities),
    ],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
