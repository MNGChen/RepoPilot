"""Command-line entry point for DevPilot."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agent.graph import graph


def main() -> None:
    """Run one repository question through the DevPilot graph."""
    load_dotenv()
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print('Usage: python -m app.main "Analyze the structure of this project."')
        return

    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    final_message = result["messages"][-1]
    print(final_message.content)


if __name__ == "__main__":
    main()
