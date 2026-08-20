"""Shared Chroma and embedding configuration for DevPilot V2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.tools.repository import PROJECT_ROOT


VECTORSTORE_DIRECTORY: Final = PROJECT_ROOT / "workspace" / "vectorstore"
COLLECTION_NAME: Final = "devpilot_repository"
DEFAULT_RESULT_COUNT: Final = 4
MAX_RETRIEVED_CHUNK_CHARS: Final = 1_500
CHUNK_TRUNCATION_MARKER: Final = "\n... [repository chunk truncated]"


def create_embeddings() -> OpenAIEmbeddings:
    """Create the OpenAI embedding client from environment configuration."""
    load_dotenv()
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name)


def get_vector_store(embeddings: OpenAIEmbeddings | None = None) -> Chroma:
    """Open DevPilot's persistent local Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or create_embeddings(),
        persist_directory=str(VECTORSTORE_DIRECTORY),
    )


def retrieve_repository_chunks(
    query: str,
    *,
    result_count: int = DEFAULT_RESULT_COUNT,
) -> list[Document]:
    """Return repository chunks semantically similar to a natural-language query."""
    if not query.strip():
        raise ValueError("A retrieval query cannot be empty.")
    if not VECTORSTORE_DIRECTORY.is_dir():
        raise RuntimeError(
            "Repository index not found. Run `python -m app.rag.ingest` first."
        )

    return get_vector_store().similarity_search(query, k=result_count)


def format_retrieval_results(documents: list[Document]) -> str:
    """Format a bounded set of retrieved chunks for a tool observation."""
    if not documents:
        return "No matching repository chunks were found."

    sections = []
    for document in documents:
        source = document.metadata.get("source", "unknown source")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        sections.append(
            f"--- {source} (chunk {chunk_index}) ---\n"
            f"{_truncate_retrieved_chunk(document.page_content)}"
        )
    return "\n\n".join(sections)


def _truncate_retrieved_chunk(content: str) -> str:
    """Keep each RAG result useful without allowing one chunk to dominate."""
    if len(content) <= MAX_RETRIEVED_CHUNK_CHARS:
        return content

    prefix_length = MAX_RETRIEVED_CHUNK_CHARS - len(CHUNK_TRUNCATION_MARKER)
    return f"{content[:prefix_length]}{CHUNK_TRUNCATION_MARKER}"


def search_repository(query: str) -> str:
    """Search repository content semantically and return source-grounded chunks.

    This function is intentionally shaped as a simple string-in, string-out
    tool. LangGraph will expose it to the model in the next integration step.
    """
    try:
        results = retrieve_repository_chunks(query)
    except (RuntimeError, ValueError) as error:
        return str(error)

    return format_retrieval_results(results)


if __name__ == "__main__":
    import sys

    retrieval_query = " ".join(sys.argv[1:]).strip()
    if not retrieval_query:
        print('Usage: python -m app.rag.retriever "How does this project work?"')
    else:
        try:
            results = retrieve_repository_chunks(retrieval_query)
        except (RuntimeError, ValueError) as error:
            print(error)
        else:
            print(format_retrieval_results(results))
