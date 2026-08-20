"""Small, persistent long-term memory store for DevPilot V3."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.retriever import create_embeddings
from app.tools.repository import PROJECT_ROOT


MEMORY_STORE_DIRECTORY: Final = PROJECT_ROOT / "workspace" / "memory_store"
MEMORY_COLLECTION_NAME: Final = "devpilot_memory"
MEMORY_TYPES: Final = {
    "user_preference",
    "project_fact",
    "architecture_decision",
}
MAX_MEMORY_LENGTH: Final = 1_000
MAX_RETRIEVED_MEMORIES: Final = 3
MAX_GLOBAL_PREFERENCES: Final = 3
MIN_RELEVANCE_SCORE: Final = 0.65


def get_memory_store() -> Chroma:
    """Open DevPilot's persistent memory collection, separate from RAG."""
    return Chroma(
        collection_name=MEMORY_COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(MEMORY_STORE_DIRECTORY),
    )


def save_memory(content: str, memory_type: str) -> str:
    """Persist one concise, durable memory chosen by the agent."""
    cleaned_content = content.strip()
    if not cleaned_content:
        return "Memory was not saved: content cannot be empty."
    if len(cleaned_content) > MAX_MEMORY_LENGTH:
        return f"Memory was not saved: content exceeds {MAX_MEMORY_LENGTH} characters."
    if memory_type not in MEMORY_TYPES:
        valid_types = ", ".join(sorted(MEMORY_TYPES))
        return f"Memory was not saved: memory_type must be one of {valid_types}."

    memory_id = str(uuid4())
    document = Document(
        page_content=cleaned_content,
        metadata={
            "memory_id": memory_id,
            "memory_type": memory_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    get_memory_store().add_documents([document], ids=[memory_id])
    return f"Memory saved as {memory_type}."


def retrieve_relevant_memories(query: str) -> list[Document]:
    """Return global preferences plus semantic matches for a new user query."""
    if not query.strip() or not MEMORY_STORE_DIRECTORY.is_dir():
        return []

    memory_store = get_memory_store()
    preferences = _get_user_preferences(memory_store)
    semantic_matches = memory_store.similarity_search_with_relevance_scores(
        query,
        k=MAX_RETRIEVED_MEMORIES,
    )
    relevant_documents = [
        document
        for document, score in semantic_matches
        if score >= MIN_RELEVANCE_SCORE
    ]
    return _deduplicate_memories([*preferences, *relevant_documents])


def format_memory_context(memories: list[Document]) -> str:
    """Create a compact context block for the LLM, retaining memory type."""
    if not memories:
        return ""

    lines = ["Relevant long-term memory:"]
    for memory in memories:
        memory_type = memory.metadata.get("memory_type", "memory")
        lines.append(f"- ({memory_type}) {memory.page_content}")
    return "\n".join(lines)


def _get_user_preferences(memory_store: Chroma) -> list[Document]:
    """Return only the newest global preferences for every reply."""
    stored = memory_store.get(
        where={"memory_type": "user_preference"},
        include=["documents", "metadatas"],
    )
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    preferences = [
        Document(page_content=document, metadata=metadata or {})
        for document, metadata in zip(documents, metadatas, strict=True)
        if document is not None
    ]
    preferences.sort(
        key=lambda memory: str(memory.metadata.get("created_at", "")), reverse=True
    )
    return preferences[:MAX_GLOBAL_PREFERENCES]


def _deduplicate_memories(memories: list[Document]) -> list[Document]:
    """Keep one copy when a preference is also a semantic search result."""
    unique_memories: list[Document] = []
    seen_ids: set[str] = set()

    for memory in memories:
        memory_id = str(memory.metadata.get("memory_id", memory.page_content))
        if memory_id not in seen_ids:
            unique_memories.append(memory)
            seen_ids.add(memory_id)

    return unique_memories
