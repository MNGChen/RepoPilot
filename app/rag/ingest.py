"""Load supported text and code files from DevPilot's target repository."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.retriever import get_vector_store
from app.tools.repository import TARGET_REPOSITORY


SUPPORTED_FILE_TYPES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
}
IGNORED_DIRECTORIES = {".git", "node_modules", "__pycache__"}
MAX_DOCUMENT_SIZE_BYTES = 500_000
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150


def load_repository_documents() -> list[Document]:
    """Return supported, readable files as documents with repository metadata.

    The loader is intentionally separate from the runtime agent. It reads only
    the fixed target repository, skips generated directories and symlinks, and
    never follows a path outside that repository.
    """
    if not TARGET_REPOSITORY.is_dir():
        raise FileNotFoundError("Repository not found: workspace/test_repo")

    documents: list[Document] = []
    repository_root = TARGET_REPOSITORY.resolve()

    for directory, subdirectories, filenames in os.walk(repository_root, followlinks=False):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in IGNORED_DIRECTORIES and not (Path(directory) / name).is_symlink()
        ]

        for filename in filenames:
            file_path = Path(directory) / filename
            document = _load_file(file_path, repository_root)
            if document is not None:
                documents.append(document)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks while retaining source metadata.

    Chunk size is measured in characters. The overlap repeats a small amount of
    neighboring text so an idea spanning a chunk boundary remains searchable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    split_documents = splitter.split_documents(documents)
    next_chunk_index: defaultdict[str, int] = defaultdict(int)
    chunks: list[Document] = []

    # assign chunk index to each chunk based on its source file
    for document in split_documents:
        metadata = dict(document.metadata)
        source = str(metadata["source"])
        metadata["chunk_index"] = next_chunk_index[source]
        next_chunk_index[source] += 1
        chunks.append(Document(page_content=document.page_content, metadata=metadata))

    return chunks


def rebuild_repository_index() -> int:
    """Embed all repository chunks and replace DevPilot's local Chroma index.

    This is an explicit ingestion action, not something the runtime agent does.
    Rebuilding avoids stale or duplicated chunks after repository files change.
    """
    chunks = chunk_documents(load_repository_documents())
    if not chunks:
        raise ValueError("No supported, non-empty files were found to index.")

    vector_store = get_vector_store()
    try:
        vector_store.delete_collection()
    except ValueError:
        # No previous collection exists on the first indexing run.
        pass

    vector_store = get_vector_store()
    vector_store.add_documents(chunks, ids=_chunk_ids(chunks))
    return len(chunks)


def _chunk_ids(chunks: list[Document]) -> list[str]:
    """Create stable identifiers for chunks inside a rebuilt collection."""
    return [f"{chunk.metadata['source']}:{chunk.metadata['chunk_index']}" for chunk in chunks]


def _load_file(file_path: Path, repository_root: Path) -> Document | None:
    """Convert one safe supported file into a document, or skip it."""
    if file_path.is_symlink():
        return None

    file_type = SUPPORTED_FILE_TYPES.get(file_path.suffix.lower())
    if file_type is None:
        return None

    try:
        file_size = file_path.stat().st_size
    except OSError:
        return None

    if file_size > MAX_DOCUMENT_SIZE_BYTES:
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
        relative_path = file_path.resolve().relative_to(repository_root)
    except (OSError, UnicodeDecodeError, ValueError):
        return None

    if not content.strip():
        return None

    return Document(
        page_content=content,
        metadata={
            "source": relative_path.as_posix(),
            "file_type": file_type,
            "file_size_bytes": file_size,
        },
    )


if __name__ == "__main__":
    indexed_chunk_count = rebuild_repository_index()
    print(f"Indexed {indexed_chunk_count} chunks in workspace/vectorstore.")
