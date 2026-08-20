"""Unit tests for DevPilot V3 long-term memory store.

Covers the two behaviours the V3 plan calls out explicitly:
  * "该保存" — valid memories are persisted and retrievable.
  * "不该保存" — invalid input is rejected with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.memory import store as memory_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the persistent memory store to a throwaway directory."""
    memory_dir = tmp_path / "memory_store"
    monkeypatch.setattr(memory_store, "MEMORY_STORE_DIRECTORY", memory_dir)
    return memory_dir


# ---------------------------------------------------------------------------
# save_memory — "该保存" (should save)
# ---------------------------------------------------------------------------

class TestSaveMemoryShouldSave:
    """Valid, concise, durable memories must be accepted."""

    @pytest.mark.parametrize(
        "memory_type",
        ["user_preference", "project_fact", "architecture_decision"],
    )
    def test_accepts_all_valid_types(
        self, isolated_memory_dir: Path, memory_type: str
    ) -> None:
        result = memory_store.save_memory(
            "User prefers answers in English.", memory_type
        )
        assert result == f"Memory saved as {memory_type}."

    def test_persists_to_disk(self, isolated_memory_dir: Path) -> None:
        memory_store.save_memory("The project uses LangGraph.", "project_fact")
        assert isolated_memory_dir.is_dir()
        # Chroma writes at least one file into the persist directory.
        assert any(isolated_memory_dir.rglob("*"))

    def test_strips_whitespace(self, isolated_memory_dir: Path) -> None:
        result = memory_store.save_memory(
            "   User likes concise answers.   ", "user_preference"
        )
        assert result == "Memory saved as user_preference."

    def test_boundary_length_accepted(self, isolated_memory_dir: Path) -> None:
        """A memory exactly at MAX_MEMORY_LENGTH is valid."""
        content = "x" * memory_store.MAX_MEMORY_LENGTH
        result = memory_store.save_memory(content, "project_fact")
        assert result == "Memory saved as project_fact."


# ---------------------------------------------------------------------------
# save_memory — "不该保存" (should NOT save)
# ---------------------------------------------------------------------------

class TestSaveMemoryShouldNotSave:
    """Invalid or low-value input must be rejected without touching the store."""

    def test_empty_string_rejected(self, isolated_memory_dir: Path) -> None:
        result = memory_store.save_memory("", "user_preference")
        assert "not saved" in result
        assert "empty" in result.lower()

    def test_whitespace_only_rejected(self, isolated_memory_dir: Path) -> None:
        result = memory_store.save_memory("   \n\t  ", "user_preference")
        assert "not saved" in result

    def test_exceeds_max_length_rejected(self, isolated_memory_dir: Path) -> None:
        content = "x" * (memory_store.MAX_MEMORY_LENGTH + 1)
        result = memory_store.save_memory(content, "project_fact")
        assert "not saved" in result
        assert str(memory_store.MAX_MEMORY_LENGTH) in result

    def test_invalid_type_rejected(self, isolated_memory_dir: Path) -> None:
        result = memory_store.save_memory("Some note.", "random_type")
        assert "not saved" in result
        assert "memory_type" in result

    def test_rejection_does_not_create_store(self, isolated_memory_dir: Path) -> None:
        """A rejected save must not leave a memory store on disk."""
        memory_store.save_memory("", "user_preference")
        assert not isolated_memory_dir.exists()


# ---------------------------------------------------------------------------
# retrieve_relevant_memories
# ---------------------------------------------------------------------------

class TestRetrieveRelevantMemories:
    """Semantic retrieval returns only memories above the relevance threshold."""

    def test_returns_empty_for_no_store(self, isolated_memory_dir: Path) -> None:
        assert memory_store.retrieve_relevant_memories("anything") == []

    def test_returns_empty_for_blank_query(self, isolated_memory_dir: Path) -> None:
        memory_store.save_memory("User prefers English.", "user_preference")
        assert memory_store.retrieve_relevant_memories("   ") == []

    def test_user_preference_always_returned(self, isolated_memory_dir: Path) -> None:
        """Preferences are global and bypass similarity filtering."""
        memory_store.save_memory("User prefers concise answers.", "user_preference")
        results = memory_store.retrieve_relevant_memories("completely unrelated topic")
        assert len(results) == 1
        assert "concise" in results[0].page_content

    def test_semantic_match_returned(self, isolated_memory_dir: Path) -> None:
        memory_store.save_memory(
            "The authentication service uses JWT tokens for user login.",
            "architecture_decision",
        )
        results = memory_store.retrieve_relevant_memories(
            "authentication service JWT tokens login"
        )
        assert any("JWT" in doc.page_content for doc in results)

    def test_deduplicates_preference_and_semantic_match(
        self, isolated_memory_dir: Path
    ) -> None:
        """A memory that is both a preference and a semantic hit appears once."""
        memory_store.save_memory("User prefers Python.", "user_preference")
        results = memory_store.retrieve_relevant_memories("User prefers Python")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# format_memory_context
# ---------------------------------------------------------------------------

class TestFormatMemoryContext:
    """The context block must be compact and retain memory type labels."""

    def test_empty_list_returns_empty_string(self) -> None:
        assert memory_store.format_memory_context([]) == ""

    def test_includes_type_and_content(self) -> None:
        doc = Document(
            page_content="User likes concise answers.",
            metadata={"memory_type": "user_preference"},
        )
        output = memory_store.format_memory_context([doc])
        assert "Relevant long-term memory:" in output
        assert "(user_preference)" in output
        assert "concise answers" in output

    def test_multiple_memories_each_on_own_line(self) -> None:
        docs = [
            Document(page_content="First.", metadata={"memory_type": "project_fact"}),
            Document(
                page_content="Second.", metadata={"memory_type": "architecture_decision"}
            ),
        ]
        output = memory_store.format_memory_context(docs)
        assert output.count("\n- (") == 2
