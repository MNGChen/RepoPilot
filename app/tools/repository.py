"""Safe, read-only helpers for DevPilot's target repository.

The functions in this module are intentionally independent of the agent. A
later step will expose them to the LLM as LangChain tools.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_REPOSITORY = PROJECT_ROOT / "workspace" / "test_repo"
MAX_FILE_SIZE_BYTES = 100_000
MAX_TREE_ENTRIES = 500


def list_files() -> str:
    """Return a readable tree of the files in DevPilot's target repository."""
    if not TARGET_REPOSITORY.is_dir():
        return "Repository not found: workspace/test_repo"

    lines = [f"{TARGET_REPOSITORY.name}/"]
    entry_count = _append_tree(TARGET_REPOSITORY, lines, prefix="", entry_count=0)

    if entry_count >= MAX_TREE_ENTRIES:
        lines.append(f"... output stopped after {MAX_TREE_ENTRIES} entries")

    return "\n".join(lines)


def read_file(path: str) -> str:
    """Read one UTF-8 text file inside ``workspace/test_repo`` safely.

    ``path`` must be relative to the target repository. Resolving the path
    before reading also prevents a symlink from escaping the repository.
    """
    if not TARGET_REPOSITORY.is_dir():
        return "Repository not found: workspace/test_repo"

    resolved_path, error = _resolve_repository_path(path)
    if error:
        return error

    if not resolved_path.exists():
        return f"File not found: {path}"
    if not resolved_path.is_file():
        return f"Not a file: {path}"

    try:
        file_size = resolved_path.stat().st_size
    except OSError:
        return f"Could not inspect file: {path}"

    if file_size > MAX_FILE_SIZE_BYTES:
        return (
            f"File is too large to read ({file_size} bytes). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES} bytes."
        )

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"File is not valid UTF-8 text: {path}"
    except OSError:
        return f"Could not read file: {path}"

    return f"--- {path} ---\n{content}"


def _resolve_repository_path(path: str) -> tuple[Path, str | None]:
    """Resolve a user path and reject anything outside the repository."""
    requested_path = Path(path)
    if not path or requested_path.is_absolute() or requested_path.drive:
        return TARGET_REPOSITORY, "Invalid path: provide a non-empty relative path."

    repository_root = TARGET_REPOSITORY.resolve()
    candidate = (repository_root / requested_path).resolve()

    try:
        candidate.relative_to(repository_root)
    except ValueError:
        return TARGET_REPOSITORY, "Access denied: paths must stay inside workspace/test_repo."

    return candidate, None


def _append_tree(
    directory: Path,
    lines: list[str],
    *,
    prefix: str,
    entry_count: int,
) -> int:
    """Append a directory tree without following directory symlinks."""
    try:
        entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError:
        lines.append(f"{prefix}└── [unreadable directory]")
        return entry_count

    for index, entry in enumerate(entries):
        if entry_count >= MAX_TREE_ENTRIES:
            return entry_count

        is_last = index == len(entries) - 1
        branch = "└── " if is_last else "├── "

        if entry.is_symlink():
            lines.append(f"{prefix}{branch}{entry.name} [symlink]")
        elif entry.is_dir():
            lines.append(f"{prefix}{branch}{entry.name}/")
            child_prefix = prefix + ("    " if is_last else "│   ")
            entry_count = _append_tree(
                entry,
                lines,
                prefix=child_prefix,
                entry_count=entry_count + 1,
            )
            continue
        else:
            lines.append(f"{prefix}{branch}{entry.name}")

        entry_count += 1

    return entry_count
