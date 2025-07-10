import os
from pathlib import Path


def list_files_by_extension(directory: str, extension: str):
    """Return a list of file paths with a given extension in ``directory``."""
    return [str(f) for f in Path(directory).glob(f"*{extension}")]


def rename_files_with_prefix(directory: str, prefix: str) -> None:
    """Rename files in ``directory`` by prepending ``prefix`` to each name."""
    for path in Path(directory).iterdir():
        if path.is_file():
            path.rename(path.with_name(prefix + path.name))


def remove_files_by_extension(directory: str, extension: str) -> None:
    """Delete all files with ``extension`` in ``directory``."""
    for path in Path(directory).glob(f"*{extension}"):
        path.unlink()


def count_lines(path: str) -> int:
    """Count the number of lines in a text file."""
    with open(path) as f:
        return sum(1 for _ in f)


def read_text(path: str) -> str:
    """Read an entire text file and return its contents."""
    with open(path) as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path``, overwriting any existing content."""
    with open(path, "w") as f:
        f.write(text)
