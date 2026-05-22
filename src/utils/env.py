"""Environment variable helpers for local runs."""

from __future__ import annotations

from pathlib import Path
import os


def load_env_file(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ.

    Existing non-empty environment variables are not overwritten.
    Empty or whitespace-only values may be replaced from the file.
    """

    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        existing = os.environ.get(key)
        if existing is not None and existing.strip():
            continue
        os.environ[key] = value.strip()


def load_text_file(path: str) -> str:
    """Load a UTF-8 text file if it exists, else return empty string."""

    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")
