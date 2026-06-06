"""Naela narrative-assessment app package.

Importing this package auto-loads ``.env`` from the project root (if present)
so configuration like ``GEMINI_API_KEY`` is available to every entry point
(``run_app.py``, ``python -m app``, unit tests, ...).
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_once() -> None:
    """Tiny zero-dependency ``.env`` loader.

    Reads ``<project_root>/.env`` and sets each ``KEY=VALUE`` pair as an
    environment variable, unless it is already defined (so a real env var
    always wins). Lines starting with ``#`` and blank lines are ignored.
    Surrounding double or single quotes around the value are stripped.
    """
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except OSError:
        # Don't crash the app just because .env is unreadable.
        pass


_load_dotenv_once()
