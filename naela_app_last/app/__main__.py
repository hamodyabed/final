"""Allow running the app with ``python -m app``."""

from .main_window import run

if __name__ == "__main__":
    raise SystemExit(run())
