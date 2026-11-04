"""``python -m safety_ai``: the same command line as the ``safety-ai`` console script."""

from __future__ import annotations

from safety_ai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
