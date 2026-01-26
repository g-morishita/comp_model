"""Pytest configuration.

These tests are intended to run from a source checkout without requiring
installation (e.g. without `pip install -e .`).

We add the project's `src/` directory to `sys.path` so `import comp_model_core`
works when running `pytest` from the `comp_model_core/` root.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()
