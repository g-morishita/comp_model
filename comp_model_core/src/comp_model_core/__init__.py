"""comp_model_core

Core interfaces and data structures for computational behavior modeling.

This package is intentionally lightweight and dependency-minimal. It defines:

- Data containers for trials, blocks, subjects, and studies.
- Interfaces (ABCs) for environments (bandits), block runners, models, generators,
  estimators, and demonstrators.
- Parameter schemas and transforms to standardize parameter handling and validation.
- Planning utilities for specifying simulation studies via JSON/YAML.

Notes
-----
The top-level package lazily imports submodules listed in ``__all__`` to keep import
times fast and to avoid importing optional dependencies unless needed.
"""

from __future__ import annotations

import importlib
from typing import Any, List

__all__ = [
    "data",
    "interfaces",
    "plans",
    "params",
    "errors",
    "spec",
    "rng",
    "utility",
    "events",
]


def __getattr__(name: str) -> Any:
    """Lazily import a top-level submodule.

    This enables patterns like::

        import comp_model_core
        comp_model_core.params  # triggers lazy import

    Parameters
    ----------
    name : str
        Attribute name requested from the module.

    Returns
    -------
    Any
        Imported submodule if ``name`` is in ``__all__``.

    Raises
    ------
    AttributeError
        If ``name`` is not a known top-level attribute.
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Return available attributes for interactive completion.

    Returns
    -------
    list[str]
        Sorted attribute names including lazily-loadable submodules.
    """
    return sorted(list(globals().keys()) + __all__)
