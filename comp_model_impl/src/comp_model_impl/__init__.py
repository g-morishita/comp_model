"""Concrete implementations for :mod:`comp_model_core`.

This package contains models, tasks, generators, estimators, and related
components that implement the interfaces defined in :mod:`comp_model_core`.

Notes
-----
The top-level import is intentionally lightweight. Subpackages are loaded
lazily via :func:`__getattr__` so importing :mod:`comp_model_impl` does not
eagerly import all implementations.

Examples
--------
Access subpackages lazily:

>>> import comp_model_impl as impl
>>> impl.models  # triggers lazy import of comp_model_impl.models
<module 'comp_model_impl.models' ...>
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("comp-model-impl")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "analysis",
    "bandits",
    "demonstrators",
    "estimators",
    "generators",
    "likelihood",
    "models",
    "recovery",
    "tasks",
]


def __getattr__(name: str):
    if name in {
        "analysis",
        "bandits",
        "demonstrators",
        "estimators",
        "generators",
        "likelihood",
        "models",
        "recovery",
        "tasks",
    }:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
