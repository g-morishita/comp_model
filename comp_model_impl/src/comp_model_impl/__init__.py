"""comp_model_impl

Concrete implementations (models, tasks, generators, estimators) that conform to
interfaces defined in :mod:`comp_model_core`.

The top-level import is intentionally lightweight. Subpackages are loaded lazily.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("comp-model-impl")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
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
