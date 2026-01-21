"""
comp_model_core: Core interfaces and data structures for computational behavior modeling.
"""

from __future__ import annotations
import importlib

__all__ = [
    "data",
    "interfaces",
    "plans",
    "tasks",
    "params",
    "errors",
    "spec",
    "rng",
    "utility",
]

def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)
