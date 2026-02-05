"""Tests for Stan adapters package exports."""

from __future__ import annotations

import inspect

from comp_model_impl.estimators.stan import adapters


def test_adapters_init_exports_are_resolvable():
    """All names in __all__ resolve to objects on the module."""
    for name in adapters.__all__:
        assert hasattr(adapters, name)


def test_adapters_exports_include_classes():
    """Exported adapter names correspond to classes."""
    for name in adapters.__all__:
        obj = getattr(adapters, name)
        assert inspect.isclass(obj)
