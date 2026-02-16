"""Registry-based component resolution for model recovery."""

from __future__ import annotations

from typing import Any

from ..common import (
    resolve_estimator_callable as _resolve_estimator_callable,
    resolve_generator_callable as _resolve_generator_callable,
    resolve_model_callable as _resolve_model_callable,
)


def resolve_model_callable(reference: str, *, registries: Any) -> Any:
    """Resolve a model class/factory by registry key."""

    return _resolve_model_callable(reference, registries=registries)


def resolve_estimator_callable(reference: str, *, registries: Any) -> Any:
    """Resolve an estimator class/factory by registry key."""

    return _resolve_estimator_callable(reference, registries=registries)


def resolve_generator_callable(reference: str, *, registries: Any) -> Any:
    """Resolve a generator class/factory by registry key."""

    return _resolve_generator_callable(reference, registries=registries)
