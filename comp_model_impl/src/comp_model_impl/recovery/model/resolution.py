"""Registry-based component resolution for model recovery."""

from __future__ import annotations

from typing import Any


def _resolve_registry_component(
    *,
    reference: str,
    registry: Any,
    kind: str,
) -> Any:
    """Resolve a component from a named registry.

    Parameters
    ----------
    reference : str
        Registry key.
    registry : Any
        Named registry exposing ``get(name)`` and ``names()``.
    kind : {"model", "estimator"}
        Component kind used in error messages.

    Returns
    -------
    Any
        Registered class or callable.

    Raises
    ------
    ValueError
        If ``reference`` is not registered.
    """

    try:
        return registry.get(reference)
    except KeyError as e:
        available = ", ".join(registry.names())
        if available:
            msg = (
                f"Could not resolve {kind} {reference!r}. "
                f"Use a registered {kind} key. "
                f"Available {kind} keys: {available}."
            )
        else:
            msg = (
                f"Could not resolve {kind} {reference!r}. "
                f"Use a registered {kind} key."
            )
        raise ValueError(msg) from e


def resolve_model_callable(reference: str, *, registries: Any) -> Any:
    """Resolve a model class/factory by registry key."""

    return _resolve_registry_component(
        reference=reference,
        registry=registries.models,
        kind="model",
    )


def resolve_estimator_callable(reference: str, *, registries: Any) -> Any:
    """Resolve an estimator class/factory by registry key."""

    return _resolve_registry_component(
        reference=reference,
        registry=registries.estimators,
        kind="estimator",
    )
