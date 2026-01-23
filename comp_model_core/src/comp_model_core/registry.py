"""
Simple registries for named components.

Registries are used to decouple configuration (e.g., JSON/YAML plans) from concrete
Python classes. For example, a :class:`~comp_model_core.plans.block.BlockPlan` can
store ``bandit_type="bernoulli_bandit"`` and the caller can look up the corresponding
constructor in a registry.

This module intentionally keeps the registry logic minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class NamedRegistry(Generic[T]):
    """
    A mapping from string names to items.

    The registry enforces unique names and provides a small convenience API.

    Notes
    -----
    This is not intended to be a plugin system; it is simply a structured dict with
    a clearer error message and a stable ``names()`` method for introspection.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """
        Register an item under a name.

        Parameters
        ----------
        name : str
            Registry key.
        item : T
            Item to register.

        Raises
        ------
        ValueError
            If ``name`` is already registered.
        """
        if name in self._items:
            raise ValueError(f"Already registered: {name}")
        self._items[name] = item

    def get(self, name: str) -> T:
        """
        Retrieve an item by name.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        T
            The registered item.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        return self._items[name]

    def __getitem__(self, name: str) -> T:
        """
        Convenience alias for :meth:`get`.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        T
            The registered item.
        """
        return self._items[name]

    def names(self) -> list[str]:
        """
        List registered names.

        Returns
        -------
        list[str]
            Sorted list of registry keys.
        """
        return sorted(self._items.keys())


@dataclass
class Registry:
    """
    Grouped registries for common component types.

    Attributes
    ----------
    models : NamedRegistry[type]
        Registry of computational model classes.
    estimators : NamedRegistry[type]
        Registry of estimator classes.
    generators : NamedRegistry[type]
        Registry of generator classes.
    bandits : NamedRegistry[type]
        Registry of bandit/task classes.
    demonstrators : NamedRegistry[type]
        Registry of demonstrator/policy classes.
    tasks : NamedRegistry[type]
        Generic task registry (optional, project-specific).
    """

    models: NamedRegistry[type] = field(default_factory=NamedRegistry)
    estimators: NamedRegistry[type] = field(default_factory=NamedRegistry)
    generators: NamedRegistry[type] = field(default_factory=NamedRegistry)

    bandits: NamedRegistry[type] = field(default_factory=NamedRegistry)
    demonstrators: NamedRegistry[type] = field(default_factory=NamedRegistry)
    tasks: NamedRegistry[type] = field(default_factory=NamedRegistry)
