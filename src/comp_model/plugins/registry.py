"""Plugin manifests and auto-discovery registry.

This module provides a lightweight registry for discoverable model and problem
components. Discovery scans a package for ``PLUGIN_MANIFESTS`` constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from typing import Any, Callable, Literal

from comp_model.core.requirements import ComponentRequirements

ComponentKind = Literal["model", "problem"]


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    """Manifest for a discoverable component.

    Parameters
    ----------
    kind : {"model", "problem"}
        Component category.
    component_id : str
        Stable identifier unique within ``kind``.
    factory : Callable[..., Any]
        Callable that creates the component instance.
    version : str, optional
        Semantic version label for the manifest.
    description : str, optional
        Human-readable component summary.
    requirements : ComponentRequirements | None, optional
        Optional data requirements used by compatibility checks.
    """

    kind: ComponentKind
    component_id: str
    factory: Callable[..., Any]
    version: str = "1.0.0"
    description: str = ""
    requirements: ComponentRequirements | None = None


class PluginRegistry:
    """Registry for model/problem manifests with package auto-discovery."""

    def __init__(self) -> None:
        self._manifests: dict[tuple[ComponentKind, str], ComponentManifest] = {}

    def register(self, manifest: ComponentManifest) -> None:
        """Register one component manifest.

        Parameters
        ----------
        manifest : ComponentManifest
            Manifest to register.

        Raises
        ------
        ValueError
            If a different manifest already exists for the same key.
        """

        key = (manifest.kind, manifest.component_id)
        existing = self._manifests.get(key)
        if existing is None:
            self._manifests[key] = manifest
            return

        if existing != manifest:
            raise ValueError(
                "manifest conflict for "
                f"{manifest.kind}:{manifest.component_id}; already registered"
            )

    def get(self, kind: ComponentKind, component_id: str) -> ComponentManifest:
        """Return a manifest by key.

        Parameters
        ----------
        kind : {"model", "problem"}
            Component category.
        component_id : str
            Component identifier.

        Returns
        -------
        ComponentManifest
            Registered manifest.

        Raises
        ------
        KeyError
            If manifest is not registered.
        """

        return self._manifests[(kind, component_id)]

    def list(self, kind: ComponentKind | None = None) -> tuple[ComponentManifest, ...]:
        """List registered manifests.

        Parameters
        ----------
        kind : {"model", "problem"} | None, optional
            Optional kind filter.

        Returns
        -------
        tuple[ComponentManifest, ...]
            Sorted manifest tuple.
        """

        manifests = tuple(self._manifests.values())
        if kind is not None:
            manifests = tuple(item for item in manifests if item.kind == kind)

        return tuple(sorted(manifests, key=lambda item: (item.kind, item.component_id)))

    def create(self, kind: ComponentKind, component_id: str, **kwargs: Any) -> Any:
        """Create a component instance from its manifest factory.

        Parameters
        ----------
        kind : {"model", "problem"}
            Component category.
        component_id : str
            Component identifier.
        **kwargs : Any
            Factory keyword arguments.

        Returns
        -------
        Any
            Constructed component instance.
        """

        manifest = self.get(kind, component_id)
        return manifest.factory(**kwargs)

    def create_model(self, component_id: str, **kwargs: Any) -> Any:
        """Create a model component by ID."""

        return self.create("model", component_id, **kwargs)

    def create_problem(self, component_id: str, **kwargs: Any) -> Any:
        """Create a problem component by ID."""

        return self.create("problem", component_id, **kwargs)

    def discover(self, package_name: str) -> tuple[ComponentManifest, ...]:
        """Discover and register manifests in a package tree.

        Parameters
        ----------
        package_name : str
            Package root to scan. Every module may define ``PLUGIN_MANIFESTS``.

        Returns
        -------
        tuple[ComponentManifest, ...]
            Manifests discovered in the package.
        """

        discovered: list[ComponentManifest] = []
        package = importlib.import_module(package_name)

        modules = [package]
        if hasattr(package, "__path__"):
            for module_info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
                modules.append(importlib.import_module(module_info.name))

        for module in modules:
            manifests = getattr(module, "PLUGIN_MANIFESTS", ())
            for manifest in manifests:
                if not isinstance(manifest, ComponentManifest):
                    raise TypeError(
                        f"{module.__name__}.PLUGIN_MANIFESTS must contain ComponentManifest objects"
                    )
                self.register(manifest)
                discovered.append(manifest)

        return tuple(discovered)


def build_default_registry() -> PluginRegistry:
    """Build a registry with all built-in models and problems discovered."""

    registry = PluginRegistry()
    registry.discover("comp_model.models")
    registry.discover("comp_model.problems")
    return registry
