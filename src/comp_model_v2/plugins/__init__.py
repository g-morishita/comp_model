"""Plugin registration and discovery utilities."""

from .registry import ComponentKind, ComponentManifest, PluginRegistry, build_default_registry

__all__ = [
    "ComponentKind",
    "ComponentManifest",
    "PluginRegistry",
    "build_default_registry",
]
