from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class NamedRegistry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        if name in self._items:
            raise ValueError(f"Already registered: {name}")
        self._items[name] = item

    def get(self, name: str) -> T:
        return self._items[name]

    def __getitem__(self, name: str) -> T:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items.keys())


@dataclass
class Registry:
    models: NamedRegistry[type] = field(default_factory=NamedRegistry)
    estimators: NamedRegistry[type] = field(default_factory=NamedRegistry)
    generators: NamedRegistry[type] = field(default_factory=NamedRegistry)

    bandits: NamedRegistry[type] = field(default_factory=NamedRegistry)
    demonstrators: NamedRegistry[type] = field(default_factory=NamedRegistry)
    tasks: NamedRegistry[type] = field(default_factory=NamedRegistry)

