from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..spec import TaskSpec


class Demonstrator(ABC):
    @abstractmethod
    def reset(self, *, spec: TaskSpec, rng: np.random.Generator) -> None: ...

    @abstractmethod
    def act(self, *, state: Any, spec: TaskSpec, rng: np.random.Generator) -> int: ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float,
        spec: TaskSpec,
        rng: np.random.Generator,
    ) -> None: ...
