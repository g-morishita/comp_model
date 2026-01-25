from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import TaskSpec


@dataclass(slots=True)
class FixedSequenceDemonstrator(Demonstrator):
    actions: Sequence[int]
    fallback: str = "repeat_last"  # "repeat_last" or "random"
    _t: int = 0

    def reset(self, *, spec: TaskSpec, rng: np.random.Generator) -> None:
        self._t = 0

    def act(self, *, state: Any, spec: TaskSpec, rng: np.random.Generator) -> int:
        if self._t < len(self.actions):
            a = int(self.actions[self._t])
        else:
            if self.fallback == "repeat_last" and len(self.actions) > 0:
                a = int(self.actions[-1])
            else:
                a = int(rng.integers(0, spec.max_n_actions))
        self._t += 1
        return a

    def update(self, *, state: Any, action: int, outcome: float, spec: TaskSpec, rng: np.random.Generator) -> None:
        return
