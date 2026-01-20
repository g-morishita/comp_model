from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..interfaces.demonstrator import Demonstrator
from ..interfaces.model import ComputationalModel
from ..spec import TaskSpec


@dataclass(slots=True)
class RLDemonstrator(Demonstrator):
    model: ComputationalModel
    params: Mapping[str, float]

    def reset(self, *, spec: TaskSpec, rng: np.random.Generator) -> None:
        self.model.set_params(self.params)
        self.model.reset_block(spec=spec)

    def act(self, *, state: Any, spec: TaskSpec, rng: np.random.Generator) -> int:
        probs = self.model.action_probs(state=state, spec=spec)
        return int(rng.choice(spec.n_actions, p=probs))

    def update(self, *, state: Any, action: int, outcome: float, spec: TaskSpec, rng: np.random.Generator) -> None:
        self.model.update(state=state, action=int(action), outcome=float(outcome), spec=spec, info=None)
