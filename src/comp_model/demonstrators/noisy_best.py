from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..interfaces.demonstrator import Demonstrator
from ..spec import TaskSpec


@dataclass(slots=True)
class NoisyBestArmDemonstrator(Demonstrator):
    """
    Chooses argmax(reward_probs) with prob p_best else random among others.
    """
    reward_probs: Sequence[float]
    p_best: float = 0.8

    def reset(self, *, spec: TaskSpec, rng: np.random.Generator) -> None:
        return

    def act(self, *, state: Any, spec: TaskSpec, rng: np.random.Generator) -> int:
        k = spec.n_actions
        best = int(np.argmax(np.asarray(self.reward_probs, dtype=float)))
        if float(rng.random()) < float(self.p_best):
            return best
        others = [a for a in range(k) if a != best]
        return int(rng.choice(others))

    def observe_outcome(self, *, state: Any, action: int, outcome: float, spec: TaskSpec, rng: np.random.Generator) -> None:
        return
