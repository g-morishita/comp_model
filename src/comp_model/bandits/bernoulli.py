from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from ..spec import TaskSpec, OutcomeType


@dataclass(slots=True)
class BernoulliBandit:
    """
    K-armed Bernoulli bandit.
    Outcomes: o ~ Bernoulli(p[action]) returning float 0.0/1.0
    """
    probs: Sequence[float]
    state: int = 0  # constant state by default

    def __post_init__(self) -> None:
        if len(self.probs) < 2:
            raise ValueError("BernoulliBandit requires at least 2 arms.")
        for p in self.probs:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"Invalid prob {p}; must be in [0,1].")

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            n_actions=len(self.probs),
            outcome_type=OutcomeType.BINARY,
            is_social=False,
        )

    def reset_block(self, *, rng: np.random.Generator) -> None:
        # if you later want random walk probs etc, do it here
        self.state = 0

    def step(self, *, action: int, rng: np.random.Generator) -> float:
        a = int(action)
        if a < 0 or a >= len(self.probs):
            raise ValueError("Action out of range.")
        return 1.0 if float(rng.random()) < float(self.probs[a]) else 0.0

    def get_state(self) -> Any:
        return self.state
