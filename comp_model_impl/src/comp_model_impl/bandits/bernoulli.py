from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from comp_model_core.spec import TaskSpec, OutcomeType
from comp_model_core.interfaces.bandit import Bandit, BanditStep

@dataclass(slots=True)
class BernoulliBandit(Bandit):
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

    def reset(self, *, rng: np.random.Generator) -> None:
        # if you later want random walk probs etc, do it here
        self.state = 0

    def step(self, action: int, rng: np.random.Generator) -> BanditStep:
        a = int(action)
        p = float(self.probs[a])
        out = 1.0 if float(rng.random()) < p else 0.0
        return BanditStep(outcome=out, done=False, info=None)

    def get_state(self) -> Any:
        return self.state
