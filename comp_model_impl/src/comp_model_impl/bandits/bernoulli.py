from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind


@dataclass(slots=True)
class BernoulliBanditEnv(BanditEnv):
    """K-armed Bernoulli bandit environment.

    Notes
    -----
    - Environment returns *true* outcomes only.
    - Outcome visibility / noisy feedback is handled by a BlockRunner wrapper.
    """

    probs: Sequence[float]
    state: int = 0  # single context by default

    def __post_init__(self) -> None:
        if len(self.probs) < 2:
            raise ValueError("BernoulliBanditEnv requires at least 2 arms.")
        for p in self.probs:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"Invalid prob {p}; must be in [0,1].")

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            n_actions=len(self.probs),
            outcome_type=OutcomeType.BINARY,
            outcome_range=(0.0, 1.0),
            outcome_is_bounded=True,
            is_social=False,
            state_kind=StateKind.DISCRETE,
            n_states=1,
        )

    def reset(self, *, rng: np.random.Generator) -> Any:
        self.state = 0
        return self.state

    def step(self, *, action: int, rng: np.random.Generator) -> EnvStep:
        a = int(action)
        p = float(self.probs[a])
        out = 1.0 if float(rng.random()) < p else 0.0
        return EnvStep(outcome=out, done=False, info=None)

    def get_state(self) -> Any:
        return self.state
