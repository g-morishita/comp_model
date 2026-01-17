from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from ..interfaces.bandit import SocialObservation
from ..spec import TaskSpec, OutcomeType


@dataclass(slots=True)
class SocialBernoulliBandit:
    """
    K-armed Bernoulli bandit + demonstrator.

    - reward to participant: Bernoulli(probs[action])
    - demonstrator produces an observed action each trial:
        with prob p_demo_best chooses argmax(probs)
        else chooses uniformly among non-best arms
    """
    probs: Sequence[float]
    p_demo_best: float = 0.8  # demonstrator accuracy
    state: int = 0

    def __post_init__(self) -> None:
        if len(self.probs) < 2:
            raise ValueError("SocialBernoulliBandit requires at least 2 arms.")
        if not (0.0 <= float(self.p_demo_best) <= 1.0):
            raise ValueError("p_demo_best must be in [0,1].")
        for p in self.probs:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"Invalid prob {p}; must be in [0,1].")

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            n_actions=len(self.probs),
            outcome_type=OutcomeType.BINARY,
            is_social=True,
        )

    def reset_block(self, *, rng: np.random.Generator) -> None:
        self.state = 0

    def get_state(self) -> int:
        return self.state

    def step(self, *, action: int, rng: np.random.Generator) -> float:
        a = int(action)
        if a < 0 or a >= len(self.probs):
            raise ValueError("Action out of range.")
        return 1.0 if float(rng.random()) < float(self.probs[a]) else 0.0

    def observe_others(self, *, rng: np.random.Generator) -> SocialObservation:
        k = len(self.probs)
        best = int(np.argmax(np.asarray(self.probs, dtype=float)))

        if float(rng.random()) < float(self.p_demo_best):
            demo = best
        else:
            others = [a for a in range(k) if a != best]
            demo = int(rng.choice(others))

        return SocialObservation(others_choices=[demo], others_outcomes=None, info={"best_arm": best})
