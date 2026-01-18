from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..interfaces.bandit import Bandit, SocialBandit, SocialObservation, BanditStep
from ..interfaces.demonstrator import Demonstrator
from ..spec import TaskSpec


@dataclass(slots=True)
class SocialBanditWrapper(SocialBandit):
    """
    SocialBandit = Bandit + observe_others().

    This wrapper composes:
      - base: Bandit (outcome dynamics)
      - demonstrator: Demonstrator (others' action generator; can be RL, scripted, etc.)

    The demonstrator can optionally learn from its own sampled outcome.
    """
    base: Bandit
    demonstrator: Demonstrator
    reveal_demo_outcome: bool = False

    @property
    def spec(self) -> TaskSpec:
        s = self.base.spec
        return TaskSpec(n_actions=s.n_actions, outcome_type=s.outcome_type, is_social=True)

    def reset(self, rng: np.random.Generator) -> Any:
        st = self.base.reset(rng=rng)
        self.demonstrator.reset(spec=self.spec, rng=rng)
        return st

    def get_state(self) -> Any:
        return self.base.get_state()

    def step(self, *, action: int, rng: np.random.Generator) -> BanditStep:
        return self.base.step(action, rng)

    def observe_others(self, rng: np.random.Generator) -> SocialObservation:
        state = self.get_state()
        a = int(self.demonstrator.act(state=state, spec=self.spec, rng=rng))
        out = float(self.base.step(action=a, rng=rng).outcome)
        self.demonstrator.observe_outcome(state=state, action=a, outcome=out, spec=self.spec, rng=rng)

        return SocialObservation(
            others_choices=[a],
            others_outcomes=[out] if self.reveal_demo_outcome else None,
            info=None,
        )
