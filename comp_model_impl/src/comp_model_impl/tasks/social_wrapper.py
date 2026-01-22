from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model_core.interfaces.bandit import Bandit, SocialBandit, SocialObservation, BanditStep
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import TaskSpec
from copy import deepcopy

@dataclass(slots=True)
class SocialBanditWrapper(SocialBandit):
    """
    SocialBandit = Bandit + observe_others().
    Bandit is deepcopied, so the state is not shared.
    Outcome visibilities are globally controlled by arguments: `reveal_self_outcome` and `reveal_demo_outcome`


    This wrapper composes:
      - base: Bandit (outcome dynamics)
      - demonstrator: Demonstrator (others' action generator; can be RL, scripted, etc.)

    The demonstrator can optionally learn from its own sampled outcome.
    """
    base: Bandit
    demonstrator: Demonstrator
    reveal_demo_outcome: bool
    reveal_self_outcome: bool

    def __post_init__(self):
        self.base = deepcopy(self.base)

    @property
    def spec(self) -> TaskSpec:
        s = self.base.spec
        return TaskSpec(
            n_actions=s.n_actions, 
            outcome_type=s.outcome_type, 
            outcome_range=s.outcome_range, 
            outcome_is_bounded=s.outcome_is_bounded, 
            has_state=s.has_state, is_social=True,
            )

    def reset(self, rng: np.random.Generator) -> Any:
        st = self.base.reset(rng=rng)
        self.demonstrator.reset(spec=self.spec, rng=rng)
        return st

    def get_state(self) -> Any:
        return self.base.get_state()

    def step(self, *, action: int, rng: np.random.Generator) -> BanditStep:
        step = self.base.step(action=action, rng=rng)

        return BanditStep(
            outcome=step.outcome,
            observed_outcome=step.outcome if self.reveal_self_outcome else None,
            info={**(step.info or {}), "self_outcome_observed": self.reveal_self_outcome}
        )

    def observe_others(self, *, rng: np.random.Generator) -> SocialObservation:
        state = self.get_state()
        a = int(self.demonstrator.act(state=state, spec=self.spec, rng=rng))
        
        step = self.base.step(action=a, rng=rng)
        observed_outcome = step.observed_outcome
        true_outcome = step.outcome

        self.demonstrator.update(state=state, action=a, outcome=true_outcome, spec=self.spec, rng=rng)  # Demonstrator observed true outcome.

        return SocialObservation(
            others_choices=[a],
            others_outcomes=[true_outcome],
            observed_others_outcomes=[observed_outcome] if self.reveal_demo_outcome else None,
            info={"demo_outcome_observed": self.reveal_demo_outcome},
        )
