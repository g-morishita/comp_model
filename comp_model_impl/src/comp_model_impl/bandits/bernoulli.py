"""Bernoulli bandit environment.

This module provides :class:`BernoulliBanditEnv`, a minimal K-armed Bernoulli
bandit that emits binary outcomes in a single, discrete state. It is intended
as a small, fast environment for simulation and parameter recovery runs.

Notes
-----
Outcome visibility, noise, and social feedback are handled outside the
environment (e.g., by block runners and trial specifications). This environment
always returns *true* outcomes.

Examples
--------
Direct usage:

>>> import numpy as np
>>> from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
>>> rng = np.random.default_rng(0)
>>> env = BernoulliBanditEnv(probs=[0.2, 0.8])
>>> env.reset(rng=rng)
0
>>> step = env.step(action=1, rng=rng)
>>> step.outcome in (0.0, 1.0)
True

In a study plan:

.. code-block:: yaml

   bandit_type: BernoulliBanditEnv
   bandit_config:
     probs: [0.2, 0.8]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import numpy as np

from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind


@dataclass(slots=True)
class BernoulliBanditEnv(BanditEnv):
    """K-armed Bernoulli bandit with binary rewards.

    Each action selects an arm with probability ``p`` of reward 1 and
    probability ``1 - p`` of reward 0. The environment exposes a single
    discrete state, so the task is effectively stateless.

    Parameters
    ----------
    probs : Sequence[float]
        Reward probabilities for each arm. Each value must be in ``[0, 1]`` and
        the number of arms must be at least 2.
    state : int, optional
        Integer state identifier. Defaults to ``0`` and is reset to ``0`` on
        :meth:`reset`.

    Notes
    -----
    - The :attr:`spec` describes action count and outcome range to downstream
      components such as models and estimators.
    - Use :meth:`from_config` to instantiate from plan YAML/JSON.

    References
    ----------
    Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An
    Introduction* (2nd ed.), Chapter 2.
    """

    probs: Sequence[float]
    state: int = 0  # single context by default

    def __post_init__(self) -> None:
        if len(self.probs) < 2:
            raise ValueError("BernoulliBanditEnv requires at least 2 arms.")
        for p in self.probs:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"Invalid prob {p}; must be in [0,1].")

    @classmethod
    def from_config(cls, cfg):
        return cls(probs=cfg["probs"])

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
