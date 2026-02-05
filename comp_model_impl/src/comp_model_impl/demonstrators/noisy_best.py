"""Noisy best-arm demonstrator.

This module provides :class:`NoisyBestArmDemonstrator`, a simple policy that
chooses the empirically best arm with probability ``p_best`` and otherwise
samples uniformly from the remaining arms. It is a lightweight model of
suboptimal demonstrators that are mostly but not perfectly optimal.

Notes
-----
- The demonstrator is memoryless: it does not update based on outcomes.
- ``p_best=1`` yields a greedy demonstrator; ``p_best=0`` yields uniform
  random choice among non-best arms.

Examples
--------
Direct usage:

>>> import numpy as np
>>> from comp_model_impl.demonstrators.noisy_best import NoisyBestArmDemonstrator
>>> demo = NoisyBestArmDemonstrator(reward_probs=[0.2, 0.8], p_best=0.9)
>>> demo.reset(spec=None, rng=np.random.default_rng(0))  # spec is only used for n_actions
>>> demo.act(state=None, spec=type("S", (), {"n_actions": 2})(), rng=np.random.default_rng(1)) in (0, 1)
True

In a study plan:

.. code-block:: yaml

   demonstrator_type: NoisyBestArmDemonstrator
   demonstrator_config:
     p_best: 0.9
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import numpy as np

from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import EnvironmentSpec


@dataclass(slots=True)
class NoisyBestArmDemonstrator(Demonstrator):
    """Mostly-greedy demonstrator over Bernoulli reward probabilities.

    Parameters
    ----------
    reward_probs : Sequence[float]
        Reward probabilities for each arm. The demonstrator treats the maximum
        value as the best arm.
    p_best : float
        Probability of choosing the best arm. Must be in ``[0, 1]``.

    Notes
    -----
    This class does not validate that ``reward_probs`` match the environment
    spec; that coupling is expected to be ensured by plan configuration.
    """
    reward_probs: Sequence[float]
    p_best: float

    @classmethod
    def from_config(cls, bandit_cfg: Mapping[str, Any], demo_cfg: Mapping[str, Any]) -> "NoisyBestArmDemonstrator":
        return cls(reward_probs=bandit_cfg["probs"], p_best=demo_cfg["p_best"])

    def reset(self, *, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        return

    def act(self, *, state: Any, spec: EnvironmentSpec, rng: np.random.Generator) -> int:
        k = spec.n_actions
        best = int(np.argmax(np.asarray(self.reward_probs, dtype=float)))
        if float(rng.random()) < float(self.p_best):
            return best
        others = [a for a in range(k) if a != best]
        return int(rng.choice(others))

    def update(self, *, state: Any, action: int, outcome: float, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        return
