from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...interfaces.model import SocialComputationalModel
from ...interfaces.bandit import SocialObservation
from ...spec import TaskSpec, RewardType


def _softmax(u: np.ndarray, beta: float) -> np.ndarray:
    z = beta * u
    z = z - float(np.max(z))
    expz = np.exp(z)
    return expz / float(np.sum(expz))


def _signed_reward(r: float, spec: TaskSpec) -> float:
    """
    VS2 equations in Najar2020 use ±1 rewards and symmetric updates.
    If the task provides binary rewards (0/1), map to -1/+1 by default.
    Otherwise assume reward is already on a signed scale.
    """
    if spec.reward_type == RewardType.BINARY:
        return 2.0 * float(r) - 1.0
    return float(r)


def _sym_update(q: np.ndarray, a: int, lr: float, target: float) -> None:
    """
    Symmetric update:
      q[a]  += lr * ( target - q[a])
      q[~a] += lr * (-target - q[~a])
    """
    b = 1 - a
    q[a] += lr * (target - q[a])
    q[b] += lr * ((-target) - q[b])


def _kappa_bonus(last_choice: int | None, a: int, kappa: float) -> float:
    """
    Simple perseveration: +kappa if repeating last private choice, else -kappa.
    """
    if last_choice is None or kappa == 0.0:
        return 0.0
    return +kappa if a == last_choice else -kappa


@dataclass(slots=True)
class VS(SocialComputationalModel):
    """
    Najar et al. (2020) Value Shaping (VS): demonstrations directly shape the learner’s values.

    Parameters
    ----------
    alpha_p : float
        Private outcome learning rate.
    alpha_i : float
        Imitation/value-shaping rate used on observation trials.
    beta : float
        Inverse temperature for softmax.
    kappa : float
        Choice autocorrelation / perseveration strength.

    Notes
    -----
    - Assumes 2 actions (as in the paper task).
    - Observation update uses a pseudo-reward target of +1 for demonstrated action (and -1 for the other),
      implemented as a symmetric update on Q. 
    - Private learning is RW2-style symmetric update using signed reward r (±1). 
    """
    alpha_p: float = 0.2
    alpha_i: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []          # per-state values, each shape (2,)
        self._last_choice: list[int | None] = []  # per-state last private choice (for kappa)

    @property
    def param_names(self) -> Sequence[str]:
        return ("alpha_p", "alpha_i", "beta", "kappa")

    def supports(self, spec: TaskSpec) -> bool:
        return spec.n_actions == 2 and spec.is_social

    def reset_block(self, *, spec: TaskSpec) -> None:
        self._q = []
        self._last_choice = []

    def _ensure_state(self, s: int) -> None:
        while len(self._q) <= s:
            self._q.append(np.zeros(2, dtype=float))
        while len(self._last_choice) <= s:
            self._last_choice.append(None)

    def action_probs(self, *, state: Any, spec: TaskSpec) -> np.ndarray:
        s = int(state)
        self._ensure_state(s)
        q = self._q[s]

        # add perseveration term
        u = q.copy()
        u[0] += _kappa_bonus(self._last_choice[s], 0, self.kappa)
        u[1] += _kappa_bonus(self._last_choice[s], 1, self.kappa)
        return _softmax(u, self.beta)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Observation step: update learner Q values using the demonstrator action as pseudo-reward.
        """
        if not social.others_choices:
            return
        d = int(social.others_choices[0])

        s = int(state)
        self._ensure_state(s)
        q = self._q[s]

        # VS: directly shape Q with symmetric update toward demonstrated action (target=+1)
        _sym_update(q, d, float(self.alpha_i), target=1.0)

    def update(
        self,
        *,
        state: Any,
        action: int,
        reward: float,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Private outcome step: RW2 symmetric update.
        """
        s = int(state)
        self._ensure_state(s)
        q = self._q[s]

        r = _signed_reward(float(reward), spec)
        _sym_update(q, int(action), float(self.alpha_p), target=r)

        # update last private choice for perseveration
        self._last_choice[s] = int(action)
