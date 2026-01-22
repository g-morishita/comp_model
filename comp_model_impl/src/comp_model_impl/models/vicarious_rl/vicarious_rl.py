from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import TaskSpec
from comp_model_core.utility import _softmax

from .schema import vicarious_rl_schema


@dataclass(slots=True)
class Vicarious_RL(SocialComputationalModel):
    """
    Vicarious reinforcement learning model.

    Parameters
    ----------
    alpha_o : float
        Vicarious outcome learning rate.
    beta : float
        Softmax inverse temperature.
    """
    alpha_o: float = 0.2
    beta: float = 3.0
    
    # config (not estimated)
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []

    @property
    def param_schema(self) -> ParameterSchema:
        return vicarious_rl_schema(
            alpha_o_default=float(self.alpha_o),
            beta_default=float(self.beta),
            beta_max=float(self.beta_max),
        )

    def supports(self, spec: TaskSpec) -> bool:
        return spec.is_social and spec.n_actions >= 2

    def reset_block(self, *, spec: TaskSpec) -> None:
        self._q = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))

        # If action count differs across blocks (rare), reset that state's vector safely.
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)

    def action_probs(self, *, state: Any, spec: TaskSpec) -> np.ndarray:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        q = self._q[s]
        return _softmax(q, self.beta)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if not social.others_choices:
            return
        
        if not social.observed_others_outcomes:
            return

        co = int(social.others_choices[0])
        oo = int(social.observed_others_outcomes[0])

        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        if 0 <= co < nA:
            # chosen-only social shaping toward pseudo_reward
            self._q[s][co] += float(self.alpha_o) * (float(oo) - self._q[s][co])

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        return
