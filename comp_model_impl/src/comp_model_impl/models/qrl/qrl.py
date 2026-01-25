from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.spec import TaskSpec
from comp_model_core.utility import _softmax
from comp_model_core.params import ParameterSchema

from .schema import qrl_schema


@dataclass(slots=True)
class QRL(ComputationalModel):
    """
    Standard Q Reinforcement learning model

    Parameters
    ----------
    alpha : float
        outcome learning rate.
    beta : float
        Softmax inverse temperature.
    """
    alpha: float = 0.2
    beta: float = 5.0

    # config (not estimated by default)
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []
        
    @property
    def param_schema(self) -> ParameterSchema:
        # defaults reflect current object state
        return qrl_schema(
            alpha_default=float(self.alpha),
            beta_default=float(self.beta),
            beta_max=float(self.beta_max),
        )
    
    def supports(self, spec: TaskSpec) -> bool:
        return not spec.is_social and spec.max_n_actions >= 2

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
        nA = int(spec.max_n_actions)
        self._ensure_state(s, nA)

        q = self._q[s]
        return _softmax(q, self.beta)

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if outcome is None:
            return
        s = int(state)
        nA = int(spec.max_n_actions)
        self._ensure_state(s, nA)

        a = int(action)
        if 0 <= a < nA:
            self._q[s][a] += float(self.alpha) * (float(outcome) - self._q[s][a])
            