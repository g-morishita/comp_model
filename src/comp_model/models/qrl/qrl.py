from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...interfaces.model import ComputationalModel
from ...spec import TaskSpec
from ...utility import _softmax


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

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []
        
    @property
    def param_names(self) -> Sequence[str]:
        return ("alpha", "beta")
    
    def supports(self, spec: TaskSpec) -> bool:
        return not spec.is_social and spec.n_actions >= 2

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

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        a = int(action)
        if 0 <= a < nA:
            self._q[s][a] += float(self.alpha) * (float(outcome) - self._q[s][a])
            