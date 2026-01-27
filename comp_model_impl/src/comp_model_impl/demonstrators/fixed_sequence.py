"""Demonstrator that plays a fixed action sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import numpy as np

from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import EnvironmentSpec


@dataclass(slots=True)
class FixedSequenceDemonstrator(Demonstrator):

    """
    Demonstrator that returns a predetermined sequence of actions.
    
    Parameters
    ----------
    actions : Sequence[int]
        Action indices to emit, one per call to :meth:`act`.
    
    Raises
    ------
    ValueError
        If more actions are requested than are available in ``actions``.
    """

    actions: Sequence[int]
    _t: int = 0

    @classmethod
    def from_config(cls, bandit_cfg: Mapping[str, Any], demo_cfg: Mapping[str, Any]) -> "FixedSequenceDemonstrator":
        return cls(actions=demo_cfg["actions"])
    
    def reset(self, *, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        self._t = 0

    def act(self, *, state: Any, spec: EnvironmentSpec, rng: np.random.Generator) -> int:
        if self._t < len(self.actions):
            a = int(self.actions[self._t])
        else:
            raise ValueError("the length of actions is less than the number of trials.")
        self._t += 1
        return a

    def update(self, *, state: Any, action: int, outcome: float, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        return
