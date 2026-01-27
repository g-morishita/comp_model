"""Demonstrator backed by a computational model.

The demonstrator reuses a :class:`~comp_model_core.interfaces.model.ComputationalModel`
policy to generate demonstration choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.spec import EnvironmentSpec


@dataclass(slots=True)
class RLDemonstrator(Demonstrator):

    """
    Demonstrator driven by a computational model policy.
    
    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model used to produce action probabilities.
    
    Notes
    -----
    The demonstrator calls :meth:`model.action_probs` and samples an action using
    the provided RNG.
    """

    model: ComputationalModel

    @classmethod
    def from_config(cls, bandit_cfg: Mapping[str, Any], demo_cfg: Mapping[str, Any]) -> "RLDemonstrator":
        model = demo_cfg["model"]()
        params = demo_cfg["params"]
        model.set_params(params=params)
        return cls(model=model)

    def reset(self, *, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        self.model.reset_block(spec=spec)

    def act(self, *, state: Any, spec: EnvironmentSpec, rng: np.random.Generator) -> int:
        probs = self.model.action_probs(state=state, spec=spec)
        return int(rng.choice(spec.n_actions, p=probs))

    def update(self, *, state: Any, action: int, outcome: float, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        self.model.update(state=state, action=int(action), outcome=float(outcome), spec=spec, info=None)
