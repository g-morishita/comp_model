"""Model-based demonstrator.

This module defines :class:`RLDemonstrator`, which wraps a
:class:`~comp_model_core.interfaces.model.ComputationalModel` and uses the
model's action probabilities to generate demonstrator choices. This allows
you to simulate social demonstrations produced by the same class of models
you later fit to subjects.

Notes
-----
- Use an asocial model (e.g., ``QRL``) for demonstrators, since they only
  observe their own outcomes in the current runner setup.
- Action selection is stochastic, sampled from the model's action probabilities.
- The demonstrator's internal model state is reset at block boundaries.

Examples
--------
Direct instantiation:

>>> import numpy as np
>>> from comp_model_impl.demonstrators.rl_agent import RLDemonstrator
>>> from comp_model_impl.models import QRL
>>> demo = RLDemonstrator(model=QRL())
>>> demo.reset(spec=None, rng=np.random.default_rng(0))

From a study plan (via config):

.. code-block:: yaml

   demonstrator_type: RLDemonstrator
   demonstrator_config:
     model: QRL
     params:
       alpha: 0.2
       beta: 3.0
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

    """Demonstrator driven by a computational model policy.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model used to produce action probabilities.

    Notes
    -----
    The demonstrator calls :meth:`model.action_probs` and samples an action
    using the provided RNG. Outcomes are passed through to ``model.update``.
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
