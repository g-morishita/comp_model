"""Fixed-sequence demonstrator.

This module provides :class:`FixedSequenceDemonstrator`, a simple policy that
emits a predetermined sequence of actions. It is useful for deterministic
social signals in simulations, debugging, or unit tests.

Notes
-----
- The sequence is consumed in order; requesting more actions than provided
  raises a :class:`ValueError`.
- This demonstrator ignores outcomes and does not learn.

Examples
--------
Direct usage:

>>> import numpy as np
>>> from comp_model_impl.demonstrators.fixed_sequence import FixedSequenceDemonstrator
>>> demo = FixedSequenceDemonstrator(actions=[0, 1, 1, 0])
>>> demo.reset(spec=None, rng=np.random.default_rng(0))  # spec is ignored
>>> demo.act(state=None, spec=None, rng=np.random.default_rng(0))
0

In a study plan:

.. code-block:: yaml

   demonstrator_type: FixedSequenceDemonstrator
   demonstrator_config:
     actions: [0, 1, 1, 0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import numpy as np

from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import EnvironmentSpec


@dataclass(slots=True)
class FixedSequenceDemonstrator(Demonstrator):

    """Deterministic demonstrator that replays a fixed action list.

    The demonstrator emits actions in the order provided at construction time
    and raises an error once the sequence is exhausted. It ignores outcomes and
    does not learn, making it ideal for controlled social signals in tests or
    synthetic simulations.

    Parameters
    ----------
    actions : Sequence[int]
        Action indices to emit, one per call to :meth:`act`.

    Notes
    -----
    - The internal pointer is reset to the start on :meth:`reset`.
    - Outcomes passed to :meth:`update` are ignored.

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
