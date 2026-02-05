"""Demonstrator implementations and factory helpers.

This package provides demonstrator policies used in social learning tasks.
Demonstrators choose actions (and optionally emit outcomes) that the observer
model can use as social information.

Notes
-----
Demonstrators are separate from environments. The environment defines task
outcomes, while demonstrators define another agent's behavior that can be
observed by the subject model.

Examples
--------
In a study plan:

.. code-block:: yaml

   demonstrator_type: NoisyBestArmDemonstrator
   demonstrator_config:
     p_best: 0.9

Direct import:

.. code-block:: python

   from comp_model_impl.demonstrators import NoisyBestArmDemonstrator
   demo = NoisyBestArmDemonstrator(p_best=0.9)
"""

from .fixed_sequence import FixedSequenceDemonstrator
from .noisy_best import NoisyBestArmDemonstrator
from .rl_agent import RLDemonstrator

__all__ = [
    "FixedSequenceDemonstrator",
    "NoisyBestArmDemonstrator",
    "RLDemonstrator",
]
