"""Bandit task implementations.

This package collects environment classes that implement bandit tasks used
throughout the simulation and recovery pipelines. Each environment implements
the :class:`comp_model_core.interfaces.bandit.BanditEnv` interface, providing
``spec``, ``reset``, and ``step``.

Notes
-----
These environments are intentionally minimal. Outcome visibility, noise,
and social observation are handled by trial specifications and runner logic,
not inside the environment classes.

Examples
--------
In a study plan:

.. code-block:: yaml

   bandit_type: BernoulliBanditEnv
   bandit_config:
     probs: [0.2, 0.8]

Direct import:

.. code-block:: python

   from comp_model_impl.bandits import BernoulliBanditEnv
   env = BernoulliBanditEnv(probs=[0.2, 0.8])
"""

from .bernoulli import BernoulliBanditEnv
from .lottery_choice import LotteryChoiceBanditEnv

__all__ = [
    "BernoulliBanditEnv",
    "LotteryChoiceBanditEnv",
]
