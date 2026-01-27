"""Registry factory for concrete implementations.

This module defines :func:`make_registry`, which returns a
:class:`~comp_model_core.registry.Registry` populated with built-in models,
bandit environments, and demonstrators shipped in :mod:`comp_model_impl`.
"""

# comp_model_impl/register.py
from comp_model_core.registry import Registry

from .bandits.bernoulli import BernoulliBanditEnv
from .models.qrl.qrl import QRL
from .demonstrators.noisy_best import NoisyBestArmDemonstrator
from .demonstrators.rl_agent import RLDemonstrator

def make_registry() -> Registry:

    """
    Create a default implementation registry.
    
    Returns
    -------
    comp_model_core.registry.Registry
        Registry populated with the implementations shipped in :mod:`comp_model_impl`.
    """

    r = Registry()

    # Models
    r.models.register("QRL", QRL)

    # Bandit environments
    r.bandits.register("BernoulliBanditEnv", BernoulliBanditEnv)

    # Demonstrators
    r.demonstrators.register("NoisyBestArmDemonstrator", NoisyBestArmDemonstrator)
    r.demonstrators.register("RLDemonstrator", RLDemonstrator)

    return r
