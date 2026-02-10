"""Implementation registry for :mod:`comp_model_impl`.

This module defines :func:`make_registry`, which returns a
:class:`comp_model_core.registry.Registry` populated with built-in models,
bandit environments, and demonstrators shipped in :mod:`comp_model_impl`.

Notes
-----
If you add a new model, bandit, or demonstrator implementation and want it
available via plan configuration or factory helpers, register it here.

This registry is what allows YAML/JSON study plans to resolve string names
(e.g., ``bandit_type: BernoulliBanditEnv``) into concrete classes at runtime.
"""

# comp_model_impl/register.py
from comp_model_core.registry import Registry

from .bandits.bernoulli import BernoulliBanditEnv
from .models import (
    QRL,
    VS,
    VicQ_AP_DualW,
    Vicarious_RL,
    Vicarious_VS,
    Vicarious_VS_Stay,
    UnidentifiableQRL,
    Vicarious_AP_DB_STAY,
    Vicarious_Dir_DB_Stay,
    Vicarious_DB_Stay,
)
from .demonstrators import NoisyBestArmDemonstrator, RLDemonstrator, FixedSequenceDemonstrator

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
    r.models.register("VS", VS)
    r.models.register("Vicarious_RL", Vicarious_RL)
    r.models.register("Vicarious_VS", Vicarious_VS)
    r.models.register("Vicarious_VS_Stay", Vicarious_VS_Stay)
    r.models.register("UnidentifiableQRL", UnidentifiableQRL)
    r.models.register("Vicarious_AP_DB_STAY", Vicarious_AP_DB_STAY)
    r.models.register("Vicarious_Dir_DB_Stay", Vicarious_Dir_DB_Stay)
    r.models.register("Vicarious_DB_Stay", Vicarious_DB_Stay)
    r.models.register("VicQ_AP_DualW", VicQ_AP_DualW)

    # Bandit environments
    r.bandits.register("BernoulliBanditEnv", BernoulliBanditEnv)

    # Demonstrators
    r.demonstrators.register("FixedSequenceDemonstrator", FixedSequenceDemonstrator)
    r.demonstrators.register("NoisyBestArmDemonstrator", NoisyBestArmDemonstrator)
    r.demonstrators.register("RLDemonstrator", RLDemonstrator)

    return r
