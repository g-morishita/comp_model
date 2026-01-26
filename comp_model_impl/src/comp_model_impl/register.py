# comp_model_impl/register.py
from comp_model_core.registry import Registry

from .bandits.bernoulli import BernoulliBanditEnv
from .models.qrl.qrl import QRL
from .demonstrators.noisy_best import NoisyBestArmDemonstrator

def make_registry() -> Registry:
    r = Registry()

    # Models
    r.models.register("QRL", QRL)

    # Bandit environments
    r.bandits.register("BernoulliBanditEnv", BernoulliBanditEnv)

    # Demonstrators
    r.demonstrators.register("NoisyBestArmDemonstrator", NoisyBestArmDemonstrator)

    return r