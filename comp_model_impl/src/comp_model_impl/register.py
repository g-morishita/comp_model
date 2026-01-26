# comp_model_impl/register.py
from comp_model_core.registry import Registry
from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
from comp_model_impl.models.qrl.qrl import QRL

def make_registry() -> Registry:
    r = Registry()
    r.models.register("QRL", QRL)
    r.bandits.register("BernoulliBanditEnv", BernoulliBanditEnv)
    return r