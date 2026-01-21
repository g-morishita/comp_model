from comp_model_core.registry import Registry
from .qrl.qrl import QRL
from .vs.vs import VS

registry = Registry()
registry.models.register("QRL", QRL)
registry.models.register("VS", VS)