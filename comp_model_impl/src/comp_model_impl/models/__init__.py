"""Model implementations shipped with :mod:`comp_model_impl`.

This package exports the concrete computational models used by generators,
estimators, and YAML/JSON study plans. Most users interact with these models
indirectly via the registry in :mod:`comp_model_impl.register`, but you can also
import and instantiate them directly for custom workflows or tests.

Models
------
- :class:`~comp_model_impl.models.qrl.qrl.QRL` (asocial Q-learning)
- :class:`~comp_model_impl.models.vs.vs.VS` (social Vicarious Social model)
- :class:`~comp_model_impl.models.vicarious_rl.vicarious_rl.Vicarious_RL`
- :class:`~comp_model_impl.models.vicarious_vs.vicarious_vs.Vicarious_VS`

Wrappers
--------
Within-subject hierarchical recovery uses shared+delta parameterization via
``wrap_model_with_shared_delta_conditions`` and the conditioned wrappers.

Examples
--------
Direct use (bypassing the registry):

>>> from comp_model_impl.models import VS
>>> model = VS()

Wrap a base model for within-subject (shared + delta) conditions:

>>> from comp_model_impl.models import wrap_model_with_shared_delta_conditions
>>> ws_model = wrap_model_with_shared_delta_conditions(
...     model=VS(),
...     conditions=["A", "B"],
...     baseline_condition="A",
... )

Integration with the registry (used by YAML/JSON plans):

>>> from comp_model_impl.register import make_registry
>>> registry = make_registry()
>>> registry.models["VS"] is VS
True

See Also
--------
comp_model_impl.register.make_registry
comp_model_impl.estimators.stan.adapters
comp_model_impl.generators.event_log
"""

from __future__ import annotations
from .vicarious_rl.vicarious_rl import Vicarious_RL
from .vicarious_vs.vicarious_vs import Vicarious_VS
from .qrl.qrl import QRL
from .vs.vs import VS
from .within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
    wrap_model_with_shared_delta_conditions,
)
from .vicarious_vs_stay.vicarious_vs_stay import Vicarious_VS_Stay
from .unidentifiable_qrl.unidentifiable_qrl import UnidentifiableQRL
from .vicarious_ap_vs.vicarious_ap_vs import Vicarious_AP_VS

__all__ = [
    "Vicarious_RL",
    "Vicarious_VS",
    "QRL",
    "VS",
    "Vicarious_VS_Stay",
    "UnidentifiableQRL",
    "Vicarious_AP_VS"
    "ConditionedSharedDeltaModel",
    "ConditionedSharedDeltaSocialModel",
    "wrap_model_with_shared_delta_conditions",
]
