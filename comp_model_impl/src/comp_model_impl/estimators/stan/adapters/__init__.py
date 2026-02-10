"""Stan adapter registry for model-specific templates.

This package exposes adapter classes that map computational models to Stan
program templates and define required priors. Adapters are resolved by the
Stan estimators at runtime.

Notes
-----
If you add a new model with a Stan template, implement a matching adapter and
export it here so it can be discovered by the adapter registry.

Examples
--------
>>> from comp_model_impl.estimators.stan.adapters import VSStanAdapter
>>> # adapter = VSStanAdapter(model=VS())  # doctest: +SKIP
"""

from .vicarious_rl import VicariousRLStanAdapter
from .vicarious_ap_vs import VicariousAPVSStanAdapter
from .vicarious_ap_db_stay import VicariousAPDBStayStanAdapter
from .vicarious_dir_db_stay import VicariousDirDBStayStanAdapter
from .vicarious_db_stay import VicariousDBStayStanAdapter
from .vicarious_db_stay_within_subject import VicariousDBStayWithinSubjectStanAdapter
from .qrl import QRLStanAdapter
from .vs import VSStanAdapter
from .vicarious_vs import VicariousVSStanAdapter
from .vicarious_vs_stay import VicariousVSStayStanAdapter
from .vicarious_rl_within_subject import VicariousRLWithinSubjectStanAdapter
from .vicQ_ap_dualw import VicQAPDualWStanAdapter
from .vicQ_ap_indep_dualw import VicQAPIndepDualWStanAdapter
from .vs_within_subject import VSWithinSubjectStanAdapter
from .base import StanAdapter

__all__ = [
    "StanAdapter",
    "QRLStanAdapter",
    "VicariousRLStanAdapter",
    "VicariousAPVSStanAdapter",
    "VicariousAPDBStayStanAdapter",
    "VicariousDirDBStayStanAdapter",
    "VicariousDBStayStanAdapter",
    "VicariousDBStayWithinSubjectStanAdapter",
    "VSStanAdapter",
    "VicariousVSStanAdapter",
    "VicariousVSStayStanAdapter",
    "VicariousRLWithinSubjectStanAdapter",
    "VicQAPDualWStanAdapter",
    "VicQAPIndepDualWStanAdapter",
    "VSWithinSubjectStanAdapter",
]
