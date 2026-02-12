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
from .vicarious_rl_stay import VicariousRLStayStanAdapter
from .ap_rl_stay import APRLStayStanAdapter
from .ap_rl_nostay import APRLNoStayStanAdapter
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
from .vicarious_rl_stay_within_subject import VicariousRLStayWithinSubjectStanAdapter
from .ap_rl_stay_within_subject import APRLStayWithinSubjectStanAdapter
from .ap_rl_nostay_within_subject import APRLNoStayWithinSubjectStanAdapter
from .vicQ_ap_dualw_stay import VicQAPDualWStayStanAdapter
from .vicQ_ap_indep_dualw import VicQAPIndepDualWStanAdapter
from .vicQ_ap_dualw_stay_within_subject import VicQAPDualWStayWithinSubjectStanAdapter
from .vicQ_ap_dualw_nostay import VicQAPDualWNoStayStanAdapter
from .vicQ_ap_dualw_nostay_within_subject import VicQAPDualWNoStayWithinSubjectStanAdapter
from .vs_within_subject import VSWithinSubjectStanAdapter
from .base import StanAdapter

__all__ = [
    "StanAdapter",
    "QRLStanAdapter",
    "VicariousRLStanAdapter",
    "VicariousRLStayStanAdapter",
    "APRLStayStanAdapter",
    "APRLNoStayStanAdapter",
    "VicariousAPVSStanAdapter",
    "VicariousAPDBStayStanAdapter",
    "VicariousDirDBStayStanAdapter",
    "VicariousDBStayStanAdapter",
    "VicariousDBStayWithinSubjectStanAdapter",
    "VSStanAdapter",
    "VicariousVSStanAdapter",
    "VicariousVSStayStanAdapter",
    "VicariousRLWithinSubjectStanAdapter",
    "VicariousRLStayWithinSubjectStanAdapter",
    "APRLStayWithinSubjectStanAdapter",
    "APRLNoStayWithinSubjectStanAdapter",
    "VicQAPDualWStayStanAdapter",
    "VicQAPIndepDualWStanAdapter",
    "VicQAPDualWStayWithinSubjectStanAdapter",
    "VicQAPDualWNoStayStanAdapter",
    "VicQAPDualWNoStayWithinSubjectStanAdapter",
    "VSWithinSubjectStanAdapter",
]
