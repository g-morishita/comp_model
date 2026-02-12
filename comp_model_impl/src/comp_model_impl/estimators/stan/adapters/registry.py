"""Stan adapter registry.

This module resolves the correct :class:`StanAdapter` for a given model,
including within-subject shared+delta wrappers.

Examples
--------
>>> from comp_model_impl.estimators.stan.adapters.registry import resolve_stan_adapter
>>> from comp_model_impl.models import VS
>>> adapter = resolve_stan_adapter(VS())
>>> adapter.program("indiv").key
'vs'
"""

from __future__ import annotations

from comp_model_core.interfaces.model import ComputationalModel

from ....models import (
    QRL,
    VS,
    AP_RL_Stay,
    AP_RL_NoStay,
    VicQ_AP_DualW_Stay,
    VicQ_AP_DualW_NoStay,
    VicQ_AP_IndepDualW,
    Vicarious_RL,
    Vicarious_RL_Stay,
    Vicarious_AP_VS,
    Vicarious_AP_DB_STAY,
    Vicarious_Dir_DB_Stay,
    Vicarious_DB_Stay,
    Vicarious_VS,
    Vicarious_VS_Stay,
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)

from .vicarious_rl_within_subject import VicariousRLWithinSubjectStanAdapter
from .vicarious_rl_stay_within_subject import VicariousRLStayWithinSubjectStanAdapter
from .ap_rl_stay_within_subject import APRLStayWithinSubjectStanAdapter
from .ap_rl_nostay_within_subject import APRLNoStayWithinSubjectStanAdapter
from .vicarious_db_stay_within_subject import VicariousDBStayWithinSubjectStanAdapter
from .vicQ_ap_dualw_stay_within_subject import VicQAPDualWStayWithinSubjectStanAdapter
from .vicQ_ap_dualw_nostay_within_subject import VicQAPDualWNoStayWithinSubjectStanAdapter
from .vs_within_subject import VSWithinSubjectStanAdapter
from .base import StanAdapter
from .qrl import QRLStanAdapter
from .vicarious_rl import VicariousRLStanAdapter
from .vicarious_rl_stay import VicariousRLStayStanAdapter
from .ap_rl_stay import APRLStayStanAdapter
from .ap_rl_nostay import APRLNoStayStanAdapter
from .vicarious_ap_vs import VicariousAPVSStanAdapter
from .vicarious_ap_db_stay import VicariousAPDBStayStanAdapter
from .vicarious_dir_db_stay import VicariousDirDBStayStanAdapter
from .vicarious_db_stay import VicariousDBStayStanAdapter
from .vicarious_vs import VicariousVSStanAdapter
from .vicarious_vs_stay import VicariousVSStayStanAdapter
from .vicQ_ap_dualw_stay import VicQAPDualWStayStanAdapter
from .vicQ_ap_dualw_nostay import VicQAPDualWNoStayStanAdapter
from .vicQ_ap_indep_dualw import VicQAPIndepDualWStanAdapter
from .vs import VSStanAdapter


def resolve_stan_adapter(model: ComputationalModel) -> StanAdapter:
    """Resolve the appropriate Stan adapter for a model instance.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model instance to adapt.

    Returns
    -------
    StanAdapter
        Adapter instance that knows how to export data/config and interpret
        Stan outputs.

    Raises
    ------
    ValueError
        If no adapter is registered for the given model type.
    """
    # Within-subject wrapper models
    if isinstance(model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
        base = getattr(model, "base_model", None)
        if isinstance(base, VS):
            return VSWithinSubjectStanAdapter(model)
        if isinstance(base, Vicarious_RL):
            return VicariousRLWithinSubjectStanAdapter(model)
        if isinstance(base, Vicarious_RL_Stay):
            return VicariousRLStayWithinSubjectStanAdapter(model)
        if isinstance(base, AP_RL_Stay):
            return APRLStayWithinSubjectStanAdapter(model)
        if isinstance(base, AP_RL_NoStay):
            return APRLNoStayWithinSubjectStanAdapter(model)
        if isinstance(base, Vicarious_DB_Stay):
            return VicariousDBStayWithinSubjectStanAdapter(model)
        if isinstance(base, VicQ_AP_DualW_Stay):
            return VicQAPDualWStayWithinSubjectStanAdapter(model)
        if isinstance(base, VicQ_AP_DualW_NoStay):
            return VicQAPDualWNoStayWithinSubjectStanAdapter(model)
        raise ValueError(
            f"No within-subject Stan adapter registered for wrapped base model: {type(base).__name__}"
        )

    if isinstance(model, VS):
        return VSStanAdapter(model)
    if isinstance(model, QRL):
        return QRLStanAdapter(model)
    if isinstance(model, Vicarious_RL):
        return VicariousRLStanAdapter(model)
    if isinstance(model, Vicarious_RL_Stay):
        return VicariousRLStayStanAdapter(model)
    if isinstance(model, AP_RL_Stay):
        return APRLStayStanAdapter(model)
    if isinstance(model, AP_RL_NoStay):
        return APRLNoStayStanAdapter(model)
    if isinstance(model, Vicarious_AP_VS):
        return VicariousAPVSStanAdapter(model)
    if isinstance(model, Vicarious_AP_DB_STAY):
        return VicariousAPDBStayStanAdapter(model)
    if isinstance(model, Vicarious_Dir_DB_Stay):
        return VicariousDirDBStayStanAdapter(model)
    if isinstance(model, Vicarious_DB_Stay):
        return VicariousDBStayStanAdapter(model)
    if isinstance(model, Vicarious_VS):
        return VicariousVSStanAdapter(model)
    if isinstance(model, Vicarious_VS_Stay):
        return VicariousVSStayStanAdapter(model)
    if isinstance(model, VicQ_AP_IndepDualW):
        return VicQAPIndepDualWStanAdapter(model)
    if isinstance(model, VicQ_AP_DualW_Stay):
        return VicQAPDualWStayStanAdapter(model)
    if isinstance(model, VicQ_AP_DualW_NoStay):
        return VicQAPDualWNoStayStanAdapter(model)
    raise ValueError(f"No Stan adapter registered for model: {model.__class__.__name__}")
