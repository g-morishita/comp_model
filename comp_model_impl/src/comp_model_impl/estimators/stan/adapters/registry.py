"""Stan adapter registry."""

from __future__ import annotations

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicarious_rl.vicarious_rl import Vicarious_RL
from comp_model_impl.models.vicarious_vs.vicarious_vs import Vicarious_VS
from comp_model_impl.models.vs.vs import VS

from .base import StanAdapter
from .vicarious_rl import VicariousRLStanAdapter
from .vicarious_vs import VicariousVSStanAdapter
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
    if isinstance(model, VS):
        return VSStanAdapter(model)
    if isinstance(model, Vicarious_RL):
        return VicariousRLStanAdapter(model)
    if isinstance(model, Vicarious_VS):
        return VicariousVSStanAdapter(model)
    raise ValueError(f"No Stan adapter registered for model: {model.__class__.__name__}")
