"""Convenience constructors for Vicarious_VS parameter bounds.
"""

from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import vicarious_vs_stay_schema


def vicarious_vs_stay_bounds_space(*, kappa_abs_max: float = 1.0, beta_max: float = 20.0) -> ParameterBoundsSpace:

    """
    Return a bounds space for Vicarious_VS_Stay parameters.
    
    Parameters
    ----------
    beta_max : float, optional
        Upper bound for the inverse temperature.
    
    Returns
    -------
    comp_model_core.params.ParameterBoundsSpace
        Bounds space consistent with :func:`~comp_model_impl.models.vicarious_vs.schema.vicarious_vs_schema`.
    """

    return vicarious_vs_stay_schema(kappa_abs_max=float(kappa_abs_max), beta_max=float(beta_max)).bounds_space()
