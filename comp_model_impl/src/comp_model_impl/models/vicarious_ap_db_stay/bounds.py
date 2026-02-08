"""Convenience constructors for Vicarious AP-VS parameter bounds.
"""

from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import vicarious_db_vs_stay_schema


def vicarious_db_vs_stay_bounds_space(
    *,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
    demo_bias_rel_abs_max: float = 5.0,
) -> ParameterBoundsSpace:
   """
    Return a bounds space for Vicarious AP-VS parameters.

    Parameters
    ----------
    beta_max : float, optional
        Upper bound for the inverse temperature parameter.
    kappa_abs_max : float, optional
        Absolute bound for the perseveration parameter.
    demo_bias_rel_abs_max : float, optional
        Absolute bound for the reliability-modulated demonstrator-choice bias parameter.

    Returns
    -------
comp_model_core.params.ParameterBoundsSpace
        Bounds space consistent with :func:`~comp_model_impl.models.vs.schema.vs_schema`.
    """
   return vicarious_db_vs_stay_schema(
       beta_max=float(beta_max),
       kappa_abs_max=float(kappa_abs_max),
       demo_bias_rel_abs_max=float(demo_bias_rel_abs_max),
   ).bounds_space()
