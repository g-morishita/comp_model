"""Convenience constructors for Vicarious_RL parameter bounds.
"""

from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import vicarious_rl_schema


def vicarious_rl_bounds_space(*, beta_max: float = 20.0) -> ParameterBoundsSpace:

    """
    Return a bounds space for Vicarious_RL parameters.
    
    Parameters
    ----------
    beta_max : float, optional
        Upper bound for the inverse temperature.
    
    Returns
    -------
    comp_model_core.params.ParameterBoundsSpace
        Bounds space consistent with :func:`~comp_model_impl.models.vicarious_rl.schema.vicarious_rl_schema`.
    """

    return vicarious_rl_schema(beta_max=float(beta_max)).bounds_space()
