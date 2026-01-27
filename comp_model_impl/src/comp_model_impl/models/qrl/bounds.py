"""Convenience constructors for QRL parameter bounds.
"""

from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import qrl_schema


def qrl_bounds_space(*, beta_max: float = 20.0) -> ParameterBoundsSpace:

    """
    Return a bounds space for QRL parameters.
    
    Parameters
    ----------
    beta_max : float, optional
        Upper bound for the inverse temperature.
    
    Returns
    -------
    comp_model_core.params.ParameterBoundsSpace
        Bounds space consistent with :func:`~comp_model_impl.models.qrl.schema.qrl_schema`.
    """

    return qrl_schema(beta_max=float(beta_max)).bounds_space()
