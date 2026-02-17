"""Parameter schema for the QRL model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def qrl_schema(
    *,
    alpha_default: float = 0.2,
    beta_default: float = 5.0,
) -> ParameterSchema:

    """
    Construct the QRL parameter schema.

    Parameters
    ----------
    alpha_default : float, optional
        Default learning rate.
    beta_default : float, optional
        Default inverse temperature.
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            # alpha in (0,1)
            ParamDef("alpha", float(alpha_default), Bound(0.0, 1.0), transform=Sigmoid()),
            # beta in (1e-6, +inf) (bounded to avoid crazy temps)
            ParamDef("beta", float(beta_default), Bound(1e-6, float('inf')),
                     transform=LowerBoundedSoftplus(1e-6)),
        )
    )
