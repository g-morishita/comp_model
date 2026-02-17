"""Parameter schema for the VS model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def vs_schema(
    *,
    alpha_p_default: float = 0.2,
    alpha_i_default: float = 0.2,
    beta_default: float = 3.0,
    kappa_default: float = 0.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:

    """
    Construct the VS parameter schema.

    Parameters
    ----------
    alpha_p_default : float, optional
        Default private learning rate.
    alpha_i_default : float, optional
        Default social learning rate.
    beta_default : float, optional
        Default inverse temperature.
    kappa_default : float, optional
        Default perseveration parameter.
    kappa_abs_max : float, optional
        Maximum absolute kappa.
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            ParamDef("alpha_p", float(alpha_p_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_i", float(alpha_i_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("beta", float(beta_default), Bound(1e-6, float("inf")),
                     transform=LowerBoundedSoftplus(1e-6)),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
        )
    )
