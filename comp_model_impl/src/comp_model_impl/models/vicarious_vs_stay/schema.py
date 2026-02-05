"""Parameter schema for the Vicarious_VS model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def vicarious_vs_stay_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    kappa_default: float = 0.3,
    kappa_abs_max: float = 1.0,
    beta_default: float = 3.0,
    beta_max: float = 20.0,
) -> ParameterSchema:

    """
    Construct the Vicarious_VS_Stay parameter schema.
    
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
        )
    )
