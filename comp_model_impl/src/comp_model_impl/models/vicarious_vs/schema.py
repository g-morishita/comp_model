"""Parameter schema for the Vicarious_VS model.
"""

from __future__ import annotations

from comp_model_core.params import Bound, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def vicarious_vs_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_default: float = 3.0,
) -> ParameterSchema:

    """
    Construct the Vicarious_VS parameter schema.
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("beta", float(beta_default), Bound(1e-6, float("inf")),
                     transform=LowerBoundedSoftplus(1e-6)),
        )
    )
