"""Parameter schema for the independent dual-weight VicQ+AP model."""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def vicQ_ap_indep_dualw_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_q_default: float = 5.0,
    beta_a_default: float = 5.0,
    kappa_default: float = 2.0,
    kappa_abs_max: float = 5.0
) -> ParameterSchema:
    """
    Construct the Vicarious_RL parameter schema.
    Returns
    -------
    comp_model_core.params.ParameterSchema
        Schema describing parameter bounds and transforms.
    """

    return ParameterSchema(
        params=(
            # alpha in (0,1)
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            # beta in (1e-6, +inf) (bounded to avoid crazy temps)
            ParamDef("beta_q", float(beta_q_default), Bound(1e-6, float("inf")),
                     transform=LowerBoundedSoftplus(1e-6)),
            ParamDef("beta_a", float(beta_a_default), Bound(1e-6, float("inf")),
                     transform=LowerBoundedSoftplus(1e-6)),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
        )
    )
