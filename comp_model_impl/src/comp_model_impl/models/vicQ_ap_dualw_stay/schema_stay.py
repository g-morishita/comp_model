"""Parameter schema for the mixture dual-weight VicQ+AP stay model."""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def vicQ_ap_dualw_stay_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_default: float = 6.0,
    w_default: float = 0.5,
    kappa_default: float = 2.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:
    """Construct the VicQ_AP_DualW_Stay (beta-mix) parameter schema."""

    return ParameterSchema(
        params=(
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef(
                "beta",
                float(beta_default),
                Bound(1e-6, float("inf")),
                transform=LowerBoundedSoftplus(1e-6),
            ),
            ParamDef("w", float(w_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef(
                "kappa",
                float(kappa_default),
                Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max)),
            ),
        )
    )
