"""Parameter schema for the mixture dual-weight VicQ+AP no-stay model."""

from __future__ import annotations

from comp_model_core.params import Bound, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def vicQ_ap_dualw_nostay_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_default: float = 6.0,
    w_default: float = 0.5,
) -> ParameterSchema:
    """Construct the VicQ_AP_DualW_NoStay (beta-mix) parameter schema."""

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
        )
    )
