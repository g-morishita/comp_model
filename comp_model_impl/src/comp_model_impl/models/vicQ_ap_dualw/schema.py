"""Parameter schema for the mixture dual-weight VicQ+AP model."""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, ParamDef, ParameterSchema, Sigmoid


def vicQ_ap_dualw_schema(
    *,
    alpha_o_default: float = 0.2,
    alpha_a_default: float = 0.2,
    beta_default: float = 6.0,
    w_default: float = 0.5,
    kappa_default: float = 2.0,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:
    """Construct the VicQ_AP_DualW (beta-mix) parameter schema."""

    return ParameterSchema(
        params=(
            ParamDef("alpha_o", float(alpha_o_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef(
                "beta",
                float(beta_default),
                Bound(1e-6, float(beta_max)),
                transform=BoundedTanh(1e-6, float(beta_max)),
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
