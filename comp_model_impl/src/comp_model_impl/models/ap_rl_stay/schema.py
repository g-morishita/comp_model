"""Parameter schema for AP_RL_Stay."""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, LowerBoundedSoftplus, ParamDef, ParameterSchema, Sigmoid


def ap_rl_stay_schema(
    *,
    alpha_a_default: float = 0.2,
    beta_default: float = 6.0,
    kappa_default: float = 0.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:
    """Construct the AP_RL_Stay parameter schema."""

    return ParameterSchema(
        params=(
            ParamDef("alpha_a", float(alpha_a_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef(
                "beta",
                float(beta_default),
                Bound(1e-6, float("inf")),
                transform=LowerBoundedSoftplus(1e-6),
            ),
            ParamDef(
                "kappa",
                float(kappa_default),
                Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max)),
            ),
        )
    )
