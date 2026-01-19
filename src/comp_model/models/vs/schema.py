from __future__ import annotations

from ...params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def vs_schema(
    *,
    alpha_p_default: float = 0.2,
    alpha_i_default: float = 0.2,
    beta_default: float = 3.0,
    kappa_default: float = 0.0,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
) -> ParameterSchema:
    return ParameterSchema(
        params=(
            ParamDef("alpha_p", float(alpha_p_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("alpha_i", float(alpha_i_default), Bound(0.0, 1.0), transform=Sigmoid()),
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
            ParamDef("kappa", float(kappa_default), Bound(-float(kappa_abs_max), float(kappa_abs_max)),
                     transform=BoundedTanh(-float(kappa_abs_max), float(kappa_abs_max))),
        )
    )
