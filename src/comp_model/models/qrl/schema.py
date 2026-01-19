from __future__ import annotations

from ...params import Bound, ParamDef, ParameterSchema, Sigmoid, BoundedTanh


def qrl_schema(
    *,
    alpha_default: float = 0.2,
    beta_default: float = 5.0,
    beta_max: float = 20.0,
) -> ParameterSchema:
    return ParameterSchema(
        params=(
            # alpha in (0,1)
            ParamDef("alpha", float(alpha_default), Bound(0.0, 1.0), transform=Sigmoid()),
            # beta in (1e-6, beta_max) (bounded to avoid crazy temps)
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max)),
                     transform=BoundedTanh(1e-6, float(beta_max))),
        )
    )
