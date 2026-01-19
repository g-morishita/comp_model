from __future__ import annotations

from ...params import Bound, ParamDef, ParameterSchema


def qrl_schema(
    *,
    alpha_default: float = 0.2,
    beta_default: float = 5.0,
    beta_max: float = 20.0,
) -> ParameterSchema:
    return ParameterSchema(
        params=(
            ParamDef("alpha", float(alpha_default), Bound(0.0, 1.0)),
            ParamDef("beta", float(beta_default), Bound(1e-6, float(beta_max))),
        )
    )
