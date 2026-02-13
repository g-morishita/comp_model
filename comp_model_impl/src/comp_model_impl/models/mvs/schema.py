"""Parameter schema for the MVS lottery-choice model."""

from __future__ import annotations

from comp_model_core.params import Bound, BoundedTanh, ParamDef, ParameterSchema


def mvs_schema(
    *,
    lambda_var_default: float = -0.25,
    delta_default: float = 0.2,
    beta_default: float = 2.5,
    lambda_abs_max: float = 10.0,
    delta_abs_max: float = 10.0,
    beta_max: float = 20.0,
) -> ParameterSchema:
    """Construct a parameter schema for the MVS utility model."""
    return ParameterSchema(
        params=(
            ParamDef(
                "lambda_var",
                float(lambda_var_default),
                Bound(-float(lambda_abs_max), float(lambda_abs_max)),
                transform=BoundedTanh(-float(lambda_abs_max), float(lambda_abs_max)),
            ),
            ParamDef(
                "delta",
                float(delta_default),
                Bound(-float(delta_abs_max), float(delta_abs_max)),
                transform=BoundedTanh(-float(delta_abs_max), float(delta_abs_max)),
            ),
            ParamDef(
                "beta",
                float(beta_default),
                Bound(1e-6, float(beta_max)),
                transform=BoundedTanh(1e-6, float(beta_max)),
            ),
        )
    )
