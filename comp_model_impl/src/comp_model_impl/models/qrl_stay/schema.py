"""Parameter schema for QRL_Stay."""

from __future__ import annotations

from comp_model_core.params import (
    Bound,
    LowerBoundedSoftplus,
    ParamDef,
    ParameterSchema,
    Sigmoid,
)


def qrl_stay_schema(
    *,
    alpha_default: float = 0.2,
    beta_default: float = 5.0,
    kappa_default: float = 1.0,
    kappa_abs_max: float = float("inf"),
) -> ParameterSchema:
    """Construct the QRL_Stay parameter schema."""

    return ParameterSchema(
        params=(
            # alpha in (0,1)
            ParamDef("alpha", float(alpha_default), Bound(0.0, 1.0), transform=Sigmoid()),
            # beta in (1e-6, +inf)
            ParamDef(
                "beta",
                float(beta_default),
                Bound(1e-6, float("inf")),
                transform=LowerBoundedSoftplus(1e-6),
            ),
            # kappa in [-kappa_abs_max, +kappa_abs_max]; if abs max is inf this is unbounded.
            ParamDef(
                "kappa",
                float(kappa_default),
                Bound(-float(kappa_abs_max), float(kappa_abs_max)),
            ),
        )
    )
