from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import vicarious_rl_schema


def vicarious_rl_bounds_space(*, beta_max: float = 20.0) -> ParameterBoundsSpace:
    return vicarious_rl_schema(beta_max=float(beta_max)).bounds_space()
