from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import vicarious_vs_schema


def vicarious_vs_bounds_space(*, beta_max: float = 20.0) -> ParameterBoundsSpace:
    return vicarious_vs_schema(beta_max=float(beta_max)).bounds_space()
