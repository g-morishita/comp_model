from __future__ import annotations

from ...params import ParameterBoundsSpace
from .schema import vs_schema


def vs_bounds_space(*, beta_max: float = 20.0, kappa_abs_max: float = 5.0) -> ParameterBoundsSpace:
    return vs_schema(beta_max=float(beta_max), kappa_abs_max=float(kappa_abs_max)).bounds_space()
