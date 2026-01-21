from __future__ import annotations

from comp_model_core.params import ParameterBoundsSpace
from .schema import qrl_schema


def qrl_bounds_space(*, beta_max: float = 20.0) -> ParameterBoundsSpace:
    return qrl_schema(beta_max=float(beta_max)).bounds_space()
