from __future__ import annotations

from ...params.bounds import Bound, ParameterBoundsSpace


def qrl_bounds_space(
    *,
    beta_max: float = 20.0,
) -> ParameterBoundsSpace:
    """
    Q-RL parameter bounds.

    You should tune beta_max range to your task scale.
    """
    names = ("alpha", "beta")
    bounds = {
        "alpha": Bound(0.0, 1.0),
        "beta": Bound(1e-6, float(beta_max)),
    }
    return ParameterBoundsSpace(names=names, bounds=bounds)
