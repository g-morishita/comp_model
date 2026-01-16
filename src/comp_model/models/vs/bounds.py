from __future__ import annotations

from ...params.bounds import Bound, ParameterBoundsSpace


def vs_bounds_space(
    *,
    beta_max: float = 20.0,
    kappa_abs_max: float = 5.0,
) -> ParameterBoundsSpace:
    """
    VS parameter bounds.

    You should tune beta_max and kappa range to your task scale.
    """
    names = ("alpha_p", "alpha_i", "beta", "kappa")
    bounds = {
        "alpha_p": Bound(0.0, 1.0),
        "alpha_i": Bound(0.0, 1.0),
        "beta": Bound(1e-6, float(beta_max)),
        "kappa": Bound(-float(kappa_abs_max), float(kappa_abs_max)),
    }
    return ParameterBoundsSpace(names=names, bounds=bounds)
