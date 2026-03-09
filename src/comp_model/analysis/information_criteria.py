"""Information-criterion utilities for model comparison."""

from __future__ import annotations

import math


def aic(*, log_likelihood: float, n_parameters: int) -> float:
    """Compute Akaike Information Criterion (AIC)."""

    return float(2.0 * float(n_parameters) - 2.0 * float(log_likelihood))


def bic(*, log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute Bayesian Information Criterion (BIC)."""

    if n_observations <= 0:
        raise ValueError("n_observations must be > 0")
    return float(math.log(float(n_observations)) * float(n_parameters) - 2.0 * float(log_likelihood))


__all__ = ["aic", "bic"]
