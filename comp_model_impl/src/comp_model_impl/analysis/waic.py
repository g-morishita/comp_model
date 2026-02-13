"""WAIC utilities from posterior pointwise log-likelihood draws.

This module provides a small, dependency-light implementation of WAIC
(Watanabe-Akaike information criterion) for Bayesian model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WAICSummary:
    """WAIC summary statistics.

    Attributes
    ----------
    waic : float
        WAIC value where lower is better.
    elpd_waic : float
        Expected log predictive density under WAIC.
    p_waic : float
        Effective number of parameters (WAIC penalty term).
    lppd : float
        Log pointwise predictive density.
    n_obs : int
        Number of pointwise observations used.
    """

    waic: float
    elpd_waic: float
    p_waic: float
    lppd: float
    n_obs: int

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "waic": float(self.waic),
            "elpd_waic": float(self.elpd_waic),
            "p_waic": float(self.p_waic),
            "lppd": float(self.lppd),
            "n_obs": float(self.n_obs),
        }


def _log_mean_exp(a: np.ndarray, axis: int) -> np.ndarray:
    """Compute log(mean(exp(a))) in a numerically stable way."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def compute_waic_from_log_lik_draws(
    log_lik_draws: Any,
    *,
    draw_axis: int = 0,
) -> WAICSummary:
    """Compute WAIC from posterior draws of pointwise log-likelihood.

    Parameters
    ----------
    log_lik_draws : Any
        Array-like object containing posterior draws of pointwise log-likelihood.
        Expected shape is ``(n_draws, n_obs)`` by default, but additional
        observation dimensions are allowed and will be flattened.
    draw_axis : int, optional
        Axis corresponding to posterior draws.

    Returns
    -------
    WAICSummary
        WAIC summary statistics.

    Notes
    -----
    Let ``log_lik[s, i]`` be draw ``s`` and observation ``i``.

    - ``lppd_i = log(mean_s exp(log_lik[s, i]))``
    - ``p_waic_i = Var_s(log_lik[s, i])``
    - ``elpd_waic = sum_i (lppd_i - p_waic_i)``
    - ``waic = -2 * elpd_waic``
    """
    arr = np.asarray(log_lik_draws, dtype=float)
    if arr.ndim < 2:
        raise ValueError(
            "log_lik_draws must have at least 2 dimensions: draws x observations."
        )

    arr = np.moveaxis(arr, int(draw_axis), 0)
    n_draws = int(arr.shape[0])
    if n_draws <= 0:
        raise ValueError("log_lik_draws has zero draws.")

    pointwise = arr.reshape(n_draws, -1)
    if pointwise.shape[1] <= 0:
        raise ValueError("log_lik_draws has zero observations after flattening.")
    if not np.all(np.isfinite(pointwise)):
        raise ValueError("log_lik_draws contains non-finite values.")

    lppd_i = _log_mean_exp(pointwise, axis=0)
    ddof = 1 if n_draws > 1 else 0
    p_waic_i = np.var(pointwise, axis=0, ddof=ddof)

    lppd = float(np.sum(lppd_i))
    p_waic = float(np.sum(p_waic_i))
    elpd_waic = float(lppd - p_waic)
    waic = float(-2.0 * elpd_waic)

    return WAICSummary(
        waic=waic,
        elpd_waic=elpd_waic,
        p_waic=p_waic,
        lppd=lppd,
        n_obs=int(pointwise.shape[1]),
    )
