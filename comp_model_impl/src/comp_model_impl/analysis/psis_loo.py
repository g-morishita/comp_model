"""PSIS-LOO utilities from posterior pointwise log-likelihood draws.

This module provides a small implementation of PSIS-LOO
(Pareto-smoothed importance-sampling leave-one-out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True, slots=True)
class PSISLOOSummary:
    """PSIS-LOO summary statistics.

    Attributes
    ----------
    looic : float
        LOO information criterion (lower is better), ``-2 * elpd_loo``.
    elpd_loo : float
        Expected log predictive density under PSIS-LOO.
    p_loo : float
        Effective number of parameters under PSIS-LOO.
    lppd : float
        Log pointwise predictive density under the full posterior.
    n_obs : int
        Number of pointwise observations used.
    pareto_k_max : float
        Maximum estimated Pareto shape ``k`` across observations.
    """

    looic: float
    elpd_loo: float
    p_loo: float
    lppd: float
    n_obs: int
    pareto_k_max: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "looic": float(self.looic),
            "elpd_loo": float(self.elpd_loo),
            "p_loo": float(self.p_loo),
            "lppd": float(self.lppd),
            "n_obs": float(self.n_obs),
            "pareto_k_max": float(self.pareto_k_max),
        }


def _log_mean_exp(a: np.ndarray, axis: int) -> np.ndarray:
    """Compute ``log(mean(exp(a)))`` in a numerically stable way."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _psis_smoothed_shifted_weights(log_ratios: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return PSIS-smoothed shifted importance weights for one observation.

    Parameters
    ----------
    log_ratios : np.ndarray
        Log importance ratios for one observation (1D, finite).

    Returns
    -------
    tuple[np.ndarray, float, float]
        ``(weights_shifted, pareto_k, shift)``, where ``weights_shifted`` are
        importance weights after subtracting ``shift = max(log_ratios)``.
    """
    x = np.asarray(log_ratios, dtype=float)
    if x.ndim != 1:
        raise ValueError("log_ratios must be 1D.")
    if x.size == 0:
        raise ValueError("log_ratios must be non-empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("log_ratios contains non-finite values.")

    n = int(x.size)
    shift = float(np.max(x))
    w = np.exp(x - shift)

    if n < 4:
        return w, 0.0, shift

    tail_len = min(max(int(np.ceil(0.2 * n)), 1), n - 1)
    if tail_len <= 0:
        return w, 0.0, shift

    order = np.argsort(w)
    w_sorted = w[order].astype(float, copy=True)
    threshold = float(w_sorted[-tail_len - 1])
    tail = w_sorted[-tail_len:]
    excess = tail - threshold

    pareto_k = 0.0
    if np.any(excess > 0.0):
        try:
            c_hat, _, scale_hat = genpareto.fit(excess, floc=0.0)
            if np.isfinite(c_hat) and np.isfinite(scale_hat) and float(scale_hat) > 0.0:
                q = (np.arange(1, tail_len + 1, dtype=float) - 0.5) / float(tail_len)
                smooth_excess = genpareto.ppf(q, c=float(c_hat), loc=0.0, scale=float(scale_hat))
                if np.all(np.isfinite(smooth_excess)):
                    w_sorted[-tail_len:] = threshold + smooth_excess
                    pareto_k = float(c_hat)
        except Exception:  # noqa: BLE001
            pass

    # Standard truncation to stabilize variance in finite samples.
    mean_w = float(np.mean(w_sorted))
    if np.isfinite(mean_w) and mean_w > 0.0:
        cap = mean_w * (float(n) ** 0.75)
        w_sorted = np.minimum(w_sorted, cap)

    out = np.empty_like(w_sorted)
    out[order] = w_sorted
    return out, float(pareto_k), shift


def compute_psis_loo_from_log_lik_draws(
    log_lik_draws: Any,
    *,
    draw_axis: int = 0,
) -> PSISLOOSummary:
    """Compute PSIS-LOO from posterior draws of pointwise log-likelihood.

    Parameters
    ----------
    log_lik_draws : Any
        Array-like posterior pointwise log-likelihood draws.
        Default shape is ``(n_draws, n_obs)``; extra observation dimensions are
        flattened.
    draw_axis : int, optional
        Axis corresponding to posterior draws.

    Returns
    -------
    PSISLOOSummary
        PSIS-LOO summary statistics.
    """
    arr = np.asarray(log_lik_draws, dtype=float)
    if arr.ndim < 2:
        raise ValueError("log_lik_draws must have at least 2 dimensions: draws x observations.")

    arr = np.moveaxis(arr, int(draw_axis), 0)
    n_draws = int(arr.shape[0])
    if n_draws <= 0:
        raise ValueError("log_lik_draws has zero draws.")

    pointwise = arr.reshape(n_draws, -1)
    if pointwise.shape[1] <= 0:
        raise ValueError("log_lik_draws has zero observations after flattening.")
    if not np.all(np.isfinite(pointwise)):
        raise ValueError("log_lik_draws contains non-finite values.")

    n_obs = int(pointwise.shape[1])
    loo_i = np.empty((n_obs,), dtype=float)
    pareto_k = np.empty((n_obs,), dtype=float)

    for i in range(n_obs):
        # Raw IS ratio for observation i: 1 / p(y_i | theta_s)
        log_r = -pointwise[:, i]
        w_shifted, k_i, shift = _psis_smoothed_shifted_weights(log_r)
        mean_shifted = float(np.mean(w_shifted))
        if not np.isfinite(mean_shifted) or mean_shifted <= 0.0:
            raise ValueError("PSIS weights became non-finite for an observation.")
        loo_i[i] = -(float(shift) + float(np.log(mean_shifted)))
        pareto_k[i] = float(k_i)

    elpd_loo = float(np.sum(loo_i))
    looic = float(-2.0 * elpd_loo)
    lppd = float(np.sum(_log_mean_exp(pointwise, axis=0)))
    p_loo = float(lppd - elpd_loo)
    pareto_k_max = float(np.max(pareto_k)) if pareto_k.size else 0.0

    return PSISLOOSummary(
        looic=looic,
        elpd_loo=elpd_loo,
        p_loo=p_loo,
        lppd=lppd,
        n_obs=n_obs,
        pareto_k_max=pareto_k_max,
    )

