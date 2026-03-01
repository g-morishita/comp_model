"""Information-criterion utilities for model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True, slots=True)
class WAICResult:
    """Result container for WAIC computation.

    Parameters
    ----------
    waic : float
        Widely Applicable Information Criterion value (lower is better).
    lppd : float
        Log pointwise predictive density term.
    p_waic : float
        Effective number of parameters under WAIC.
    """

    waic: float
    lppd: float
    p_waic: float


@dataclass(frozen=True, slots=True)
class PSISLOOResult:
    """Result container for PSIS-LOO computation.

    Parameters
    ----------
    looic : float
        Leave-one-out information criterion value (lower is better).
    elpd_loo : float
        Expected log pointwise predictive density under LOO.
    p_loo : float
        Effective number of parameters under PSIS-LOO.
    pareto_k : numpy.ndarray
        Pareto shape diagnostics per observation.
    """

    looic: float
    elpd_loo: float
    p_loo: float
    pareto_k: np.ndarray


def aic(*, log_likelihood: float, n_parameters: int) -> float:
    """Compute Akaike Information Criterion (AIC).

    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood.
    n_parameters : int
        Number of effective free parameters.

    Returns
    -------
    float
        AIC value.
    """

    return float(2.0 * float(n_parameters) - 2.0 * float(log_likelihood))


def bic(*, log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute Bayesian Information Criterion (BIC).

    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood.
    n_parameters : int
        Number of effective free parameters.
    n_observations : int
        Number of independent observations used in fitting.

    Returns
    -------
    float
        BIC value.

    Raises
    ------
    ValueError
        If ``n_observations`` is non-positive.
    """

    if n_observations <= 0:
        raise ValueError("n_observations must be > 0")

    return float(np.log(float(n_observations)) * float(n_parameters) - 2.0 * float(log_likelihood))


def _coerce_log_likelihood_draws(log_likelihood_draws: np.ndarray) -> np.ndarray:
    """Validate and coerce posterior pointwise log-likelihood draws."""

    draws = np.asarray(log_likelihood_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("log_likelihood_draws must be a 2D array")
    if draws.shape[0] < 2:
        raise ValueError("log_likelihood_draws must contain at least two posterior draws")
    if draws.shape[1] < 1:
        raise ValueError("log_likelihood_draws must include at least one observation")
    if not np.all(np.isfinite(draws)):
        raise ValueError("log_likelihood_draws must contain only finite values")
    return draws


def _compute_lppd(draws: np.ndarray) -> float:
    """Compute log pointwise predictive density from draw matrix."""

    max_per_obs = np.max(draws, axis=0)
    stabilized = np.exp(draws - max_per_obs)
    return float(np.sum(max_per_obs + np.log(np.mean(stabilized, axis=0))))


def waic(log_likelihood_draws: np.ndarray) -> WAICResult:
    """Compute WAIC from posterior log-likelihood draws.

    Parameters
    ----------
    log_likelihood_draws : numpy.ndarray
        Array with shape ``(n_draws, n_observations)`` containing pointwise
        log-likelihood values for posterior draws.

    Returns
    -------
    WAICResult
        WAIC decomposition and criterion value.

    Raises
    ------
    ValueError
        If input array is not 2D or has invalid dimensions.
    """

    draws = _coerce_log_likelihood_draws(log_likelihood_draws)
    lppd = _compute_lppd(draws)

    # p_waic is summed posterior variance of pointwise log-likelihood.
    p_waic = float(np.sum(np.var(draws, axis=0, ddof=1)))
    waic_value = float(-2.0 * (lppd - p_waic))

    return WAICResult(waic=waic_value, lppd=lppd, p_waic=p_waic)

def _truncate_weights(weights: np.ndarray) -> np.ndarray:
    """Apply standard PSIS weight truncation."""

    mean_weight = float(np.mean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        return weights
    cap = mean_weight * (weights.size ** 0.75)
    return np.minimum(weights, cap)


def _smooth_tail_weights(
    weights: np.ndarray,
    *,
    tail_count: int,
) -> tuple[np.ndarray, float]:
    """Smooth upper-tail importance weights with a generalized Pareto fit."""

    if tail_count < 3 or tail_count >= weights.size:
        return _truncate_weights(weights), float("nan")

    sorted_indices = np.argsort(weights)
    tail_indices_unsorted = sorted_indices[-tail_count:]
    tail_indices = tail_indices_unsorted[np.argsort(weights[tail_indices_unsorted])]
    tail_values = weights[tail_indices]

    threshold = float(tail_values[0])
    excesses = tail_values - threshold
    if np.allclose(excesses, 0.0):
        return _truncate_weights(weights.copy()), 0.0

    try:
        shape, _, scale = genpareto.fit(excesses, floc=0.0)
    except Exception:  # pragma: no cover - defensive fallback for optimizer failures
        return _truncate_weights(weights.copy()), float("nan")

    if not np.isfinite(shape) or not np.isfinite(scale) or float(scale) <= 0.0:
        return _truncate_weights(weights.copy()), float("nan")

    probs = (np.arange(1, tail_count + 1, dtype=float) - 0.5) / float(tail_count)
    smoothed_tail = threshold + genpareto.ppf(probs, c=float(shape), loc=0.0, scale=float(scale))
    if not np.all(np.isfinite(smoothed_tail)):
        return _truncate_weights(weights.copy()), float("nan")

    smoothed = weights.copy()
    smoothed[tail_indices] = smoothed_tail
    return _truncate_weights(smoothed), float(shape)


def psis_loo(
    log_likelihood_draws: np.ndarray,
    *,
    tail_fraction: float = 0.2,
    min_tail_draws: int = 5,
) -> PSISLOOResult:
    """Compute Pareto-smoothed importance-sampling leave-one-out criterion.

    Parameters
    ----------
    log_likelihood_draws : numpy.ndarray
        Array with shape ``(n_draws, n_observations)`` containing pointwise
        log-likelihood values for posterior draws.
    tail_fraction : float, optional
        Fraction of largest raw importance ratios to smooth per observation.
    min_tail_draws : int, optional
        Minimum number of tail points for Pareto fitting when possible.

    Returns
    -------
    PSISLOOResult
        LOOIC decomposition and Pareto diagnostics.

    Raises
    ------
    ValueError
        If draws or smoothing configuration are invalid.
    """

    if tail_fraction <= 0.0 or tail_fraction >= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    if min_tail_draws < 3:
        raise ValueError("min_tail_draws must be >= 3")

    draws = _coerce_log_likelihood_draws(log_likelihood_draws)
    n_draws, n_observations = draws.shape
    lppd = _compute_lppd(draws)

    can_smooth = n_draws >= 4
    tail_count = 0
    if can_smooth:
        tail_count = int(np.ceil(float(n_draws) * float(tail_fraction)))
        tail_count = max(tail_count, int(min_tail_draws))
        tail_count = min(tail_count, n_draws - 1)

    elpd_points = np.zeros(n_observations, dtype=float)
    pareto_k = np.full(n_observations, np.nan, dtype=float)

    for observation_index in range(n_observations):
        raw_log_ratios = -draws[:, observation_index]
        max_log_ratio = float(np.max(raw_log_ratios))
        log_ratios = raw_log_ratios - max_log_ratio
        weights = np.exp(log_ratios)

        if can_smooth:
            smoothed_weights, k_value = _smooth_tail_weights(
                weights,
                tail_count=tail_count,
            )
            pareto_k[observation_index] = float(k_value)
        else:
            smoothed_weights = _truncate_weights(weights)

        mean_weight = float(np.mean(smoothed_weights))
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            raise ValueError("log_likelihood_draws produced invalid importance weights")
        elpd_points[observation_index] = -(np.log(mean_weight) + max_log_ratio)

    elpd_loo = float(np.sum(elpd_points))
    p_loo = float(lppd - elpd_loo)
    looic = float(-2.0 * elpd_loo)
    return PSISLOOResult(looic=looic, elpd_loo=elpd_loo, p_loo=p_loo, pareto_k=pareto_k)


__all__ = ["PSISLOOResult", "WAICResult", "aic", "bic", "psis_loo", "waic"]
