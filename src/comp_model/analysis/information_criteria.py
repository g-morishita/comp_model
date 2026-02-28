"""Information-criterion utilities for model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    draws = np.asarray(log_likelihood_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("log_likelihood_draws must be a 2D array")
    if draws.shape[0] < 2:
        raise ValueError("log_likelihood_draws must contain at least two posterior draws")
    if draws.shape[1] < 1:
        raise ValueError("log_likelihood_draws must include at least one observation")

    # lppd_i = log( mean_s exp(log_lik_{s,i}) ), stabilized by subtracting max.
    max_per_obs = np.max(draws, axis=0)
    stabilized = np.exp(draws - max_per_obs)
    lppd = float(np.sum(max_per_obs + np.log(np.mean(stabilized, axis=0))))

    # p_waic is summed posterior variance of pointwise log-likelihood.
    p_waic = float(np.sum(np.var(draws, axis=0, ddof=1)))
    waic_value = float(-2.0 * (lppd - p_waic))

    return WAICResult(waic=waic_value, lppd=lppd, p_waic=p_waic)


__all__ = ["WAICResult", "aic", "bic", "waic"]
