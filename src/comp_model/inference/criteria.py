"""Shared information-criterion helpers for inference outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from comp_model.analysis.information_criteria import psis_loo, waic


def pointwise_log_likelihood_draws_from_fit_result(fit_result: Any) -> np.ndarray:
    """Extract pointwise log-likelihood draws from a fit result.

    Parameters
    ----------
    fit_result : Any
        Inference fit result object.

    Returns
    -------
    numpy.ndarray
        Draw matrix with shape ``(n_draws, n_observations)``.

    Raises
    ------
    TypeError
        If the fit result does not provide pointwise draws.
    ValueError
        If the stored draws are malformed.
    """

    draws = getattr(fit_result, "pointwise_log_likelihood_draws", None)
    if draws is None:
        raise TypeError(
            "fit_result does not provide pointwise log-likelihood draws "
            "required for WAIC/PSIS-LOO"
        )

    array = np.asarray(draws, dtype=float)
    if array.ndim != 2:
        raise ValueError("pointwise_log_likelihood_draws must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError("pointwise_log_likelihood_draws must have at least two draws")
    if array.shape[1] < 1:
        raise ValueError("pointwise_log_likelihood_draws must include at least one observation")
    if not np.all(np.isfinite(array)):
        raise ValueError("pointwise_log_likelihood_draws must contain only finite values")
    return array


def compute_pointwise_information_criteria(fit_result: Any) -> tuple[float, float]:
    """Compute WAIC and PSIS-LOO IC values from a fit result.

    Parameters
    ----------
    fit_result : Any
        Inference fit result object with pointwise log-likelihood draws.

    Returns
    -------
    tuple[float, float]
        ``(waic, looic)`` information criteria values.
    """

    draws = pointwise_log_likelihood_draws_from_fit_result(fit_result)
    waic_value = float(waic(draws).waic)
    looic_value = float(psis_loo(draws).looic)
    return waic_value, looic_value


__all__ = [
    "compute_pointwise_information_criteria",
    "pointwise_log_likelihood_draws_from_fit_result",
]
