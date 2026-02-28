"""Tests for information-criterion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.analysis.information_criteria import WAICResult, aic, bic, waic


def test_aic_and_bic_match_closed_form() -> None:
    """AIC/BIC helpers should match direct formulas."""

    log_likelihood = -12.5
    n_parameters = 3
    n_observations = 40

    assert aic(log_likelihood=log_likelihood, n_parameters=n_parameters) == pytest.approx(31.0)
    expected_bic = np.log(float(n_observations)) * n_parameters - 2.0 * log_likelihood
    assert bic(
        log_likelihood=log_likelihood,
        n_parameters=n_parameters,
        n_observations=n_observations,
    ) == pytest.approx(expected_bic)


def test_bic_rejects_non_positive_observations() -> None:
    """BIC should reject invalid observation count."""

    with pytest.raises(ValueError, match="n_observations must be > 0"):
        bic(log_likelihood=-1.0, n_parameters=1, n_observations=0)


def test_waic_returns_result_for_valid_draw_matrix() -> None:
    """WAIC should return finite decomposition for a valid draw matrix."""

    draws = np.array(
        [
            [-0.9, -1.1, -0.7],
            [-1.0, -1.0, -0.8],
            [-0.8, -1.2, -0.6],
            [-0.95, -1.05, -0.75],
        ],
        dtype=float,
    )

    result = waic(draws)

    assert isinstance(result, WAICResult)
    assert np.isfinite(result.waic)
    assert np.isfinite(result.lppd)
    assert np.isfinite(result.p_waic)
    assert result.p_waic >= 0.0


def test_waic_rejects_invalid_shapes() -> None:
    """WAIC should validate draw matrix shape constraints."""

    with pytest.raises(ValueError, match="2D array"):
        waic(np.asarray([0.1, 0.2, 0.3], dtype=float))

    with pytest.raises(ValueError, match="at least two posterior draws"):
        waic(np.asarray([[-1.0, -0.9]], dtype=float))

    with pytest.raises(ValueError, match="at least one observation"):
        waic(np.zeros((3, 0), dtype=float))
