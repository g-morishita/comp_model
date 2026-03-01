"""Tests for information-criterion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.analysis.information_criteria import (
    PSISLOOResult,
    WAICResult,
    aic,
    bic,
    psis_loo,
    waic,
)


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


def test_psis_loo_returns_result_for_valid_draw_matrix() -> None:
    """PSIS-LOO should return finite decomposition for valid draw matrix."""

    draws = np.array(
        [
            [-0.9, -1.1, -0.7],
            [-1.0, -1.0, -0.8],
            [-0.8, -1.2, -0.6],
            [-0.95, -1.05, -0.75],
            [-0.85, -1.15, -0.72],
        ],
        dtype=float,
    )

    result = psis_loo(draws)
    assert isinstance(result, PSISLOOResult)
    assert np.isfinite(result.looic)
    assert np.isfinite(result.elpd_loo)
    assert np.isfinite(result.p_loo)
    assert result.pareto_k.shape == (3,)


def test_psis_loo_matches_constant_log_likelihood_case() -> None:
    """PSIS-LOO should reduce to exact constant-draw identity."""

    draws = np.array(
        [
            [-1.0, -2.0],
            [-1.0, -2.0],
            [-1.0, -2.0],
            [-1.0, -2.0],
            [-1.0, -2.0],
        ],
        dtype=float,
    )

    result = psis_loo(draws)
    assert result.elpd_loo == pytest.approx(-3.0)
    assert result.looic == pytest.approx(6.0)
    assert result.p_loo == pytest.approx(0.0)


def test_psis_loo_rejects_invalid_inputs() -> None:
    """PSIS-LOO should validate draw matrix and smoothing settings."""

    with pytest.raises(ValueError, match="2D array"):
        psis_loo(np.asarray([0.1, 0.2, 0.3], dtype=float))

    with pytest.raises(ValueError, match="tail_fraction must lie in"):
        psis_loo(np.zeros((4, 2), dtype=float), tail_fraction=1.0)

    with pytest.raises(ValueError, match="min_tail_draws must be >= 3"):
        psis_loo(np.zeros((4, 2), dtype=float), min_tail_draws=2)

    with pytest.raises(ValueError, match="finite values"):
        bad = np.zeros((4, 2), dtype=float)
        bad[0, 0] = np.inf
        psis_loo(bad)
