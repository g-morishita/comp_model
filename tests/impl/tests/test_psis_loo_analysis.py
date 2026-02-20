"""Tests for PSIS-LOO utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_impl.analysis.psis_loo import compute_psis_loo_from_log_lik_draws


def test_compute_psis_loo_from_log_lik_draws_returns_finite_summary() -> None:
    """PSIS-LOO summary should be finite for well-behaved draws."""
    draws = np.array(
        [
            [-1.0, -2.0, -0.5],
            [-1.2, -1.8, -0.7],
            [-0.8, -2.3, -0.4],
            [-1.1, -2.1, -0.6],
        ],
        dtype=float,
    )
    out = compute_psis_loo_from_log_lik_draws(draws)

    assert np.isfinite(out.looic)
    assert np.isfinite(out.elpd_loo)
    assert np.isfinite(out.p_loo)
    assert np.isfinite(out.pareto_k_max)
    assert out.n_obs == 3
    assert out.looic == pytest.approx(-2.0 * out.elpd_loo)


def test_compute_psis_loo_from_log_lik_draws_respects_draw_axis() -> None:
    """draw_axis should work for non-leading draw dimensions."""
    base = np.array(
        [
            [-1.0, -2.0],
            [-1.1, -2.1],
            [-0.9, -1.9],
            [-1.05, -2.05],
        ],
        dtype=float,
    )  # draws x obs

    out0 = compute_psis_loo_from_log_lik_draws(base, draw_axis=0)
    out1 = compute_psis_loo_from_log_lik_draws(base.T, draw_axis=1)

    assert out0.looic == pytest.approx(out1.looic)
    assert out0.elpd_loo == pytest.approx(out1.elpd_loo)
    assert out0.p_loo == pytest.approx(out1.p_loo)
    assert out0.n_obs == out1.n_obs == 2


def test_compute_psis_loo_from_log_lik_draws_rejects_invalid_inputs() -> None:
    """Invalid shapes and non-finite values should raise clear errors."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        _ = compute_psis_loo_from_log_lik_draws(np.array([1.0, 2.0, 3.0]))

    bad = np.array([[0.0, np.nan], [0.0, -1.0]], dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        _ = compute_psis_loo_from_log_lik_draws(bad)

