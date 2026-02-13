"""Tests for WAIC utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_impl.analysis.waic import compute_waic_from_log_lik_draws


def test_compute_waic_from_log_lik_draws_matches_manual_formula() -> None:
    """WAIC summary should match direct manual computation."""
    draws = np.array(
        [
            [-1.0, -2.0, -0.5],
            [-1.2, -1.8, -0.7],
            [-0.8, -2.3, -0.4],
            [-1.1, -2.1, -0.6],
        ],
        dtype=float,
    )

    out = compute_waic_from_log_lik_draws(draws)

    m = np.max(draws, axis=0, keepdims=True)
    lppd_i = np.squeeze(m + np.log(np.mean(np.exp(draws - m), axis=0, keepdims=True)), axis=0)
    p_i = np.var(draws, axis=0, ddof=1)
    lppd = float(np.sum(lppd_i))
    p_waic = float(np.sum(p_i))
    elpd = float(lppd - p_waic)
    waic = float(-2.0 * elpd)

    assert out.lppd == pytest.approx(lppd)
    assert out.p_waic == pytest.approx(p_waic)
    assert out.elpd_waic == pytest.approx(elpd)
    assert out.waic == pytest.approx(waic)
    assert out.n_obs == 3


def test_compute_waic_from_log_lik_draws_respects_draw_axis() -> None:
    """draw_axis should work for non-leading draw dimensions."""
    base = np.array(
        [
            [-1.0, -2.0],
            [-1.1, -2.1],
            [-0.9, -1.9],
        ],
        dtype=float,
    )  # draws x obs

    out0 = compute_waic_from_log_lik_draws(base, draw_axis=0)
    out1 = compute_waic_from_log_lik_draws(base.T, draw_axis=1)

    assert out0.waic == pytest.approx(out1.waic)
    assert out0.elpd_waic == pytest.approx(out1.elpd_waic)
    assert out0.p_waic == pytest.approx(out1.p_waic)
    assert out0.n_obs == out1.n_obs == 2


def test_compute_waic_from_log_lik_draws_rejects_invalid_inputs() -> None:
    """Invalid shapes and non-finite values should raise clear errors."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        _ = compute_waic_from_log_lik_draws(np.array([1.0, 2.0, 3.0]))

    bad = np.array([[0.0, np.nan], [0.0, -1.0]], dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        _ = compute_waic_from_log_lik_draws(bad)
