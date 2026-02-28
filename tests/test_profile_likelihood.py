"""Tests for profile-likelihood analysis helpers."""

from __future__ import annotations

import pytest

from comp_model.analysis import profile_likelihood_1d, profile_likelihood_2d


def test_profile_likelihood_1d_finds_quadratic_peak() -> None:
    """1D profile should recover the peak on the supplied grid."""

    result = profile_likelihood_1d(
        parameter_name="alpha",
        grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        evaluate_log_likelihood=lambda alpha: -((alpha - 0.75) ** 2),
    )

    assert result.parameter_name == "alpha"
    assert result.best_value == pytest.approx(0.75)
    assert result.best_log_likelihood == pytest.approx(0.0)
    assert len(result.points) == 5


def test_profile_likelihood_2d_finds_quadratic_peak() -> None:
    """2D profile should recover the best point on the Cartesian grid."""

    result = profile_likelihood_2d(
        x_parameter_name="alpha",
        y_parameter_name="beta",
        x_grid=[0.0, 0.5, 1.0],
        y_grid=[0.0, 1.0, 2.0],
        evaluate_log_likelihood=lambda alpha, beta: -((alpha - 0.5) ** 2) - ((beta - 1.0) ** 2),
    )

    assert result.best_x_value == pytest.approx(0.5)
    assert result.best_y_value == pytest.approx(1.0)
    assert result.best_log_likelihood == pytest.approx(0.0)
    assert result.grid_shape == (3, 3)
    assert len(result.points) == 9


def test_profile_likelihood_rejects_empty_grids() -> None:
    """Profile helpers should fail fast on empty grid inputs."""

    with pytest.raises(ValueError, match="grid must not be empty"):
        profile_likelihood_1d(
            parameter_name="alpha",
            grid=[],
            evaluate_log_likelihood=lambda alpha: -alpha**2,
        )

    with pytest.raises(ValueError, match="x_grid must not be empty"):
        profile_likelihood_2d(
            x_parameter_name="alpha",
            y_parameter_name="beta",
            x_grid=[],
            y_grid=[0.0],
            evaluate_log_likelihood=lambda alpha, beta: -(alpha**2 + beta**2),
        )

    with pytest.raises(ValueError, match="y_grid must not be empty"):
        profile_likelihood_2d(
            x_parameter_name="alpha",
            y_parameter_name="beta",
            x_grid=[0.0],
            y_grid=[],
            evaluate_log_likelihood=lambda alpha, beta: -(alpha**2 + beta**2),
        )
