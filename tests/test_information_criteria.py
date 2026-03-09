"""Tests for AIC/BIC helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.analysis.information_criteria import aic, bic


def test_aic_matches_closed_form() -> None:
    """AIC should match the standard closed-form expression."""

    assert aic(log_likelihood=-10.0, n_parameters=3) == pytest.approx(26.0)


def test_bic_matches_closed_form() -> None:
    """BIC should match the standard closed-form expression."""

    expected = math.log(100.0) * 3.0 - 2.0 * -10.0
    assert bic(log_likelihood=-10.0, n_parameters=3, n_observations=100) == pytest.approx(expected)


def test_bic_rejects_non_positive_observation_count() -> None:
    """BIC should reject invalid observation counts."""

    with pytest.raises(ValueError, match="n_observations must be > 0"):
        bic(log_likelihood=-10.0, n_parameters=3, n_observations=0)
