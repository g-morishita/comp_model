"""Tests for model recovery selection criteria."""

from __future__ import annotations

import math

import pytest

from comp_model_impl.recovery.model.criteria import (
    AICCriterion,
    BICCriterion,
    LogLikelihoodCriterion,
    PSISLOOCriterion,
    WAICCriterion,
    get_criterion,
)


def test_loglikelihood_criterion() -> None:
    """Log-likelihood criterion should return ll and prefer larger values."""
    c = LogLikelihoodCriterion()
    assert c.score(ll=-12.5, k=3, n_obs=20) == pytest.approx(-12.5)
    assert c.higher_is_better() is True


def test_aic_criterion() -> None:
    """AIC criterion should match the 2k - 2ll definition."""
    c = AICCriterion()
    assert c.score(ll=-10.0, k=4, n_obs=100) == pytest.approx(28.0)
    assert c.higher_is_better() is False


def test_bic_criterion() -> None:
    """BIC criterion should match the k*log(n) - 2ll definition."""
    c = BICCriterion()
    expected = 3 * math.log(50) - 2 * (-7.0)
    assert c.score(ll=-7.0, k=3, n_obs=50) == pytest.approx(expected)
    assert c.higher_is_better() is False


def test_waic_criterion() -> None:
    """WAIC criterion should return WAIC and prefer lower values."""
    c = WAICCriterion()
    assert c.score(ll=-10.0, k=4, n_obs=100, waic=123.4) == pytest.approx(123.4)
    assert c.score(ll=-10.0, k=4, n_obs=100, waic=None) == pytest.approx(float("inf"))
    assert c.higher_is_better() is False


def test_psis_loo_criterion() -> None:
    """PSIS-LOO criterion should return LOOIC and prefer lower values."""
    c = PSISLOOCriterion()
    assert c.score(ll=-10.0, k=4, n_obs=100, looic=101.2) == pytest.approx(101.2)
    assert c.score(ll=-10.0, k=4, n_obs=100, looic=None) == pytest.approx(float("inf"))
    assert c.higher_is_better() is False


def test_get_criterion_aliases_and_errors() -> None:
    """Criterion factory should accept aliases and reject unknown names."""
    assert get_criterion("ll").name == "loglike"
    assert get_criterion("loglik").name == "loglike"
    assert get_criterion("log_likelihood").name == "loglike"
    assert get_criterion("aic").name == "aic"
    assert get_criterion("bic").name == "bic"
    assert get_criterion("waic").name == "waic"
    assert get_criterion("psis_loo").name == "psis_loo"
    assert get_criterion("loo").name == "psis_loo"

    with pytest.raises(ValueError, match="Unknown criterion"):
        _ = get_criterion("not_a_criterion")
