"""Tests for legacy Bayesian MAP API behavior."""

from __future__ import annotations

import pytest

from comp_model.inference.bayes import (
    IndependentPriorProgram,
    MapFitSpec,
    ScipyMapBayesEstimator,
    TransformedScipyMapBayesEstimator,
    fit_map_model,
    fit_map_model_from_registry,
    normal_log_prior,
    uniform_log_prior,
)


def test_independent_prior_program_requires_all_parameters_by_default() -> None:
    """Missing priors should fail fast when ``require_all=True``."""

    prior = IndependentPriorProgram({}, require_all=True)
    with pytest.raises(ValueError, match="missing priors"):
        prior.log_prior({"alpha": 0.2})


def test_independent_prior_program_allows_subset_when_configured() -> None:
    """Independent prior program should permit subset priors when disabled."""

    prior = IndependentPriorProgram(
        {"alpha": uniform_log_prior(lower=0.0, upper=1.0)},
        require_all=False,
    )
    logp = prior.log_prior({"alpha": 0.2, "beta": 3.0})
    assert logp == pytest.approx(0.0)


def test_removed_scipy_map_estimator_constructors_raise() -> None:
    """Removed SciPy Bayesian estimator constructors should raise immediately."""

    with pytest.raises(RuntimeError, match="removed"):
        ScipyMapBayesEstimator(
            likelihood_program=None,  # type: ignore[arg-type]
            model_factory=lambda params: object(),
            prior_program=IndependentPriorProgram(
                {"x": normal_log_prior(mean=0.0, std=1.0)},
                require_all=False,
            ),
        )

    with pytest.raises(RuntimeError, match="removed"):
        TransformedScipyMapBayesEstimator(
            likelihood_program=None,  # type: ignore[arg-type]
            model_factory=lambda params: object(),
            prior_program=IndependentPriorProgram(
                {"x": normal_log_prior(mean=0.0, std=1.0)},
                require_all=False,
            ),
        )


def test_removed_map_helpers_raise_runtime_error() -> None:
    """Legacy MAP helper entry points should fail with migration guidance."""

    fit_spec = MapFitSpec(
        estimator_type="scipy_map",
        initial_params={"alpha": 0.5},
        bounds={"alpha": (0.0, 1.0)},
    )
    prior = IndependentPriorProgram(
        {"alpha": uniform_log_prior(lower=0.0, upper=1.0)},
        require_all=False,
    )

    with pytest.raises(RuntimeError, match="no longer supported"):
        fit_map_model(
            [],
            model_factory=lambda params: object(),
            prior_program=prior,
            fit_spec=fit_spec,
        )

    with pytest.raises(RuntimeError, match="no longer supported"):
        fit_map_model_from_registry(
            [],
            model_component_id="asocial_state_q_value_softmax",
            prior_program=prior,
            fit_spec=fit_spec,
        )

