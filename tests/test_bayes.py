"""Tests for Bayesian MAP inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.inference import (
    ActionReplayLikelihood,
    IndependentPriorProgram,
    MapFitSpec,
    ScipyMapBayesEstimator,
    TransformedScipyMapBayesEstimator,
    fit_map_model,
    fit_map_model_from_registry,
    normal_log_prior,
    uniform_log_prior,
)
from comp_model.inference.transforms import unit_interval_logit_transform
from comp_model.plugins import build_default_registry
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


@dataclass
class FixedChoiceModel:
    """Toy model with one free right-choice probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op reset."""

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[int],
    ) -> dict[int, float]:
        """Return fixed Bernoulli action probabilities."""

        assert context.available_actions == (0, 1)
        return {0: 1.0 - self.p_right, 1: self.p_right}

    def update(
        self,
        observation: Any,
        action: int,
        outcome: Any,
        *,
        context: DecisionContext[int],
    ) -> None:
        """No-op update."""


def test_scipy_map_estimator_applies_prior_shrinkage() -> None:
    """MAP fit should shift estimates toward informative priors."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.8),
        config=SimulationConfig(n_trials=20, seed=2),
    )

    estimator = ScipyMapBayesEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {
                "p_right": normal_log_prior(mean=0.2, std=0.05),
            }
        ),
    )

    fit = estimator.fit(
        trace,
        initial_params={"p_right": 0.5},
        bounds={"p_right": (0.0, 1.0)},
    )

    # The data-generating p_right is 0.8, but strong prior at 0.2 pulls MAP down.
    assert 0.25 <= fit.map_params["p_right"] <= 0.45
    assert fit.map_candidate.log_posterior == pytest.approx(
        fit.map_candidate.log_likelihood + fit.map_candidate.log_prior
    )


def test_transformed_scipy_map_estimator_works_with_unit_interval_transform() -> None:
    """Transformed MAP estimator should fit constrained probabilities."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.75),
        config=SimulationConfig(n_trials=60, seed=9),
    )

    estimator = TransformedScipyMapBayesEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {
                "p_right": uniform_log_prior(lower=0.0, upper=1.0),
            }
        ),
        transforms={"p_right": unit_interval_logit_transform()},
    )

    fit = estimator.fit(
        trace,
        initial_params={"p_right": 0.5},
    )

    assert 0.0 < fit.map_params["p_right"] < 1.0
    assert fit.scipy_diagnostics is not None
    assert fit.scipy_diagnostics.method == "L-BFGS-B"


def test_independent_prior_program_requires_all_parameters_by_default() -> None:
    """Missing priors should fail fast when require_all=True."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.6),
        config=SimulationConfig(n_trials=10, seed=1),
    )

    estimator = ScipyMapBayesEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram({}, require_all=True),
    )

    with pytest.raises(ValueError, match="missing priors"):
        estimator.fit(
            trace,
            initial_params={"p_right": 0.5},
            bounds={"p_right": (0.0, 1.0)},
        )


def test_fit_map_model_wrapper_accepts_block_data() -> None:
    """fit_map_model should run MAP fitting on BlockData datasets."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.7),
        config=SimulationConfig(n_trials=50, seed=12),
    )

    from comp_model.core.data import BlockData

    block = BlockData(block_id="b0", event_trace=trace)
    fit = fit_map_model(
        block,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {"p_right": uniform_log_prior(lower=0.0, upper=1.0)}
        ),
        fit_spec=MapFitSpec(
            estimator_type="scipy_map",
            initial_params={"p_right": 0.5},
            bounds={"p_right": (0.0, 1.0)},
        ),
    )

    assert 0.0 <= fit.map_params["p_right"] <= 1.0
    assert fit.map_candidate.log_posterior == pytest.approx(
        fit.map_candidate.log_likelihood + fit.map_candidate.log_prior
    )


def test_fit_map_model_from_registry_runs_end_to_end() -> None:
    """Registry-based MAP helper should fit built-in model components."""

    registry = build_default_registry()
    generating_model_params = {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    trace = run_episode(
        problem=StationaryBanditProblem([0.2, 0.8]),
        model=registry.create_model(
            "asocial_state_q_value_softmax",
            **generating_model_params,
        ),
        config=SimulationConfig(n_trials=40, seed=20),
    )

    fit = fit_map_model_from_registry(
        trace,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=IndependentPriorProgram(
            {
                "alpha": uniform_log_prior(lower=0.0, upper=1.0),
                "beta": uniform_log_prior(lower=0.0, upper=20.0),
                "initial_value": normal_log_prior(mean=0.0, std=1.0),
            }
        ),
        fit_spec=MapFitSpec(
            estimator_type="scipy_map",
            initial_params={"alpha": 0.5, "beta": 1.0, "initial_value": 0.0},
            bounds={
                "alpha": (0.0, 1.0),
                "beta": (0.0, 20.0),
                "initial_value": (None, None),
            },
        ),
        registry=registry,
    )

    assert set(fit.map_params) == {"alpha", "beta", "initial_value"}
